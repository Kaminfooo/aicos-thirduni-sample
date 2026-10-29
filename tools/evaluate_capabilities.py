"""Offline profile replay with decoded PTS, local weights and preserved reports.

Reports detections and processing costs. It does not infer ground-truth labels,
post live events, connect cameras or establish deployment accuracy by itself.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--video', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-fps', type=float, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error('Choose a new project-local report directory')
    if not math.isfinite(args.sample_fps) or not 0 < args.sample_fps <= 15:
        parser.error('Sample FPS must be greater than zero and at most 15')
    for path in [args.profile, *args.video]:
        if not path.is_file():
            parser.error('Input file does not exist: ' + str(path))
    output.mkdir(parents=True)
    os.environ.update(TEMP=str(ROOT / '.runtime/tmp'), TMP=str(ROOT / '.runtime/tmp'),
        YOLO_CONFIG_DIR=str(ROOT / '.runtime/inference-setup/ultralytics'),
        MPLCONFIGDIR=str(ROOT / '.runtime/inference-setup/matplotlib'),
        YOLO_AUTOINSTALL='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.path.insert(0, str(ROOT / 'edge'))
    import cv2
    import torch
    from thirduni_profiles import build_profile
    torch.set_num_threads(2)
    profile = json.loads(args.profile.read_text())
    reports, engines = [], {}
    for video in args.video:
        detectors = build_profile(profile)
        for detector in detectors:
            if detector.name in engines:
                detector._engine = engines[detector.name]
        cap = cv2.VideoCapture(str(video))
        decoded = analyzed = 0
        previous, next_sample = -1, 0
        events, costs = [], {d.name: [] for d in detectors}
        result = {'video': str(video.resolve()), 'video_sha256': sha256(video),
                  'time_source': 'decoded presentation timestamps; FPS/frame-count hints are not trusted'}
        try:
            if not cap.isOpened():
                raise ValueError('Cannot decode video')
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not math.isfinite(timestamp) or timestamp < previous:
                    raise ValueError('Decoded presentation timestamps are invalid or decreasing')
                previous = timestamp
                decoded += 1
                if timestamp < next_sample:
                    continue
                next_sample = timestamp + 1 / args.sample_fps
                analyzed += 1
                h, w = frame.shape[:2]
                width = int(profile.get('settings', {}).get('resize_width', 480))
                frame = cv2.resize(frame, (width, max(1, int(h * width / w))))
                for detector in detectors:
                    start = time.perf_counter()
                    found = detector.detect(frame, None, now=1_000_000 + timestamp)
                    costs[detector.name].append(time.perf_counter() - start)
                    if getattr(detector, '_disabled', False):
                        raise RuntimeError('Detector disabled: ' + detector.name)
                    for event in found:
                        events.append({'frame': decoded - 1, 'seconds': timestamp,
                                       'detector': detector.name, 'event': event})
            if decoded < 2 or previous <= 0:
                raise ValueError('Too few frames or no usable presentation timeline')
            result.update(status='complete', decoded_frames=decoded, analyzed_frames=analyzed,
                last_pts_seconds=previous, events=events,
                mean_detector_seconds={name: sum(values) / len(values) if values else None for name, values in costs.items()})
        except Exception as error:
            result.update(status='error', error=str(error), decoded_frames=decoded, analyzed_frames=analyzed, events=events)
        finally:
            cap.release()
        reports.append(result)
        print(video.name, result['status'], len(events), 'events', flush=True)
        if result['status'] == 'error':
            break
        engines.update({d.name: d._engine for d in detectors})
    model_paths = {Path(cfg[key]) for cfg in profile.values() if isinstance(cfg, dict)
                   for key in ('model', 'weights') if cfg.get(key)}
    report = {'mode': 'offline_replay_only', 'accuracy_not_established': True,
        'profile': profile, 'profile_sha256': sha256(args.profile), 'sample_fps': args.sample_fps,
        'model_sha256': {str(p): sha256(p) for p in model_paths},
        'source_sha256': {name: sha256(ROOT / name) for name in
            ('tools/evaluate_capabilities.py', 'edge/thirduni_profiles.py', 'edge/pose_confirmed_person_down.py',
             'edge/temporal_posture.py', 'edge/persistent_hazards.py', 'edge/site_behaviors.py', 'edge/detectors.py')},
        'effective_detector_settings': [{'name': d.name, 'settings': d.cfg} for d in detectors],
        'packages': {n: importlib.metadata.version(n) for n in ('ultralytics', 'torch', 'opencv-python')},
        'reports': reports, 'notes': ['No labels or medical causes inferred.', 'No live incidents, notifications or approvals created.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    code = 1 if any(r['status'] == 'error' for r in reports) else 0
    (output / 'exit-code.txt').write_text(str(code))
    return code


if __name__ == '__main__':
    sys.exit(main())
