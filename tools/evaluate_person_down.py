#!/usr/bin/env python3
"""Replay unchanged AICOS person-down detectors with errors, onset timing and hashes.

Requires existing local weights; never downloads models. Optional .capture.json
sidecars preserve receipt timing, and .labels.json sidecars annotate onset frames.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from eval_support import clip_metadata, frame_time, latency, sha256

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'}
CLOCK_BASE = 1_000_000.0


def discover_dataset(root):
    return [(path, positive) for directory, positive in (('positive', True), ('negative', False))
            for path in sorted((root / directory).rglob('*'))
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS]


def metrics(results):
    all_rows = list(results)
    rows = [r for r in all_rows if not r.get('error')]
    tp = sum(r['expected_positive'] and r['alerted'] for r in rows)
    fn = sum(r['expected_positive'] and not r['alerted'] for r in rows)
    fp = sum(not r['expected_positive'] and r['alerted'] for r in rows)
    tn = sum(not r['expected_positive'] and not r['alerted'] for r in rows)
    div = lambda a, b: a / b if b else None
    times = [r['onset_to_alert_seconds'] for r in rows if r['expected_positive'] and r['alerted']
             and r.get('onset_to_alert_seconds') is not None]
    starts = [r['first_alert_seconds'] for r in rows if r['expected_positive'] and r['alerted']]
    return {'clips': len(all_rows), 'valid_clips': len(rows), 'error_clips': len(all_rows) - len(rows),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': div(tp, tp + fp), 'recall': div(tp, tp + fn),
            'specificity': div(tn, tn + fp), 'false_positive_rate': div(fp, fp + tn),
            'f1': div(2 * tp, 2 * tp + fp + fn),
            'early_alert_clips': sum(bool(r.get('early_alerts')) for r in rows),
            'ambiguous_alert_clips': sum(bool(r.get('ambiguous_alerts')) for r in rows),
            'interval_labeled_positive_clips': sum(r['expected_positive'] and bool(r.get('onset_interval_seconds')) for r in rows),
            'positive_clips_without_onset': sum(r['expected_positive'] and r.get('onset_seconds') is None for r in rows),
            'latency_samples': len(times), 'mean_time_to_alert_seconds': div(sum(times), len(times)),
            'mean_clip_start_to_alert_seconds': div(sum(starts), len(starts))}


def merged(base, override):
    out = deepcopy(base)
    for key, value in override.items():
        out[key] = merged(out.get(key, {}), value) if isinstance(value, dict) else deepcopy(value)
    return out


def effective_settings(args):
    sys.path.insert(0, str(ROOT / 'edge'))
    from detectors import DEFAULTS  # Model engine import is lazy.
    yolo, camera, resize = deepcopy(DEFAULTS), {}, 640
    if args.config:
        import yaml
        document = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        cameras = document.get('cameras', [])
        choices = [c for c in cameras if c.get('id') == args.camera_id] if args.camera_id else cameras
        if len(choices) != 1:
            raise ValueError('Select exactly one configured camera with --camera-id')
        camera = choices[0]
        settings = document.get('settings', {})
        resize = int(settings.get('resize_width', 640))
        yolo = merged(yolo, settings.get('detectors', {}).get('yolo', {}))
        yolo = merged(yolo, camera.get('detectors', {}).get('yolo', {}))
    for key in ('model', 'conf', 'imgsz'):
        value = getattr(args, key)
        if value is not None:
            yolo[key] = value
    if args.seconds is not None:
        yolo['person_down']['seconds'] = args.seconds
    for key in ('crowd', 'fight', 'panic', 'intrusion', 'loitering', 'abandoned_object', 'snatch', 'vehicles'):
        yolo[key] = {**yolo.get(key, {}), 'enabled': False}
    yolo['person_down']['enabled'] = True
    model = Path(yolo['model']).expanduser().resolve()
    if args.mode in ('heuristic', 'both', 'pose-confirmed'):
        if not model.is_file():
            raise ValueError('Pose weights must already exist locally. Set --model to their path; automatic download is disabled.')
        yolo['model'] = str(model)
    fall = {'weights': str(args.fall_weights.resolve()) if args.fall_weights else '',
            'conf': args.fall_conf, 'imgsz': args.fall_imgsz, 'seconds': args.fall_seconds,
            'cooldown_seconds': 999999.0}
    if args.mode in ('trained', 'both', 'confirmed') and (not args.fall_weights or not args.fall_weights.is_file()):
        raise ValueError('Trained mode requires existing local --fall-weights; automatic download is disabled.')
    rotation = int(camera.get('rotate', 0)) % 360
    if rotation not in (0, 90, 180, 270) or resize <= 0:
        raise ValueError('Invalid camera rotation or resize_width')
    effective = {'yolo': yolo, 'fall': fall, 'resize_width': args.resize_width or resize,
                 'rotate': rotation, 'camera_id': camera.get('id'), 'every': args.every}
    if args.mode == 'pose-confirmed':
        from pose_confirmed_person_down import PoseConfirmedPersonDownDetector
        effective['pose_confirmation'] = PoseConfirmedPersonDownDetector({
            'model': yolo['model'], 'conf': yolo['conf'], 'imgsz': yolo['imgsz'],
            'seconds': yolo['person_down']['seconds']}).cfg
    return effective


def make_detectors(args):
    detectors = []
    if args.mode in ('heuristic', 'both'):
        from detectors import YoloBehaviorSuite
        detectors.append(YoloBehaviorSuite(camera={}, config=args.effective['yolo']))
    if args.mode in ('trained', 'both'):
        from falldet import FallModelDetector
        detectors.append(FallModelDetector(config=args.effective['fall']))
    if args.mode == 'confirmed':
        from confirmed_person_down import ConfirmedPersonDownDetector
        detectors.append(ConfirmedPersonDownDetector(config=args.effective['fall']))
    if args.mode == 'pose-confirmed':
        from pose_confirmed_person_down import PoseConfirmedPersonDownDetector
        detectors.append(PoseConfirmedPersonDownDetector(args.effective['pose_confirmation']))
    return detectors


def evaluate_video(path, args):
    import cv2
    row = {'file': str(path), 'video_sha256': sha256(path), 'alerted': None}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('cannot open video')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('invalid FPS; refusing to invent playback timing')
        total_value = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(total_value) or total_value < 1:
            raise ValueError('invalid or unknown frame count')
        total = int(total_value)
        meta = clip_metadata(path, fps, total)
        row.update({k: v for k, v in meta.items() if k != 'frame_times_seconds'})
        row.update(fps=fps, frames_total=total, encoded_duration_seconds=total / fps)
        count = analyzed = 0
        first_alert = None
        alerts, early, ambiguous, detectors = [], [], [], None
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            index = count
            count += 1
            if count > total:
                raise ValueError('decoded frame count exceeds timing metadata')
            # Decode the complete clip even after alerting to catch corrupt tails.
            if index % args.every or (args.stop_after_first and first_alert is not None):
                continue
            if detectors is None:
                detectors = make_detectors(args)
            analyzed += 1
            vt = frame_time(meta, index, fps)
            rotation = args.effective['rotate']
            if rotation:
                frame = cv2.rotate(frame, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                          270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
            h, w = frame.shape[:2]
            width = args.effective['resize_width']
            if w > width:
                frame = cv2.resize(frame, (width, max(1, int(h * width / w))))
            for detector in detectors:
                found = detector.detect(frame, None, now=CLOCK_BASE + vt)
                if getattr(detector, '_disabled', False):
                    raise RuntimeError('Detector disabled itself during replay; see stderr for model-load/inference error')
                for event in found:
                    if 'person down' not in str(event.get('type', '')).lower():
                        continue
                    entry = {'time_seconds': vt, 'frame': index, 'severity': event.get('severity'),
                             'confidence': event.get('confidence'), 'extra': event.get('extra') or {}}
                    cutoff = meta['confirmed_down_seconds']
                    if cutoff is not None and vt < cutoff:
                        interval = meta.get('onset_interval_seconds')
                        (ambiguous if interval and vt >= interval[0] else early).append(entry)
                        continue
                    alerts.append(entry)
                    if first_alert is None:
                        first_alert = vt
        if count == 0:
            raise ValueError('video contains no decodable frames')
        if count != total:
            raise ValueError(f'truncated video: decoded {count} of {total} frames')
        row.update(frames_decoded=count, frames_analyzed=analyzed, alerted=first_alert is not None,
                   first_alert_seconds=first_alert, onset_to_alert_seconds=latency(first_alert, meta['onset_seconds']),
                   alerts=alerts, early_alerts=early, ambiguous_alerts=ambiguous)
        if first_alert is not None and meta.get('onset_interval_seconds'):
            lower, upper = meta['onset_interval_seconds']
            row['onset_to_alert_interval_seconds'] = [first_alert - upper, first_alert - lower]
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, cv2.error) as error:
        row.update(error=str(error), alerted=None)
    finally:
        cap.release()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset', type=Path)
    ap.add_argument('--mode', choices=('heuristic', 'trained', 'both', 'confirmed', 'pose-confirmed'), default='heuristic')
    ap.add_argument('--config', type=Path)
    ap.add_argument('--camera-id')
    ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--seconds', type=float)
    ap.add_argument('--model', help='existing local pose weights; overrides YAML/default')
    ap.add_argument('--conf', type=float)
    ap.add_argument('--imgsz', type=int)
    ap.add_argument('--resize-width', type=int)
    ap.add_argument('--fall-weights', type=Path)
    ap.add_argument('--fall-conf', type=float, default=0.35)
    ap.add_argument('--fall-imgsz', type=int, default=640)
    ap.add_argument('--fall-seconds', type=float, default=3)
    ap.add_argument('--stop-after-first', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--json-out', type=Path)
    args = ap.parse_args()
    if args.every < 1 or (args.resize_width is not None and args.resize_width < 1):
        ap.error('frame sampling and resize width must be positive')
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        ap.error('persistence seconds must be positive and finite')
    if args.json_out and args.json_out.exists():
        ap.error('result file already exists; choose a new filename to preserve earlier results')
    pairs = discover_dataset(args.dataset)
    if not pairs:
        ap.error('No videos found in positive/ and negative/')
    try:
        args.effective = effective_settings(args)
    except (ValueError, OSError) as error:
        ap.error(str(error))
    rows = []
    for path, positive in pairs:
        row = evaluate_video(path, args)
        if not positive and (row.get('onset_seconds') is not None or row.get('onset_interval_seconds')):
            row.update(error='Negative clip must not carry a person-down onset annotation', alerted=None)
        row['expected_positive'] = positive
        row['relative_file'] = path.relative_to(args.dataset).as_posix()
        rows.append(row)
        print(('ERROR' if row.get('error') else 'ALERT' if row['alerted'] else 'quiet'), path.name)
    summary = metrics(rows)
    files = [Path(__file__), ROOT / 'tools/eval_support.py', ROOT / 'edge/detectors.py', ROOT / 'VERSION']
    if args.config:
        files.append(args.config)
    if args.mode in ('heuristic', 'both', 'pose-confirmed'):
        files.append(Path(args.effective['yolo']['model']))
    if args.mode == 'pose-confirmed':
        files.extend([ROOT / 'edge/pose_confirmed_person_down.py', ROOT / 'edge/temporal_posture.py'])
    if args.mode in ('trained', 'both', 'confirmed'):
        files.extend([args.fall_weights, ROOT / 'edge/falldet.py'])
    if args.mode == 'confirmed':
        files.append(ROOT / 'edge/confirmed_person_down.py')
    output = {'schema_version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'mode': args.mode, 'settings': args.effective, 'stop_after_first': args.stop_after_first,
              'python': platform.python_version(),
              'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
              'source_and_model_sha256': {str(p): sha256(p) for p in files},
              'summary': summary, 'results': rows,
              'latency_definition': 'first qualifying alert minus annotated person-down onset; missing onset excluded',
              'sampling_note': 'Offline replay sampling, not a live-camera throughput benchmark.'}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open('x', encoding='utf-8') as stream:
            json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
    return 1 if summary['error_clips'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
