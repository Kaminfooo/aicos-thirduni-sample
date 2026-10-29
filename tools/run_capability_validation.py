"""Sequential, label-aware local screening. Never tunes models or enables live profiles."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.runtime/capability-validation-20261027'
DATA = ROOT / 'data/capability_validation_20261027'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    inventory = [
        ('person_down', 'Pose confirmation plus temporal persistence', 'judge workspace; candidate camera profile', 'reference results retained; field validation incomplete'),
        ('fire', 'PersistentHazardDetector + fire/smoke model', 'offline staged only', 'new labeled screening planned'),
        ('smoke', 'PersistentHazardDetector + fire/smoke model', 'offline staged only', 'new labeled screening planned'),
        ('violence_model', 'PersistentHazardDetector + violence model; legacy ViolenceDetector also present', 'offline staged; not judge workspace', 'new labeled screening planned'),
        ('fight_heuristic', 'YoloBehaviorSuite._fight', 'configurable legacy behavior', 'controlled-rule tests; no validated violence recall'),
        ('crowd', 'instant legacy count or SiteBehaviorSuite count/region/dwell', 'staged site-aware profile', 'requires scene-specific occupancy labels and thresholds'),
        ('panic', 'multiple fast person tracks', 'configurable legacy heuristic', 'does not establish panic or intent'),
        ('intrusion', 'person feet inside configured polygon', 'configurable legacy rule', 'zone configuration and real model tracking required'),
        ('loitering', 'track persistence inside configured polygon', 'configurable legacy rule', 'configured duration; no suspicious-intent inference'),
        ('unattended_object', 'bag classes + ownership distance/time', 'opt-in legacy rule', 'requires compatible object model and representative positive clips'),
        ('snatch', 'person contact then rapid separation', 'opt-in experimental heuristic', 'does not establish theft; no validated positive dataset'),
        ('stopped_vehicle', 'tracked vehicle stillness in road region', 'offline staged', 'requires lane/parking geometry and duration labels'),
        ('congestion', 'stationary vehicle count', 'offline staged', 'requires road region and calibrated count'),
        ('wrong_way', 'vehicle motion against configured image direction', 'offline staged', 'requires explicit direction; camera motion is a confounder'),
        ('motion', 'image difference threshold', 'legacy demo fallback', 'motion only, not semantic incident detection'),
        ('camera_integrity_health', 'view-baseline changes, freshness and network/capture metrics', 'separate reliability subsystem', 'not a semantic incident/model accuracy result')]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'DETECTOR-INVENTORY.json').write_text(json.dumps({'capabilities': [dict(zip(('id','implementation','activation','validation'), x)) for x in inventory],
        'judge_workspace': 'person_down only; unchanged by this evaluation',
        'production_or_public_activation_changed': False}, indent=2), encoding='utf-8')
    plan = {'created_before_new_inference': True, 'sequence': ['fire_smoke', 'violence', 'crowd_site', 'traffic', 'other_rule_coverage'],
        'acceptance_screen': 'Every selected positive must alert its labelled class; selected negatives must not alert that class. Report failures without tuning.',
        'measurement': 'clip-level TP/FN/FP/TN, raw event counts, decoded duration, false alerts per negative camera-hour; no onset latency without onset labels',
        'sample_fps': 3, 'resize_width': 480, 'frozen_hazard_settings': {'conf': .5, 'imgsz': 640, 'seconds': 2, 'min_observations': 3, 'gap_tolerance': 1, 'cooldown_seconds': 60},
        'unknown_training_overlap': True, 'not_field_accuracy': True,
        'boundaries': 'No backend posting, camera/network stream, notification, model install, production threshold or live-profile changes',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'edge/detectors.py',ROOT / 'edge/persistent_hazards.py',ROOT / 'edge/site_behaviors.py',ROOT / 'edge/pose_confirmed_person_down.py']}}
    p = WORK / 'VALIDATION-PLAN.json'
    if not p.exists():
        p.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print('Detector inventory and predeclared screening plan saved.')


def cases(kind):
    if kind == 'fire_smoke':
        attribution = json.loads((DATA / 'firesense/subset/ATTRIBUTION.json').read_text())
        return [{'path': ROOT / row['file'], 'label': row['family'], 'positive': row['publisher_label'] == 'positive',
                 'source_sha256': row['sha256'], 'source_group': row['archive']} for row in attribution['files']]
    attribution = json.loads((DATA / 'airtlab/ATTRIBUTION.json').read_text())
    return [{'path': ROOT / row['file'], 'label': 'violence', 'positive': row['label'] == 'violent',
             'source_sha256': row['sha256'], 'source_group': 'AIRTLab cam1'} for row in attribution['files']]


def evaluate(kind):
    report_dir = WORK / (kind + '-frozen-screen')
    if report_dir.exists():
        raise RuntimeError('Report already exists; inspect it rather than rerunning or overwriting')
    selected = cases(kind)
    plan = json.loads((WORK / 'VALIDATION-PLAN.json').read_text())
    for name, expected in plan['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Detector source changed since predeclaration: ' + name)
    report_dir.mkdir()
    temp = WORK / 'tmp'
    temp.mkdir(exist_ok=True)
    os.environ.update(TEMP=str(temp), TMP=str(temp), YOLO_AUTOINSTALL='false',
        YOLO_CONFIG_DIR=str(WORK / 'ultralytics'), MPLCONFIGDIR=str(WORK / 'matplotlib'),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.path.insert(0, str(ROOT / 'edge'))
    import cv2
    import torch
    import math
    from persistent_hazards import PersistentHazardDetector, LocalLabelEngine
    torch.set_num_threads(2)
    cfg = dict(plan['frozen_hazard_settings'], kind=kind,
               weights=str(ROOT / 'models/pretrained' / ('fire-smoke-yolo26n.pt' if kind == 'fire_smoke' else 'violence-yolov8n.pt')))
    labels = ['fire', 'smoke'] if kind == 'fire_smoke' else ['violence']
    engine = LocalLabelEngine(cfg['weights'], labels, cfg['conf'], cfg['imgsz'])
    report = {'kind': kind, 'settings': cfg, 'model_sha256': sha(cfg['weights']),
              'source_sha256': plan['source_sha256'], 'cases': [], 'accuracy_not_established': True,
              'packages': {n: importlib.metadata.version(n) for n in ('ultralytics','torch','opencv-python')},
              'label_scope': 'Only the publisher-labelled class is scored; other class output is retained but unlabelled.'}
    started = time.monotonic()
    for index, case in enumerate(selected):
        path = case['path']
        if sha(path) != case['source_sha256']:
            raise RuntimeError('Video hash mismatch: ' + str(path))
        detector = PersistentHazardDetector(cfg, engine=engine)
        cap = cv2.VideoCapture(str(path))
        decoded = analyzed = 0
        previous, next_sample, first = -1, 0, None
        events, observations = [], []
        cost, clip_start = 0, time.monotonic()
        status, error = 'complete', None
        try:
            if not cap.isOpened():
                raise ValueError('Video codec is not supported by the installed OpenCV')
            while True:
                if time.monotonic() - clip_start > 240 or time.monotonic() - started > 1500:
                    raise TimeoutError('Bounded evaluation time exceeded; remaining cases stay untested')
                ok, frame = cap.read()
                if not ok:
                    break
                stamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not math.isfinite(stamp) or stamp < previous:
                    raise ValueError('Decoded presentation timestamps are invalid/decreasing')
                previous = stamp
                first = stamp if first is None else first
                decoded += 1
                if stamp < next_sample:
                    continue
                next_sample = stamp + 1 / 3
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (480, max(1, int(h * 480 / w))))
                before = time.monotonic()
                found = detector.detect(frame, now=1_000_000 + stamp)
                cost += time.monotonic() - before
                analyzed += 1
                if detector._disabled:
                    raise RuntimeError('Model disabled; not a negative result')
                events.extend({'seconds': stamp, 'event': e} for e in found)
                observations.append(dict(seconds=stamp, **{k: v for k, v in detector.last_decision.items() if k != 'time'}))
            if decoded < 2 or previous <= (first or 0):
                raise ValueError('Not enough decoded frames/timeline')
        except Exception as exc:
            status, error = 'error', str(exc)
        finally:
            cap.release()
        labelled_events = [e for e in events if e['event']['extra']['label'] == case['label']]
        predicted = bool(labelled_events)
        outcome = ('TP' if predicted else 'FN') if case['positive'] else ('FP' if predicted else 'TN')
        row = {'file': str(path.relative_to(ROOT)), 'sha256': case['source_sha256'], 'source_group': case['source_group'],
               'label': case['label'], 'expected_positive': case['positive'], 'status': status, 'error': error,
               'decoded_frames': decoded, 'analysed_frames': analyzed, 'decoded_seconds': max(0, previous - (first or 0)),
               'events': events, 'labelled_event_count': len(labelled_events), 'outcome': outcome if status == 'complete' else 'ERROR',
               'mean_inference_seconds': cost / analyzed if analyzed else None}
        report['cases'].append(row)
        (report_dir / (f'{index:02d}-' + path.stem + '.trace.json')).write_text(json.dumps(observations), encoding='utf-8')
        (report_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"{index+1}/{len(selected)} {path.name}: {row['outcome']} ({analyzed} samples, {len(labelled_events)} labelled events)", flush=True)
        if status == 'error':
            break
    summary = {}
    for label in labels:
        rows = [r for r in report['cases'] if r['label'] == label and r['status'] == 'complete']
        counts = {key: sum(r['outcome'] == key for r in rows) for key in ('TP','FN','FP','TN')}
        negatives = [r for r in rows if not r['expected_positive']]
        duration = sum(r['decoded_seconds'] for r in negatives)
        false_events = sum(r['labelled_event_count'] for r in negatives)
        summary[label] = dict(counts, negative_seconds=duration, false_events=false_events,
                             false_events_per_negative_hour=false_events * 3600 / duration if duration else None)
    report.update(summary=summary, complete_cases=len(report['cases']), requested_cases=len(selected),
                  process_seconds=round(time.monotonic() - started, 2),
                  screening_passed=len(report['cases']) == len(selected) and all(r['outcome'] in ('TP','TN') for r in report['cases']))
    (report_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    code = 2 if any(r['status'] == 'error' for r in report['cases']) else 0 if report['screening_passed'] else 1
    (report_dir / 'screening-exit.txt').write_text(str(code))
    print(json.dumps({'summary': summary, 'screening_passed': report['screening_passed'], 'exit': code}), flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','fire_smoke','violence'])
    action = parser.parse_args().action
    if action == 'prepare':
        prepare()
    else:
        sys.exit(evaluate(action))
