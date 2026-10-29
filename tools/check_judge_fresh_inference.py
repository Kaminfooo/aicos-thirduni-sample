"""Bounded local acceptance run; does not connect a physical camera or network feed."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from fastapi.testclient import TestClient
from deploy.judge_workspace import create_app, isolated

OUT = ROOT / '.runtime/judge-workspace-validation-20261027'
OUT.mkdir(parents=True, exist_ok=True)


def encode(frame):
    h, w = frame.shape[:2]
    scale = min(480 / w, 480 / h, 1)
    frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
    ok, data = cv2.imencode('.png', frame)
    if not ok:
        raise RuntimeError('Lossless PNG encoding failed')
    return base64.b64encode(data).decode()


def main():
    report = {'status': 'running', 'runtime': str(isolated.BOOT), 'cases': [],
              'physical_camera_tested': False, 'browser_permission_tested': False,
              'source': 'existing reference YOLO26n pose; fresh lossless PNG frame inference through judge API'}
    start = time.monotonic()
    with TestClient(create_app()) as client:
        for mode, name, expected in [('scenario', 'controlled-fall.mp4', True), ('file', 'walking.mp4', False)]:
            client.post('/judge/api/run', json={'mode': mode}).raise_for_status()
            path = isolated.ASSETS / name
            capture = cv2.VideoCapture(str(path))
            next_sample, frames, alert, alert_seconds = 0, 0, False, None
            try:
                while True:
                    if time.monotonic() - start > 240:
                        raise RuntimeError('Acceptance test exceeded its four-minute budget')
                    ok, frame = capture.read()
                    if not ok:
                        break
                    seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if seconds < next_sample:
                        continue
                    next_sample = seconds + 1 / 3
                    response = client.post('/judge/api/frame', json={'image': encode(frame), 'seconds': seconds})
                    if response.status_code != 200:
                        raise RuntimeError(str(response.status_code) + ' ' + response.text)
                    result = response.json()
                    frames += 1
                    if result['alert']:
                        alert, alert_seconds = True, result['seconds']
                        break
            finally:
                capture.release()
            client.post('/judge/api/finish', json={}).raise_for_status()
            case = {'input_mode': mode, 'video': name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'sampled_frames': frames, 'alert': alert, 'alert_seconds': alert_seconds,
                    'expected_demo_behavior': expected, 'passed': alert == expected}
            if alert:
                before = client.get('/judge/api/incident').json()
                assert before['verification']['ok'], before
                assert client.post('/judge/api/decision', json={'status': 'approved'}).status_code == 409
                client.post('/judge/api/review', json={'decision': 'accept', 'note': 'Controlled simulation; cause unknown. Fresh inference acceptance test.'}).raise_for_status()
                client.post('/judge/api/decision', json={'status': 'approved'}).raise_for_status()
                after = client.get('/judge/api/incident').json()
                case['signed_evidence_after_review_and_approval'] = after['verification']['ok']
                case['audit_actions'] = [r['action'] for r in after['audit']]
                case['media_statuses'] = [client.get('/judge/api/media/' + kind).status_code for kind in ('snapshot', 'clip')]
                case['passed'] &= case['signed_evidence_after_review_and_approval'] and case['media_statuses'] == [200, 200]
            report['cases'].append(case)
            print(json.dumps(case), flush=True)
            if not case['passed']:
                raise RuntimeError('Fresh inference acceptance case failed; no version changes attempted')
        client.post('/judge/api/run', json={'mode': 'camera'}).raise_for_status()
        blank = encode(np.zeros((270, 480, 3), dtype=np.uint8))
        first = client.post('/judge/api/frame', json={'image': blank, 'seconds': 60})
        first.raise_for_status()
        time.sleep(.35)
        second = client.post('/judge/api/frame', json={'image': blank, 'seconds': 61})
        second.raise_for_status()
        assert first.json()['seconds'] < .1 and second.json()['seconds'] < 5
        assert not first.json()['alert'] and not second.json()['alert']
        client.post('/judge/api/finish', json={}).raise_for_status()
        report['cases'].append({'input_mode': 'camera', 'source': 'two synthetic blank frames; no physical camera',
                                'passed': True, 'server_elapsed_time_enforced': True, 'alert': False})
    report.update(status='passed', elapsed_seconds=round(time.monotonic() - start, 2))
    (OUT / 'fresh-inference-result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Fresh inference acceptance passed; actual camera permission remains a manual check.', flush=True)


if __name__ == '__main__':
    main()
