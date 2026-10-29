"""Replay the verified recorded incident into the separate loopback demo DB.

Uses recorded neural outputs and the reviewed temporal result; does not run a
camera/model, approve an incident or send notifications. Idempotent event UUID.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import threading
import time
import uuid
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime/thirduni-demo'), str(ROOT / 'edge')]
from manage_demo import request, existing, BASE
from rtsp_worker import CameraWorker, DEFAULT_SETTINGS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    temporal_path = ROOT / '.runtime/thirduni-completion-stage/temporal-recorded-result.json'
    temporal = json.loads(temporal_path.read_text())
    for source, expected in temporal['source_sha256'].items():
        if digest(ROOT / source) != expected:
            raise RuntimeError('Temporal detector changed; regenerate its recorded replay before seeding')
    source_predictions = ROOT / '.runtime/recorded-video-test/full-frame-20261024/frame-decisions.jsonl'
    assert digest(source_predictions) == temporal['input_sha256']
    video = ROOT / 'data/person_down_eval/diagnostic/live_1792880671799191000_c244ccd1.mp4'
    metadata = json.loads(video.with_suffix('.capture.json').read_text())
    assert digest(video) == metadata['video_sha256']
    recorded = temporal['events'][0]
    event = json.loads(json.dumps(recorded['event']))
    event['type'] = '[REPLAY] ' + event['type']
    event['event_uuid'] = str(uuid.uuid5(uuid.NAMESPACE_URL, metadata['video_sha256'] + temporal['source_sha256']['edge/pose_confirmed_person_down.py']))
    event['extra'].update(is_replay=True, recording_sha256=metadata['video_sha256'],
        capture_seconds=recorded['time_seconds'], source_frame=recorded['frame'],
        location_source='synthetic demonstration point; not the recording location',
        replay_method='Recorded real neural predictions plus reviewed temporal observations')
    frames, trigger, index = [], None, 0
    cap = cv2.VideoCapture(str(video))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        capture_time = metadata['frame_times_seconds'][index]
        if recorded['time_seconds'] - 4 <= capture_time <= recorded['time_seconds']:
            frames.append((capture_time, frame))
        if index == recorded['frame']:
            trigger = frame
        index += 1
    cap.release()
    assert index == metadata['frames_decoded'] and trigger is not None
    class Bridge:
        response = None
        def post_event(self, payload):
            status, raw = request('/api/events', payload, device=True)
            self.response = json.loads(raw)
            return status == 201
    bridge = Bridge()
    camera = {'id': 'REPLAY-THIRDUNI-01', 'name': 'Recorded ThirdUni demonstration',
              'district': 'DEMONSTRATION: synthetic map point', 'lat': 34.88803, 'lon': 3.490238}
    worker = CameraWorker(camera, {**DEFAULT_SETTINGS, 'integrity_enabled': False}, bridge, [], threading.Event())
    now = time.time()
    for timestamp, frame in frames:
        worker.buffer.add(now - (recorded['time_seconds'] - timestamp), frame)
    worker._emit(event, trigger)
    assert bridge.response and bridge.response['status'] == 'stored'
    event_id = bridge.response['id']
    values = existing.credentials()
    _, raw = request('/api/auth/login', {'username': values['username'], 'password': values['password']})
    token = json.loads(raw)['token']
    try:
        media = {}
        for label, expected_format in (('snapshot', 'JPEG'), ('clip', 'GIF')):
            status, raw = request(f'/api/events/{event_id}/{label}', token=token)
            image = Image.open(io.BytesIO(raw))
            assert status == 200 and image.format == expected_format
            image.verify()
            media[label] = {'bytes': len(raw), 'format': expected_format, 'sha256': hashlib.sha256(raw).hexdigest()}
        _, raw = request(f'/api/events/{event_id}/triage', token=token)
        drafts = json.loads(raw)['drafts']
        if drafts:
            draft = drafts[0]
        else:
            _, raw = request(f'/api/events/{event_id}/triage', {}, token)
            draft = json.loads(raw)
        assert draft['content']['production_recommendations_blocked']
        assert draft['content']['facts']['transition_observation'] == 'gradual_lowering_observed'
        _, raw = request(f'/api/events/{event_id}/evidence/verify', token=token)
        evidence = json.loads(raw)
        assert evidence['ok']
        _, raw = request('/api/evidence/chain/verify', token=token)
        assert json.loads(raw)['ok']
        _, raw = request('/api/events', token=token)
        stored = next(e for e in json.loads(raw)['events'] if e['id'] == event_id)
        assert stored['status'] == 'detected' and stored['extra']['is_replay']
    finally:
        request('/api/auth/logout', {}, token)
    result = {'event_id': event_id, 'draft_id': draft['id'], 'incident_status': stored['status'],
              'draft_status': draft['review_status'], 'transition': 'gradual_lowering_observed',
              'medical_cause': 'unknown', 'media': media, 'signed_evidence': evidence,
              'database_separate_from_live': True, 'camera_used': False, 'model_rerun': False,
              'production_recommendations_blocked': True, 'location_is_synthetic': True,
              'dashboard': BASE + '/dashboard/live-ops.html',
              'triage': BASE + '/dashboard/triage.html?event_id=' + str(event_id)}
    (ROOT / '.runtime/thirduni-completion-stage/demo-verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
