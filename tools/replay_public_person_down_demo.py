"""Seed the licensed replay into the existing loopback demo, without approval or dispatch.

Preserves private historical incidents. This local demo database must not be published.
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


def main():
    bundle = ROOT / 'submission/public-demo-media'
    review = json.loads((bundle / 'MEDIA-REVIEW.json').read_text())
    video = bundle / review['selected_demo_source']
    if hashlib.sha256(video.read_bytes()).hexdigest() != review['copied_source_sha256'][video.name]:
        raise RuntimeError('Public source video hash mismatch')
    record = review['selected_event']
    event = json.loads(json.dumps(record['event']))
    event['type'] = '[REPLAY] ' + event['type']
    event['event_uuid'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'public-assist:' + review['copied_source_sha256'][video.name] + review['profile_sha256']))
    event['extra'].update(is_replay=True, source_video_sha256=review['copied_source_sha256'][video.name],
        source_credit='ASSIST-IoT research dataset; Danilenka et al.; CC BY 4.0',
        source_page='https://github.com/ASSIST-IoT-SRIPAS/multimodal-fall-detection-dataset',
        source_license='https://creativecommons.org/licenses/by/4.0/',
        location_source='synthetic demonstration point; not the recording location',
        capture_seconds=record['seconds'], controlled_simulation=True,
        view_context='Reference mosaic includes two views of the same participant; alert is in inset view',
        replay_profile_sha256=review['profile_sha256'])
    cap = cv2.VideoCapture(str(video))
    frames, trigger, index = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if record['seconds'] - 6 <= timestamp <= record['seconds']:
            frames.append((timestamp, frame))
        if index == record['frame']:
            trigger = frame
        index += 1
    cap.release()
    if trigger is None:
        raise RuntimeError('Recorded triggering frame unavailable')
    class Bridge:
        response = None
        def post_event(self, payload):
            status, raw = request('/api/events', payload, device=True)
            self.response = json.loads(raw)
            return status == 201
    bridge = Bridge()
    worker = CameraWorker({'id': 'REPLAY-PUBLIC-PERSON-01', 'name': 'Licensed controlled fall replay',
        'district': 'DEMONSTRATION: synthetic map point', 'lat': 34.88803, 'lon': 3.490238},
        {**DEFAULT_SETTINGS, 'integrity_enabled': False}, bridge, [], threading.Event())
    now = time.time()
    for timestamp, frame in frames:
        worker.buffer.add(now - (record['seconds'] - timestamp), frame)
    worker._emit(event, trigger)
    if not bridge.response:
        raise RuntimeError('Demo did not accept replay event')
    eid = bridge.response['id']
    values = existing.credentials()
    _, raw = request('/api/auth/login', {'username': values['username'], 'password': values['password']})
    token = json.loads(raw)['token']
    try:
        for kind in ('snapshot', 'clip'):
            _, raw = request(f'/api/events/{eid}/{kind}', token=token)
            Image.open(io.BytesIO(raw)).verify()
        _, raw = request(f'/api/events/{eid}/triage', token=token)
        drafts = json.loads(raw)['drafts']
        if drafts:
            draft = drafts[0]
        else:
            _, raw = request(f'/api/events/{eid}/triage', {}, token)
            draft = json.loads(raw)
        _, raw = request(f'/api/events/{eid}/evidence/verify', token=token)
        if not json.loads(raw)['ok']:
            raise RuntimeError('Signed public replay evidence verification failed')
        _, raw = request('/api/evidence/chain/verify', token=token)
        if not json.loads(raw)['ok']:
            raise RuntimeError('Evidence chain verification failed')
    finally:
        request('/api/auth/logout', {}, token)
    result = {'event_id': eid, 'draft_id': draft['id'], 'draft_review_status': draft['review_status'],
              'triage_url': BASE + '/dashboard/triage.html?event_id=' + str(eid),
              'signed_media_verified': True, 'source': video.name, 'alert_seconds': record['seconds'],
              'personal_media_in_new_incident': False, 'existing_history_preserved': True,
              'public_publication_performed': False, 'model_rerun': False, 'camera_used': False,
              'incident_approval_performed': False}
    (ROOT / '.runtime/public-demo-media-20261025/local-public-incident.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
