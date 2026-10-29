"""Real media, calibrated coordinates and signed approval in a temporary backend.

Controlled detector inputs test wiring only; no camera or external service runs.
"""
import io
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import time
import unittest
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT / 'edge')]


class EventPipeline(unittest.TestCase):
    def test_detector_location_media_rbac_audit_and_signed_approval(self):
        from crypto_utils import write_keypair
        from rtsp_worker import CameraWorker, DEFAULT_SETTINGS
        from detectors import YoloBehaviorSuite
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as directory:
            td = Path(directory)
            self.assertTrue(td.resolve().is_relative_to(ROOT))
            write_keypair(td / 'private.pem', td / 'public.pem')
            # Child test process only. No live database or live credentials.
            for key in list(os.environ):
                if key.startswith('AICOS_'):
                    del os.environ[key]
            os.environ.update(DB_PATH=str(td / 'test.db'), SNAPSHOT_DIR=str(td / 'evidence'),
                AICOS_API_KEY=secrets.token_urlsafe(40), AICOS_ALLOW_SHARED_OPERATOR_KEYS='false',
                AICOS_NOTIFY_ON='off', AICOS_INTEGRATION_WEBHOOK_URLS='',
                AICOS_EVIDENCE_SIGNING_REQUIRED='true',
                AICOS_EVIDENCE_PRIVATE_KEY=str(td / 'private.pem'),
                AICOS_EVIDENCE_PUBLIC_KEY=str(td / 'public.pem'))
            import main
            try:
                with TestClient(main.app) as client:
                    headers = {'X-AICOS-KEY': os.environ['AICOS_API_KEY']}
                    class Bridge:
                        def post_event(self, payload):
                            response = client.post('/api/events', headers=headers, json=payload)
                            self.response = response
                            return response.status_code == 201
                    class Engine:
                        def infer(self, frame):
                            return [{'box': (24, 10, 40, 48), 'cls': 0, 'conf': .95}]
                    zone = {'name': 'TEST ONLY zone', 'type': 'intrusion',
                            'points': [[0, 0], [1, 0], [1, 1], [0, 1]]}
                    cfg = {key: {'enabled': False} for key in ('person_down', 'fight', 'crowd',
                           'panic', 'loitering', 'abandoned_object', 'snatch', 'vehicles')}
                    cfg['intrusion'] = {'enabled': True}
                    detector = YoloBehaviorSuite(camera={'zones': [zone]}, config=cfg, engine=Engine())
                    frame = np.zeros((64, 64, 3), dtype=np.uint8)
                    event = detector.detect(frame, now=time.time())[0]
                    camera = {'id': 'TEST-REFERENCE-PRESERVATION', 'lat': 0, 'lon': 0,
                              'district': 'synthetic integration test', 'calibration': [
                              {'image': [0, 0], 'world': [35, 3]},
                              {'image': [1, 0], 'world': [35, 3.001]},
                              {'image': [1, 1], 'world': [35.001, 3.001]},
                              {'image': [0, 1], 'world': [35.001, 3]}]}
                    bridge = Bridge()
                    worker = CameraWorker(camera, {**DEFAULT_SETTINGS, 'integrity_enabled': False},
                                          bridge, [detector], threading.Event())
                    now = time.time()
                    for index in range(5):
                        worker.buffer.add(now - 1 + index * .2, frame + index * 30)
                    worker._emit(event, frame)
                    self.assertEqual(bridge.response.status_code, 201, bridge.response.text)
                    event_id = bridge.response.json()['id']
                    password = secrets.token_urlsafe(24)
                    sessions = {}
                    for role in ('viewer', 'operator'):
                        main.user_auth.create_user('test_' + role, password, role)
                        response = client.post('/api/auth/login', json={'username': 'test_' + role,
                                                                       'password': password})
                        self.assertEqual(response.status_code, 200)
                        sessions[role] = {'Authorization': 'Bearer ' + response.json()['token']}
                    stored = client.get('/api/events', headers=sessions['viewer']).json()['events'][0]
                    self.assertEqual(stored['id'], event_id)
                    self.assertEqual(stored['status'], 'detected')
                    self.assertEqual(stored['extra']['located'], 'calibrated')
                    self.assertAlmostEqual(stored['lat'], 35.00075, places=6)
                    self.assertAlmostEqual(stored['lon'], 3.0005, places=6)
                    self.assertEqual((stored['has_snapshot'], stored['has_clip']), (1, 1))
                    for endpoint, expected_format in (('snapshot', 'JPEG'), ('clip', 'GIF')):
                        path = f'/api/events/{event_id}/{endpoint}'
                        self.assertEqual(client.get(path).status_code, 401)
                        response = client.get(path, headers=sessions['viewer'])
                        self.assertEqual(response.status_code, 200)
                        image = Image.open(io.BytesIO(response.content))
                        self.assertEqual(image.format, expected_format)
                        image.verify()
                    verify = f'/api/events/{event_id}/evidence/verify'
                    before = client.get(verify, headers=sessions['viewer']).json()
                    self.assertTrue(before['ok'])
                    status_url = f'/api/events/{event_id}/status'
                    self.assertEqual(client.post(status_url, headers=sessions['viewer'],
                                                json={'status': 'approved'}).status_code, 403)
                    approved = client.post(status_url, headers=sessions['operator'], json={'status': 'approved'})
                    self.assertEqual(approved.status_code, 200)
                    after = client.get(verify, headers=sessions['viewer']).json()
                    self.assertTrue(after['ok'])
                    self.assertGreater(after['revision'], before['revision'])
                    self.assertTrue(client.get('/api/evidence/chain/verify', headers=sessions['viewer']).json()['ok'])
                    audit = main._conn.execute('SELECT action, note FROM audit WHERE event_id=?', (event_id,)).fetchall()
                    self.assertTrue(any(row['action'] == 'approved' and 'test_operator' in row['note'] for row in audit))
                    with client.websocket_connect('/ws/events') as ws:
                        ws.send_json({'kind': 'auth', 'token': sessions['viewer']['Authorization'][7:]})
                        message = ws.receive_json()
                        self.assertEqual(message['kind'], 'snapshot')
                        self.assertEqual(message['events'][0]['id'], event_id)
                    result = {'synthetic_inputs': True, 'calibrated_map_coordinates': True,
                              'real_jpeg_and_gif_decode': True, 'media_requires_auth': True,
                              'viewer_cannot_approve': True, 'operator_approval_audited': True,
                              'evidence_revisions_and_chain_verify': True, 'websocket_event_payload': True}
                    report_path = ROOT / '.runtime/starter-review/pipeline-checks.json'
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(json.dumps(result, indent=2))
            finally:
                main._conn.close()
