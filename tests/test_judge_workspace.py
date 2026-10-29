"""Focused sandbox checks. Fake inference tests boundaries, not model accuracy."""
import base64
from io import BytesIO
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from PIL import Image
from deploy.judge_workspace import create_app, COOKIE, core, isolated, inference_interpreter, workspace_hosts
from deploy.judge_capabilities import IDS


class Stub:
    def start(self):
        pass

    def close(self):
        pass

    def drop(self, session):
        pass

    def infer(self, session, image, seconds, settings=None):
        return {'events': [{'agent': 'emergency', 'type': 'Person down / possible medical emergency',
            'severity': 4, 'confidence': .8, 'extra': {'human_review_required': True}}],
            'observation': {'reason': 'alert', 'persons': 1}, 'model_sha256': 'test-only-stub'}


def frame_image(width=480, height=270):
    data = BytesIO()
    Image.new('RGB', (width, height), 'white').save(data, format='JPEG')
    return base64.b64encode(data.getvalue()).decode()


class JudgeWorkspaceTests(unittest.TestCase):
    def test_portable_worker_and_exact_external_host(self):
        missing=isolated.BOOT/'no-inference-environment'
        self.assertEqual(inference_interpreter(missing,platform='posix',executable='/usr/local/bin/python').as_posix(), '/usr/local/bin/python')
        with self.assertRaises(RuntimeError):
            inference_interpreter(missing,platform='nt')
        self.assertIn('aicos-example.onrender.com',workspace_hosts('aicos-example.onrender.com'))
        for invalid in ('*','https://example.com','example.com:8080','example.com/path'):
            with self.assertRaises(RuntimeError):workspace_hosts(invalid)

    def test_camera_location_validation_and_zero_coordinates(self):
        for position in ({'latitude':1}, {'longitude':2}, {'latitude':91,'longitude':0},
                         {'latitude':0,'longitude':181}, {'camera_name':'   '}):
            self.assertEqual(self.client.post('/judge/api/run',json={'mode':'file',**position}).status_code,422)
        settings={'mode':'file','camera_name':'Entrance A','location_label':'Synthetic test site',
                  'latitude':0,'longitude':0}
        self.client.post('/judge/api/run',json=settings).raise_for_status()
        self.alert()
        event=self.client.get('/judge/api/incident').json()['event']
        self.assertEqual((event['lat'],event['lon']),(0,0))
        self.assertTrue(event['extra']['location_configured'])
        self.assertEqual(event['extra']['camera_name'],'Entrance A')
        self.assertEqual(event['extra']['location_source'],'operator-configured camera position')
        listed=self.client.get('/judge/api/incidents').json()['incidents'][0]['location']
        self.assertEqual((listed['latitude'],listed['longitude']),(0,0))

    def test_unknown_location_is_explicit_and_assets_are_bounded(self):
        self.alert()
        event=self.client.get('/judge/api/incident').json()['event']
        self.assertFalse(event['extra']['location_configured'])
        listed=self.client.get('/judge/api/incidents').json()['incidents'][0]['location']
        self.assertIsNone(listed['latitude'])
        self.assertIsNone(listed['longitude'])
        for name in ('leaflet.js','leaflet.css','location-map.js','location-map.css','LEAFLET-LICENSE.txt','camera-wall.js','camera-wall.css'):
            self.assertEqual(self.client.get('/workspace-assets/'+name).status_code,200)
        self.assertEqual(self.client.get('/workspace-assets/judge_workspace.py').status_code,404)
        self.assertIn('geolocation=(self)',self.client.get('/').headers['permissions-policy'])

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(Stub()))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        self.client.post('/judge/api/run', json={'mode': 'scenario'}).raise_for_status()

    def alert(self):
        result = self.client.post('/judge/api/frame', json={'image': frame_image(), 'seconds': 0})
        result.raise_for_status()
        return result.json()

    def test_no_seed_or_review_without_detector_alert(self):
        self.assertEqual(self.client.get('/judge/api/incident').status_code, 409)
        self.assertEqual(self.client.post('/judge/api/decision', json={'status': 'approved'}).status_code, 409)

    def test_all_three_modes_and_no_arbitrary_urls(self):
        for mode in ('scenario', 'file', 'camera'):
            self.assertEqual(self.client.post('/judge/api/run', json={'mode': mode}).status_code, 200)
        self.assertEqual(self.client.post('/judge/api/run', json={'mode': 'http://127.0.0.1'}).status_code, 422)
        self.assertEqual(self.client.get('/judge/assets/other.mp4').status_code, 404)
        self.assertEqual(self.client.post('/api/events', json={}).status_code, 404)

    def test_session_isolation_and_fresh_run(self):
        self.alert()
        self.assertEqual(self.client.get('/judge/api/media/snapshot').status_code, 200)
        old_cookie = self.client.cookies.get(COOKIE)
        self.client.cookies.clear()
        self.assertEqual(self.client.get('/judge/api/incident').status_code, 401)
        self.assertEqual(self.client.get('/judge/api/media/snapshot').status_code, 401)
        self.client.cookies.set(COOKIE, old_cookie)
        self.client.post('/judge/api/run', json={'mode': 'file'}).raise_for_status()
        self.assertEqual(self.client.get('/judge/api/incident').status_code, 409)

    def test_signed_evidence_review_then_separate_approval(self):
        self.alert()
        before = self.client.get('/judge/api/incident').json()
        self.assertTrue(before['verification']['ok'])
        self.assertEqual(before['event']['status'], 'detected')
        self.assertEqual(self.client.post('/judge/api/decision', json={'status': 'approved'}).status_code, 409)
        self.assertEqual(self.client.post('/judge/api/review', json={'decision': 'accept', 'note': ' '}).status_code, 422)
        reviewed = self.client.post('/judge/api/review', json={'decision': 'accept', 'note': 'Synthetic test; cause unknown.'})
        self.assertEqual(reviewed.status_code, 200)
        self.assertEqual(self.client.get('/judge/api/incident').json()['event']['status'], 'detected')
        self.assertEqual(self.client.post('/judge/api/decision', json={'status': 'approved'}).status_code, 200)
        self.assertEqual(self.client.post('/judge/api/decision', json={'status': 'approved'}).status_code, 409)
        after = self.client.get('/judge/api/incident').json()
        self.assertTrue(after['verification']['ok'])
        self.assertIn('approved', [row['action'] for row in after['audit']])

    def test_invalid_or_unbounded_frames_and_time(self):
        for image in ('not-an-image', frame_image(641, 270)):
            self.assertEqual(self.client.post('/judge/api/frame', json={'image': image, 'seconds': 0}).status_code, 422)
        self.assertEqual(self.client.post('/judge/api/frame', json={'image': frame_image(), 'seconds': 91}).status_code, 422)
        self.alert()
        self.assertEqual(self.client.post('/judge/api/frame', json={'image': frame_image(), 'seconds': 0}).status_code, 422)
        self.assertEqual(self.client.post('/judge/api/frame', json={'image': frame_image(), 'seconds': .1}).status_code, 422)

    def test_origin_actual_body_limit_and_camera_policy(self):
        self.assertEqual(self.client.post('/judge/api/run', json={'mode': 'file'}, headers={'Origin': 'https://other.example'}).status_code, 403)
        self.assertEqual(self.client.post('/judge/api/frame', content=b'X' * 700001).status_code, 413)
        home = self.client.get('/')
        self.assertEqual(home.status_code, 200)
        self.assertIn('camera=(self)', home.headers['permissions-policy'])
        self.assertIn('microphone=()', home.headers['permissions-policy'])

    def test_finish_blocks_more_frames_preserves_incident(self):
        self.alert()
        self.assertEqual(self.client.post('/judge/api/finish', json={}).status_code, 200)
        self.assertEqual(self.client.post('/judge/api/frame', json={'image': frame_image(), 'seconds': 1}).status_code, 409)
        self.assertTrue(self.client.get('/judge/api/incident').json()['verification']['ok'])

    def test_runtime_isolation_shared_keys_off_notifications_off(self):
        self.assertTrue(isolated.BOOT.is_relative_to(ROOT / '.runtime/judge-workspace-20261026'))
        self.assertFalse(core.ALLOW_SHARED_OPERATOR_KEYS)
        import os
        self.assertEqual(os.environ['AICOS_NOTIFY_ON'], 'off')
        self.assertEqual(os.environ['AICOS_AI_REPORTS'], 'false')

    def test_detector_catalog_and_validation(self):
        rows=self.client.get('/judge/api/capabilities').json()['capabilities']
        self.assertEqual({r['id'] for r in rows},set(IDS))
        for payload in ({'detectors':[]},{'detectors':['unknown']},
            {'detectors':['fire_smoke','fire_smoke']},{'detectors':['vehicles']},
            {'detectors':['intrusion'],'region':[.9,0,.1,1]},
            {'detectors':['crowd'],'crowd_warn':1},
            {'direction':'left'},{'region':[0,0,1,2]}):
            self.assertEqual(self.client.post('/judge/api/run',json={'mode':'file',**payload}).status_code,422,payload)
        response=self.client.post('/judge/api/run',json={'mode':'camera','detectors':list(IDS),
            'region':[.1,.1,.9,.9],'loiter_seconds':10,'direction':'right'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['settings']['detectors'],list(IDS))

    def test_multi_alert_evidence_decisions_and_isolation(self):
        class Multi(Stub):
            def infer(self,session,image,seconds,settings=None):
                return {'events':[{'agent':'environmental' if k=='fire_smoke' else 'security',
                    'type':t,'severity':3,'confidence':.7,'extra':{'judge_capability':k}}
                    for k,t in [('fire_smoke','Possible fire detected'),('violence','Possible physical violence detected')]],
                    'observation':{'reason':'alert','persons':0},'detectors':[], 'model_sha256':'test-only-stub'}
        with TestClient(create_app(Multi())) as client:
            client.post('/judge/api/run',json={'mode':'file','detectors':['fire_smoke','violence']}).raise_for_status()
            first=client.post('/judge/api/frame',json={'image':frame_image(),'seconds':0}).json()
            self.assertEqual(len(first['incidents']),2)
            client.post('/judge/api/frame',json={'image':frame_image(),'seconds':1}).raise_for_status()
            self.assertEqual(len(client.get('/judge/api/incidents').json()['incidents']),2)
            ids=[r['event_id'] for r in first['incidents']]
            for ident in ids:
                query='?event_id='+str(ident)
                self.assertTrue(client.get('/judge/api/incident'+query).json()['verification']['ok'])
                self.assertEqual(client.get('/judge/api/media/snapshot'+query).status_code,200)
            query='?event_id='+str(ids[1])
            client.post('/judge/api/review'+query,json={'decision':'accept','note':'Stub-based test only.'}).raise_for_status()
            client.post('/judge/api/decision'+query,json={'status':'approved'}).raise_for_status()
            self.assertEqual(client.post('/judge/api/decision'+query,json={'status':'approved'}).status_code,409)
            self.assertEqual(client.get('/judge/api/incident?event_id='+str(ids[0])).json()['event']['status'],'detected')
            self.assertEqual(client.get('/judge/api/incident'+query).json()['event']['status'],'approved')
            client.post('/judge/api/run',json={'mode':'camera'}).raise_for_status()
            self.assertEqual(client.get('/judge/api/incident'+query).status_code,409)
            self.assertEqual(client.get('/judge/api/media/snapshot'+query).status_code,409)


if __name__ == '__main__':
    unittest.main()
