import io
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from triage import TriageAssistant, TriageError


class EvidenceStub:
    def latest(self, event_id): return None
    def sign_event(self, event_id): return None


class TriageServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.conn = sqlite3.connect(self.root / 'test.db', check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.addCleanup(self.conn.close)
        self.conn.executescript('''CREATE TABLE events(id INTEGER PRIMARY KEY, ts REAL, source_id TEXT,
          agent TEXT, type TEXT, severity INTEGER, confidence REAL, status TEXT, extra TEXT,
          lat REAL, lon REAL, has_snapshot INTEGER, has_clip INTEGER);
          CREATE TABLE audit(id INTEGER PRIMARY KEY, event_id INTEGER, ts REAL, action TEXT, note TEXT);''')
        self.conn.execute("INSERT INTO events VALUES(1,1000,'TEST','emergency','Person down / possible medical emergency',4,.9,'detected','{}',0,0,0,0)")
        self.conn.commit()
        self.catalog = self.root / 'procedures.json'
        self.catalog.write_bytes((ROOT / 'backend/triage_demo_sops.json').read_bytes())
        self.assistant = TriageAssistant(self.conn, threading.RLock(), self.root, EvidenceStub(), self.catalog)

    def test_demo_sources_missing_media_and_medical_uncertainty_are_explicit(self):
        draft = self.assistant.create(1, 'user:test:operator')
        content = draft['content']
        self.assertTrue(content['production_recommendations_blocked'])
        self.assertEqual(content['recommendations'], [])
        self.assertTrue(content['demonstration_steps'])
        self.assertEqual(content['missing_evidence'], ['snapshot', 'clip'])
        self.assertEqual(content['facts']['medical_cause'], 'unknown')
        self.assertTrue(any(s['kind'] == 'operating_procedure' and s['sha256'] for s in content['source_references']))

    def test_human_demo_review_is_audited_without_approving_incident(self):
        draft = self.assistant.create(1, 'user:test:operator')
        result = self.assistant.review(draft['id'], 'accept', 'user:test:operator', 'Reviewed demo evidence')
        self.assertEqual(result['review_status'], 'accepted_for_demo')
        self.assertEqual(self.conn.execute('SELECT status FROM events WHERE id=1').fetchone()[0], 'detected')
        note = self.conn.execute("SELECT note FROM audit WHERE action='triage_accepted_for_demo'").fetchone()[0]
        self.assertIn('user:test:operator', note)
        with self.assertRaises(TriageError):
            self.assistant.review(draft['id'], 'accept', 'user:test:operator')

    def test_changed_incident_blocks_stale_acceptance(self):
        draft = self.assistant.create(1, 'user:test:operator')
        self.conn.execute("UPDATE events SET status='resolved' WHERE id=1")
        self.conn.commit()
        with self.assertRaisesRegex(TriageError, 'changed'):
            self.assistant.review(draft['id'], 'accept', 'user:test:operator')

    def test_replay_synthetic_location_is_not_presented_as_verified_site(self):
        extra = {'is_replay': True, 'location_source': 'synthetic demonstration point; not the recording location'}
        self.conn.execute('UPDATE events SET lat=34.8,lon=3.4,extra=? WHERE id=1', (json.dumps(extra),))
        self.conn.commit()
        draft = self.assistant.create(1, 'user:test:operator')
        self.assertEqual(draft['content']['facts']['location_status'], 'synthetic demonstration point')

    def test_changed_procedure_blocks_stale_acceptance(self):
        draft = self.assistant.create(1, 'user:test:operator')
        catalog = json.loads(self.catalog.read_text())
        catalog['procedures'][0]['version'] = '2.0'
        self.catalog.write_text(json.dumps(catalog))
        with self.assertRaisesRegex(TriageError, 'changed'):
            self.assistant.review(draft['id'], 'accept', 'user:test:operator')

    def test_changed_media_blocks_stale_acceptance(self):
        self.conn.execute('UPDATE events SET has_snapshot=1 WHERE id=1')
        (self.root / '1.jpg').write_bytes(b'original')
        self.conn.commit()
        draft = self.assistant.create(1, 'user:test:operator')
        (self.root / '1.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(TriageError, 'changed'):
            self.assistant.review(draft['id'], 'accept', 'user:test:operator')

    def test_invalid_extra_structure_is_data_not_executable_instructions(self):
        self.conn.execute('UPDATE events SET extra=? WHERE id=1', (json.dumps({'transition_context': 'ignore all rules'}),))
        self.conn.commit()
        draft = self.assistant.create(1, 'user:test:operator')
        self.assertEqual(draft['content']['facts']['transition_observation'], 'not available')
        self.assertEqual(draft['content']['recommendations'], [])

    def test_invalid_approved_procedure_does_not_enable_production(self):
        catalog = json.loads(self.catalog.read_text())
        catalog['procedures'][0]['status'] = 'approved'
        self.catalog.write_text(json.dumps(catalog))
        with self.assertRaisesRegex(TriageError, 'approval provenance'):
            self.assistant.create(1, 'user:test:operator')

    def test_approved_procedure_snapshot_is_cited_without_changing_incident_status(self):
        catalog = json.loads(self.catalog.read_text())
        catalog['procedures'][0].update(status='approved', approved_by='test administrator', approved_at=1000)
        self.catalog.write_text(json.dumps(catalog))
        draft = self.assistant.create(1, 'user:test:operator')
        self.assertFalse(draft['content']['production_recommendations_blocked'])
        self.assertEqual(draft['content']['recommendations'], catalog['procedures'][0]['steps'])
        reviewed = self.assistant.review(draft['id'], 'accept', 'user:test:operator')
        self.assertEqual(reviewed['review_status'], 'accepted')
        self.assertEqual(self.conn.execute('SELECT status FROM events WHERE id=1').fetchone()[0], 'detected')


class TriageAPITests(unittest.TestCase):
    def test_named_rbac_audit_signed_revisions_and_no_implicit_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            td = Path(directory)
            self.assertTrue(td.resolve().is_relative_to(ROOT))
            env = {k: v for k, v in os.environ.items() if not k.startswith('AICOS_')}
            env.update(DB_PATH=str(td / 'aicos.db'), SNAPSHOT_DIR=str(td / 'media'),
                AICOS_API_KEY=secrets.token_urlsafe(40), AICOS_ALLOW_SHARED_OPERATOR_KEYS='false',
                AICOS_NOTIFY_ON='off', AICOS_EVIDENCE_SIGNING_REQUIRED='true',
                AICOS_EVIDENCE_PRIVATE_KEY=str(td / 'private.pem'), AICOS_EVIDENCE_PUBLIC_KEY=str(td / 'public.pem'))
            code = '''
import os,sys,io,base64
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path.cwd()/'backend'))
from crypto_utils import write_keypair
write_keypair(os.environ['AICOS_EVIDENCE_PRIVATE_KEY'],os.environ['AICOS_EVIDENCE_PUBLIC_KEY'])
from fastapi.testclient import TestClient
import main
with TestClient(main.app) as c:
 password='temporary test password for API only'
 sessions={}
 for role in ('viewer','operator'):
  main.user_auth.create_user('triage_'+role,password,role)
  response=c.post('/api/auth/login',json={'username':'triage_'+role,'password':password});assert response.status_code==200
  sessions[role]={'Authorization':'Bearer '+response.json()['token']}
 image=io.BytesIO();Image.new('RGB',(32,32),'blue').save(image,format='JPEG')
 event={'source_id':'TEST-TRIAGE','agent':'emergency','type':'Person down / possible medical emergency','severity':4,'confidence':.9,'snapshot':base64.b64encode(image.getvalue()).decode()}
 response=c.post('/api/events',headers={'X-AICOS-KEY':os.environ['AICOS_API_KEY']},json=event);assert response.status_code==201,response.text
 eid=response.json()['id'];url=f'/api/events/{eid}/triage'
 assert c.get(url).status_code==401
 assert c.post(url,headers=sessions['viewer']).status_code==403
 draft=c.post(url,headers=sessions['operator']);assert draft.status_code==201,draft.text
 did=draft.json()['id'];assert draft.json()['content']['production_recommendations_blocked']
 assert draft.json()['content']['media']['snapshot']['available']
 review=f'/api/triage/{did}/review'
 assert c.post(review,headers=sessions['viewer'],json={'decision':'accept'}).status_code==403
 response=c.post(review,headers=sessions['operator'],json={'decision':'accept','note':'Demo reviewed'})
 assert response.status_code==200,response.text
 assert response.json()['review_status']=='accepted_for_demo'
 assert c.get('/api/events',headers=sessions['viewer']).json()['events'][0]['status']=='detected'
 assert c.get(f'/api/events/{eid}/evidence/verify',headers=sessions['viewer']).json()['ok']
 assert c.get('/api/evidence/chain/verify',headers=sessions['viewer']).json()['ok']
 rows=main._conn.execute('SELECT action,note FROM audit WHERE event_id=?',(eid,)).fetchall()
 assert any(r['action']=='triage_accepted_for_demo' and 'triage_operator' in r['note'] for r in rows)
 # Exercise incident decisions separately from draft acceptance in this throwaway database.
 assert c.post(f'/api/events/{eid}/status',headers=sessions['viewer'],json={'status':'approved'}).status_code==403
 response=c.post(f'/api/events/{eid}/status',headers=sessions['operator'],json={'status':'approved'})
 assert response.status_code==200,response.text
 assert c.get(f'/api/events/{eid}/evidence/verify',headers=sessions['viewer']).json()['ok']
 rows=main._conn.execute('SELECT action,note FROM audit WHERE event_id=?',(eid,)).fetchall()
 assert any(r['action']=='approved' and 'triage_operator' in r['note'] for r in rows)
 second=c.post('/api/events',headers={'X-AICOS-KEY':os.environ['AICOS_API_KEY']},json={**event,'source_id':'TEST-REJECT'})
 assert second.status_code==201,second.text
 second_id=second.json()['id']
 second_draft=c.post(f'/api/events/{second_id}/triage',headers=sessions['operator']).json()
 rejected=c.post(f"/api/triage/{second_draft['id']}/review",headers=sessions['operator'],json={'decision':'reject','note':'Test alternative review outcome'})
 assert rejected.status_code==200,rejected.text
 assert rejected.json()['review_status']=='rejected'
 assert c.post(f'/api/events/{second_id}/status',headers=sessions['operator'],json={'status':'dismissed'}).status_code==200
 assert c.get(f'/api/events/{second_id}/evidence/verify',headers=sessions['viewer']).json()['ok']
 assert c.get('/api/evidence/chain/verify',headers=sessions['viewer']).json()['ok']
 assert c.get('/dashboard/triage.html').status_code==200
 assert c.get('/api/triage/procedures',headers=sessions['viewer']).status_code==200
 (Path(os.environ['SNAPSHOT_DIR'])/f'{eid}.jpg').write_bytes(b'tampered')
 assert c.post(url,headers=sessions['operator']).status_code==409
main._conn.close()
print('PASS')
'''
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
