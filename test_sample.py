"""Standalone sample checks; uses only disposable data inside this sample folder."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import ROOT, create_app


class SampleTests(unittest.TestCase):
    def setUp(self):
        (ROOT / '.runtime').mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=ROOT / '.runtime')
        self.addCleanup(directory.cleanup)
        self.runtime = Path(directory.name)
        self.app = create_app(self.runtime)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def accept(self, client=None):
        c = client or self.client
        c.get('/api/demo')
        return c.post('/api/review', json={'decision': 'accept', 'note': 'Reviewed controlled replay; medical cause unknown.'})

    def test_complete_review_decision_and_hash_consistency(self):
        c = self.client
        self.assertEqual(c.get('/health').status_code, 200)
        self.assertEqual(c.get('/').status_code, 200)
        self.assertEqual(c.get('/assets/trigger.jpg').content[:2], b'\xff\xd8')
        self.assertIn(c.get('/assets/context.gif').content[:6], (b'GIF87a', b'GIF89a'))
        self.assertEqual(self.accept().json()['status'], 'detected')
        self.assertEqual(c.post('/api/decision', json={'status': 'approved'}).json()['status'], 'approved')
        log = c.get('/api/audit').json()
        self.assertTrue(log['consistent'] and log['media_match'])
        self.assertEqual([e['action'] for e in log['entries']], ['replay_loaded', 'review_accepted_for_demo', 'incident_approved'])

    def test_approval_requires_review_and_rejection_can_be_dismissed(self):
        c = self.client
        c.get('/api/demo')
        self.assertEqual(c.post('/api/decision', json={'status': 'approved'}).status_code, 409)
        self.assertEqual(c.post('/api/review', json={'decision': 'reject', 'note': 'Demonstration rejected for review.'}).json()['review'], 'rejected')
        self.assertEqual(c.post('/api/decision', json={'status': 'dismissed'}).json()['status'], 'dismissed')

    def test_browser_sessions_are_separate(self):
        self.accept()
        # A second client shares the running sample app, but carries no first-client cookie.
        other = TestClient(self.app)
        self.addCleanup(other.close)
        state = other.get('/api/demo').json()['state']
        self.assertEqual(state, {'review': 'pending', 'status': 'detected'})
        self.assertEqual(len(other.get('/api/audit').json()['entries']), 1)

    def test_new_session_does_not_erase_old_audit(self):
        c = self.client
        self.accept()
        count = self.app.state.db.execute('SELECT COUNT(*) FROM audit').fetchone()[0]
        self.assertEqual(c.post('/api/reset', json={}).json()['review'], 'pending')
        self.assertGreater(self.app.state.db.execute('SELECT COUNT(*) FROM audit').fetchone()[0], count)
        self.assertEqual(len(c.get('/api/audit').json()['entries']), 1)

    def test_editing_audit_content_is_detected(self):
        self.client.get('/api/demo')
        payload = json.dumps({'time': 0, 'actor': 'edited', 'action': 'edited', 'details': {}})
        self.app.state.db.execute('UPDATE audit SET payload=?', (payload,))
        self.app.state.db.commit()
        self.assertFalse(self.client.get('/api/audit').json()['consistent'])

    def test_restart_has_fresh_database_and_pending_state(self):
        self.accept()
        other_app = create_app(self.runtime)
        with TestClient(other_app) as other:
            self.assertNotEqual(other_app.state.boot, self.app.state.boot)
            self.assertEqual(other.get('/api/demo').json()['state'], {'review': 'pending', 'status': 'detected'})

    def test_input_and_cross_origin_limits(self):
        c = self.client
        self.assertEqual(c.post('/api/review', json={'decision': 'accept', 'note': ' '}).status_code, 422)
        self.assertEqual(c.post('/api/review', json={'decision': 'accept', 'note': 'x' * 501}).status_code, 422)
        self.assertEqual(c.post('/api/reset', headers={'Origin': 'https://other.example'}, json={}).status_code, 403)
        self.assertEqual(c.post('/api/review', content=(b'x' * 3000, b'x' * 3000)).status_code, 413)

    def test_media_mismatch_blocks_review_and_fails_health(self):
        original = Path.read_bytes
        def altered(path):
            return b'changed' if path.name == 'trigger.jpg' else original(path)
        with patch.object(Path, 'read_bytes', altered):
            self.assertEqual(self.client.get('/health').status_code, 503)
            self.assertEqual(self.accept().status_code, 409)

    def test_runtime_cannot_target_the_parent_private_project(self):
        with self.assertRaises(ValueError):
            create_app(ROOT.parent / '.runtime')


if __name__ == '__main__':
    unittest.main(verbosity=2)
