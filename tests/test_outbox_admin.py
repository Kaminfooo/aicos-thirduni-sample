import json, os, sqlite3, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'edge'))
import outbox_admin

class OutboxAdminTests(unittest.TestCase):
    def test_status_export_and_requeue(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'o.db'; out = Path(td) / 'dead.json'
            c = sqlite3.connect(db); c.row_factory = sqlite3.Row
            c.execute('CREATE TABLE outbox(id INTEGER PRIMARY KEY,event_uuid TEXT UNIQUE,created_at REAL,payload TEXT,severity INTEGER,protected INTEGER,attempts INTEGER,next_attempt_at REAL,last_error TEXT)')
            c.execute('CREATE TABLE dead_letter(id INTEGER PRIMARY KEY,event_uuid TEXT,created_at REAL,quarantined_at REAL,payload TEXT,severity INTEGER,attempts INTEGER,last_error TEXT,reason TEXT)')
            payload = json.dumps({'event_uuid':'u1','severity':5,'type':'fall'})
            c.execute('INSERT INTO dead_letter VALUES(1,?,?,?,?,?,?,?,?)', ('u1',1.0,2.0,payload,5,1,'403','permanent'))
            c.commit()
            self.assertEqual(outbox_admin.status(c)['dead_letter'], 1)
            self.assertEqual(outbox_admin.export_dead(c, str(out)), 1)
            self.assertEqual(json.loads(out.read_text())[0]['event_uuid'], 'u1')
            self.assertTrue(outbox_admin.requeue(c, 1))
            self.assertEqual(outbox_admin.status(c)['protected_queued'], 1)
            self.assertEqual(outbox_admin.status(c)['dead_letter'], 0)
            c.close()

if __name__ == '__main__': unittest.main()
