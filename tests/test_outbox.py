import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


class OutboxTests(unittest.TestCase):
    def test_failed_event_is_persisted_then_retried(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "edge"))
        from rtsp_worker import BackendClient

        with tempfile.TemporaryDirectory() as td:
            os.environ["AICOS_OUTBOX_PATH"] = str(Path(td) / "outbox.db")
            client = BackendClient(
                "http://127.0.0.1:9",
                "device-key-abcdefghijklmnopqrstuvwxyz-123456",
            )
            try:
                client._send = lambda payload: (False, "offline")
                accepted = client.post_event({
                    "source_id": "CAM-X", "agent": "security", "type": "test",
                    "severity": 3, "confidence": 0.9,
                })
                self.assertTrue(accepted)
                with client._lock:
                    n = client._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
                self.assertEqual(n, 1)

                client._send = lambda payload: (True, "")
                deadline = time.time() + 6
                while time.time() < deadline:
                    with client._lock:
                        n = client._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
                    if n == 0:
                        break
                    time.sleep(0.25)
                self.assertEqual(n, 0)
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
