"""Production-like webhook delivery tests using a real local HTTP server."""
import http.server
import os
import sqlite3
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from integrations import IntegrationBus


class _Handler(http.server.BaseHTTPRequestHandler):
    status = 204
    body = b""
    def do_POST(self):
        n = int(self.headers.get("content-length", "0") or 0)
        if n:
            self.rfile.read(n)
        self.send_response(type(self).status)
        self.end_headers()
        self.wfile.write(type(self).body)
    def log_message(self, *_args):
        pass


class RealHTTPClassification(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/hook"
        self.old = {k: os.environ.get(k) for k in (
            "AICOS_INTEGRATION_WEBHOOK_URLS", "AICOS_INTEGRATION_STAGES"
        )}
        os.environ["AICOS_INTEGRATION_WEBHOOK_URLS"] = self.url
        os.environ["AICOS_INTEGRATION_STAGES"] = "approved"
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.bus = IntegrationBus(self.conn, threading.RLock())
        self.bus.close()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.conn.close()
        for k, v in self.old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

    def _queue(self):
        self.bus.queue("approved", {"id": 1, "event_uuid": "evt", "status": "approved"})

    def test_real_404_http_error_is_permanent(self):
        _Handler.status = 404; _Handler.body = b"missing"
        self._queue()
        self.bus.flush_once()
        st = self.bus.status()
        self.assertEqual(st["abandoned"], 1)
        self.assertEqual(st["pending"], 0)

    def test_real_500_is_transient(self):
        _Handler.status = 500; _Handler.body = b"temporary"
        self._queue()
        self.bus.flush_once()
        st = self.bus.status()
        self.assertEqual(st["abandoned"], 0)
        self.assertEqual(st["pending"], 1)
        self.assertEqual(st["retrying"], 1)

    def test_real_204_is_success(self):
        _Handler.status = 204; _Handler.body = b""
        self._queue()
        self.assertEqual(self.bus.flush_once(), 1)
        self.assertEqual(self.bus.status()["delivered"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
