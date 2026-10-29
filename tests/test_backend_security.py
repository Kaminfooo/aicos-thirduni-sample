import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class BackendSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["AICOS_API_KEY"] = "device-key-abcdefghijklmnopqrstuvwxyz-123456"
        os.environ["AICOS_OPERATOR_KEY"] = "operator-key-abcdefghijklmnopqrstuvwxyz-1234"
        os.environ["AICOS_VIEWER_KEY"] = "viewer-key-abcdefghijklmnopqrstuvwxyz-123456"
        os.environ["DB_PATH"] = str(Path(cls.tmp.name) / "aicos.db")
        os.environ["SNAPSHOT_DIR"] = str(Path(cls.tmp.name) / "snapshots")
        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        sys.path.insert(0, backend_dir)
        cls.main = importlib.import_module("main")
        cls.client = TestClient(cls.main.app)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.main._conn.close()
        except Exception:
            pass
        cls.tmp.cleanup()

    def test_health_is_public_but_events_are_not(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        self.assertEqual(self.client.get("/api/events").status_code, 401)
        ok = self.client.get(
            "/api/events", headers={"X-AICOS-OPERATOR-KEY": os.environ["AICOS_VIEWER_KEY"]}
        )
        self.assertEqual(ok.status_code, 200)

    def test_device_ingest_is_idempotent(self):
        event = {
            "event_uuid": "11111111-2222-4333-8444-555555555555",
            "source_id": "TEST-01",
            "agent": "security",
            "type": "Regression test event",
            "severity": 4,
            "confidence": 0.91,
        }
        headers = {"X-AICOS-KEY": os.environ["AICOS_API_KEY"]}
        first = self.client.post("/api/events", json=event, headers=headers)
        self.assertEqual(first.status_code, 201)
        second = self.client.post("/api/events", json=event, headers=headers)
        self.assertEqual(second.status_code, 201)
        self.assertTrue(second.json().get("duplicate"))
        viewer = {"X-AICOS-OPERATOR-KEY": os.environ["AICOS_VIEWER_KEY"]}
        events = self.client.get("/api/events", headers=viewer).json()["events"]
        ids = [x.get("event_uuid") for x in events]
        self.assertEqual(ids.count(event["event_uuid"]), 1)

    def test_websocket_auth_uses_first_message_not_url_token(self):
        import json
        with self.client.websocket_connect("/ws/events") as ws:
            ws.send_text(json.dumps({"kind": "auth", "token": os.environ["AICOS_VIEWER_KEY"]}))
            msg = ws.receive_json()
            self.assertEqual(msg.get("kind"), "snapshot")


    def test_websocket_accepts_local_session_when_shared_keys_disabled(self):
        import json
        try:
            self.main.user_auth.create_user("wsviewer", "aicos websocket password 123", "viewer")
        except Exception:
            pass
        login = self.client.post("/api/auth/login", json={"username":"wsviewer","password":"aicos websocket password 123"})
        self.assertEqual(login.status_code, 200)
        token = login.json()["token"]
        previous = self.main.ALLOW_SHARED_OPERATOR_KEYS
        self.main.ALLOW_SHARED_OPERATOR_KEYS = False
        try:
            with self.client.websocket_connect("/ws/events") as ws:
                ws.send_text(json.dumps({"kind":"auth", "token":token}))
                msg = ws.receive_json()
                self.assertEqual(msg.get("kind"), "snapshot")
        finally:
            self.main.ALLOW_SHARED_OPERATOR_KEYS = previous

    def test_http_query_token_is_not_accepted(self):
        r = self.client.get("/api/events?token=" + os.environ["AICOS_VIEWER_KEY"])
        self.assertEqual(r.status_code, 401)

    def test_local_rbac_login_and_actor_identity(self):
        try:
            self.main.user_auth.create_user("ops1", "aicos operator password 123", "operator")
        except Exception:
            pass
        login = self.client.post("/api/auth/login", json={"username":"ops1","password":"aicos operator password 123"})
        self.assertEqual(login.status_code, 200)
        token = login.json()["token"]
        me = self.client.get("/api/auth/me", headers={"Authorization":"Bearer "+token})
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["user"]["username"], "ops1")
        self.assertIn("user:ops1:operator", me.json()["actor"])

    def test_viewer_cannot_change_status_operator_can(self):
        event = {
            "event_uuid": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
            "source_id": "TEST-02",
            "agent": "emergency",
            "type": "Status auth test",
            "severity": 4,
            "confidence": 0.8,
        }
        create = self.client.post(
            "/api/events", json=event,
            headers={"X-AICOS-KEY": os.environ["AICOS_API_KEY"]},
        )
        event_id = create.json()["id"]
        viewer = self.client.post(
            f"/api/events/{event_id}/status", json={"status": "dismissed"},
            headers={"X-AICOS-OPERATOR-KEY": os.environ["AICOS_VIEWER_KEY"]},
        )
        self.assertEqual(viewer.status_code, 403)
        operator = self.client.post(
            f"/api/events/{event_id}/status", json={"status": "dismissed"},
            headers={"X-AICOS-OPERATOR-KEY": os.environ["AICOS_OPERATOR_KEY"]},
        )
        self.assertEqual(operator.status_code, 200)


if __name__ == "__main__":
    unittest.main()
