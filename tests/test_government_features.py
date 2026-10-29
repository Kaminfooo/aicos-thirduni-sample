import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from crypto_utils import load_private_key, load_public_key, sign_envelope, verify_envelope, write_keypair
from licensing import LICENSE_SCHEMA, LicenseManager, LicenseError
from certification import CERT_SCHEMA, CameraCertificateRegistry, CameraCertificateError
from evidence import EvidenceManager
from forensics import ForensicIndex


class EnvGuard:
    def __init__(self, *names):
        self.names = names
        self.old = {n: os.environ.get(n) for n in names}
    def __enter__(self): return self
    def __exit__(self, *args):
        for n, v in self.old.items():
            if v is None: os.environ.pop(n, None)
            else: os.environ[n] = v


class GovernmentFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.td = Path(self.tmp.name)
        self.private = self.td / "vendor-private.pem"
        self.public = self.td / "vendor-public.pem"
        write_keypair(self.private, self.public)

    def tearDown(self):
        self.tmp.cleanup()

    def test_signed_envelope_rejects_tamper(self):
        env = sign_envelope("test.v1", {"x": 1}, load_private_key(self.private))
        self.assertEqual(verify_envelope(env, load_public_key(self.public), "test.v1")["x"], 1)
        env["payload"]["x"] = 2
        with self.assertRaises(Exception):
            verify_envelope(env, load_public_key(self.public), "test.v1")

    def test_offline_license_enforces_module_and_camera_limit(self):
        lic = self.td / "license.json"
        payload = {
            "license_id": "L1", "customer": "Gov", "deployment_id": "D1",
            "not_before": time.time() - 10, "expires_at": time.time() + 3600,
            "grace_days": 0, "max_cameras": 1, "modules": ["violence"], "features": []
        }
        lic.write_text(json.dumps(sign_envelope(LICENSE_SCHEMA, payload, load_private_key(self.private))))
        names = ("AICOS_LICENSE_REQUIRED", "AICOS_LICENSE_FILE", "AICOS_LICENSE_PUBLIC_KEY", "AICOS_MACHINE_ID")
        with EnvGuard(*names):
            os.environ["AICOS_LICENSE_REQUIRED"] = "true"
            os.environ["AICOS_LICENSE_FILE"] = str(lic)
            os.environ["AICOS_LICENSE_PUBLIC_KEY"] = str(self.public)
            lm = LicenseManager()
            lm.assert_event_allowed("CAM-1", "Possible fight / altercation", "security", [])
            with self.assertRaises(LicenseError):
                lm.assert_event_allowed("CAM-2", "Possible fight / altercation", "security", ["CAM-1"])
            with self.assertRaises(LicenseError):
                lm.assert_event_allowed("CAM-1", "Person down / medical", "emergency", ["CAM-1"])

    def test_camera_certificate_gates_detector(self):
        certdir = self.td / "certs"; certdir.mkdir()
        payload = {
            "certificate_id": "C1", "camera_id": "CAM-1",
            "not_before": time.time() - 10, "expires_at": time.time() + 3600,
            "assurance_score": 88,
            "detectors": {"violence": {"status": "certified", "score": 85},
                          "person_down": {"status": "unsupported", "score": 0}},
        }
        (certdir / "cam1.json").write_text(json.dumps(sign_envelope(CERT_SCHEMA, payload, load_private_key(self.private))))
        names = ("AICOS_REQUIRE_CAMERA_CERT", "AICOS_CAMERA_CERT_DIR", "AICOS_CERT_PUBLIC_KEY")
        with EnvGuard(*names):
            os.environ["AICOS_REQUIRE_CAMERA_CERT"] = "true"
            os.environ["AICOS_CAMERA_CERT_DIR"] = str(certdir)
            os.environ["AICOS_CERT_PUBLIC_KEY"] = str(self.public)
            cr = CameraCertificateRegistry()
            cr.assert_event_allowed("CAM-1", "Possible fight / altercation", "security")
            with self.assertRaises(CameraCertificateError):
                cr.assert_event_allowed("CAM-1", "Person down / possible medical emergency", "emergency")
            with self.assertRaises(CameraCertificateError):
                cr.assert_event_allowed("CAM-X", "Possible fight / altercation", "security")

    def _db(self):
        conn = sqlite3.connect(self.td / "db.sqlite", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE events (
              id INTEGER PRIMARY KEY AUTOINCREMENT, event_uuid TEXT, ts REAL NOT NULL,
              source_id TEXT NOT NULL, agent TEXT NOT NULL, type TEXT NOT NULL,
              severity INTEGER NOT NULL, confidence REAL NOT NULL, district TEXT DEFAULT '',
              lat REAL DEFAULT 0, lon REAL DEFAULT 0, status TEXT DEFAULT 'detected',
              extra TEXT DEFAULT '{}', has_snapshot INTEGER DEFAULT 0, has_clip INTEGER DEFAULT 0
            );
        """)
        return conn

    def test_evidence_signing_detects_media_tamper(self):
        conn = self._db(); lock = threading.RLock(); snaps = self.td / "snaps"; snaps.mkdir()
        now = time.time()
        cur = conn.execute("INSERT INTO events(event_uuid,ts,source_id,agent,type,severity,confidence,extra,has_snapshot) VALUES(?,?,?,?,?,?,?,?,1)",
                           ("uuid-1", now, "CAM-1", "security", "Fight", 4, .9, "{}"))
        event_id = cur.lastrowid; conn.commit()
        # Minimal JPEG magic + payload is enough for evidence hashing; this unit test is not an image decoder test.
        (snaps / f"{event_id}.jpg").write_bytes(b"\xff\xd8original")
        names = ("AICOS_EVIDENCE_PRIVATE_KEY", "AICOS_EVIDENCE_PUBLIC_KEY", "AICOS_EVIDENCE_SIGNING_REQUIRED")
        with EnvGuard(*names):
            os.environ["AICOS_EVIDENCE_PRIVATE_KEY"] = str(self.private)
            os.environ["AICOS_EVIDENCE_PUBLIC_KEY"] = str(self.public)
            os.environ["AICOS_EVIDENCE_SIGNING_REQUIRED"] = "true"
            em = EvidenceManager(conn, lock, str(snaps))
            first = em.sign_event(event_id)
            self.assertTrue(first["manifest_hash"])
            self.assertTrue(em.verify(event_id)["ok"])
            # Same content is idempotent.
            self.assertTrue(em.sign_event(event_id).get("duplicate"))
            chain = em.verify_chain()
            self.assertTrue(chain["ok"])
            package = em.export_package(event_id)
            self.assertGreater(len(package), 100)
            (snaps / f"{event_id}.jpg").write_bytes(b"\xff\xd8tampered")
            result = em.verify(event_id)
            self.assertFalse(result["ok"])
            self.assertIn("snapshot hash mismatch", result["reason"])
        conn.close()

    def test_forensic_keyword_search(self):
        conn = self._db(); lock = threading.RLock(); snaps = self.td / "snaps2"; snaps.mkdir()
        conn.execute("INSERT INTO events(ts,source_id,agent,type,severity,confidence,district,status,extra) VALUES(?,?,?,?,?,?,?,?,?)",
                     (time.time(), "CAM-A", "security", "Possible fight / altercation", 4, .88, "Central", "approved", '{"zone":"Gate A"}'))
        conn.execute("INSERT INTO events(ts,source_id,agent,type,severity,confidence,district,status,extra) VALUES(?,?,?,?,?,?,?,?,?)",
                     (time.time(), "CAM-B", "emergency", "Person down", 4, .91, "North", "detected", '{}'))
        conn.commit()
        fi = ForensicIndex(conn, lock, str(snaps))
        fi.index_event(1); fi.index_event(2)
        r = fi.search(q="fight", min_severity=3)
        self.assertEqual(len(r["events"]), 1)
        self.assertEqual(r["events"][0]["source_id"], "CAM-A")
        r2 = fi.search(district="North")
        self.assertEqual(len(r2["events"]), 1)
        self.assertEqual(r2["events"][0]["source_id"], "CAM-B")
        inserted = fi.add_observations("CAM-A", [
            {"ts": time.time(), "kind": "person", "confidence": .92, "color": "red",
             "direction": "right", "bbox": [.1,.2,.2,.5], "extra": {"privacy":"non-biometric"}},
            {"ts": time.time(), "kind": "vehicle", "confidence": .85, "color": "blue",
             "direction": "left", "bbox": [.4,.3,.3,.2], "extra": {}}
        ])
        self.assertEqual(inserted, 2)
        obj = fi.search_observations(kind="person", color="red")
        self.assertEqual(len(obj["observations"]), 1)
        self.assertEqual(obj["observations"][0]["direction"], "right")
        # Non-biometric cross-camera candidate journey: configured adjacency +
        # time window + coarse appearance. This never asserts identity.
        t0 = time.time()
        fi.add_observations("CAM-A", [{"ts": t0, "kind":"person", "confidence":.93, "color":"red",
            "direction":"right", "track_id":"A-7", "appearance_key":"red-dark", "bbox":[.1,.1,.2,.5]}])
        src = fi.search_observations(source_id="CAM-A", track_id="A-7")["observations"][0]
        fi.add_observations("CAM-C", [{"ts": t0+30, "kind":"person", "confidence":.90, "color":"red",
            "direction":"right", "appearance_key":"red-dark", "bbox":[.2,.1,.2,.5]}])
        fi.set_camera_link("CAM-A", "CAM-C", min_seconds=5, max_seconds=90)
        journey = fi.journey_candidates(src["id"])
        self.assertEqual(len(journey["candidates"]), 1)
        self.assertFalse(journey["identity_claim"])
        self.assertGreater(journey["candidates"][0]["score"], .8)
        conn.close()


if __name__ == "__main__":
    unittest.main()
