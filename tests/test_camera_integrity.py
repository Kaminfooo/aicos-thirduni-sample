import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge"))
sys.path.insert(0, str(ROOT / "backend"))

from camera_integrity import IntegrityMonitor, compare_frames
from camera_registry import CameraRegistry


def scene():
    im = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(im, (60, 60), (580, 420), (180, 180, 180), 3)
    for x in range(90, 570, 60):
        cv2.circle(im, (x, 150 + (x % 120)), 18, (255, 255, 255), -1)
        cv2.putText(im, str(x), (x-20, 340), cv2.FONT_HERSHEY_SIMPLEX, .7, (220,220,220), 2)
    cv2.line(im, (80, 390), (560, 80), (255,255,255), 4)
    return im


class CameraIntegrityTests(unittest.TestCase):
    def test_identical_scene_is_ok_and_shift_is_flagged(self):
        base = scene()
        same = base.copy()
        ok = compare_frames(base, same)
        self.assertEqual(ok["status"], "ok")
        M = np.float32([[1, 0, 90], [0, 1, 30]])
        shifted = cv2.warpAffine(base, M, (640, 480))
        result = compare_frames(base, shifted)
        self.assertIn(result["status"], {"warning", "drift_suspected"})

    def test_monitor_baseline_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            mon = IntegrityMonitor("CAM-1", td)
            first = mon.assess(scene())
            self.assertEqual(first["status"], "baseline")
            second = mon.assess(scene())
            self.assertEqual(second["status"], "ok")

    def test_camera_registry_heartbeat_and_stale_state(self):
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        lock = threading.RLock()
        reg = CameraRegistry(conn, lock)
        item = reg.heartbeat("CAM-1", {"name":"Gate", "analysis_fps":12.5, "integrity_status":"ok", "integrity_score":97})
        self.assertTrue(item["online"])
        self.assertEqual(item["integrity_score"], 97.0)
        listing = reg.list()
        self.assertEqual(listing["online"], 1)
        conn.close()


if __name__ == "__main__":
    unittest.main()
