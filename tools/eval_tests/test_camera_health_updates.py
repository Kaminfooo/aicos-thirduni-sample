"""Camera-health failures and frame freshness stay distinct from heartbeat liveness."""
from pathlib import Path
import sqlite3
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge'))
sys.path.insert(0, str(ROOT / 'backend'))
from camera_health import ObservedDetector
from camera_registry import CameraRegistry


class CameraHealthUpdateTests(unittest.TestCase):
    def test_disabled_detector_is_explicitly_failed(self):
        class Detector:
            name = 'fall'
            _disabled = False
            def detect(self, frame, previous): self._disabled = True; return []
        observed = ObservedDetector(Detector())
        self.assertEqual(observed.detect(None, None), [])
        self.assertEqual(observed.status, 'failed')

    def registry(self):
        conn = sqlite3.connect(':memory:', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        return CameraRegistry(conn, threading.RLock())

    def test_fresh_heartbeat_does_not_hide_failed_inference(self):
        row = self.registry().heartbeat('C', {'extra': {'detector_status': 'failed', 'frame_age_seconds_at_heartbeat': 0}})
        self.assertTrue(row['online'])
        self.assertEqual(row['operational_status'], 'detector_failed')

    def test_fresh_heartbeat_does_not_hide_stale_frames(self):
        row = self.registry().heartbeat('C', {'extra': {'frame_age_seconds_at_heartbeat': 20}})
        self.assertTrue(row['online'])
        self.assertEqual(row['operational_status'], 'frames_stale')

    def test_legacy_and_invalid_frame_ages_remain_safe(self):
        registry = self.registry()
        for extra in ({}, {'frame_age_seconds_at_heartbeat': 'nan'}, {'frame_age_seconds_at_heartbeat': []}):
            row = registry.heartbeat('C', {'extra': extra})
            self.assertIsNone(row['frame_age_seconds'])
            self.assertEqual(row['operational_status'], 'online')
