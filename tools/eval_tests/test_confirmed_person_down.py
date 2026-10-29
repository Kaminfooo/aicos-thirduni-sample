"""Persistent side-view confirmation rejects bends, short glimpses and tracking gaps."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge'))
from confirmed_person_down import ConfirmedPersonDownDetector


class ConfirmedPersonDownTests(unittest.TestCase):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    def detector(self, box=(20, 30, 100, 60)):
        class Model:
            def best_fall(self, frame): return .9, box
        return ConfirmedPersonDownDetector({'seconds': 3, 'cooldown_seconds': 90}, model=Model())

    def test_persistent_horizontal_model_signal_requires_human_review(self):
        detector = self.detector()
        for n in range(6):
            self.assertEqual(detector.detect(self.frame, now=1000000 + n * .5), [])
        events = detector.detect(self.frame, now=1000003)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['severity'], 4)
        self.assertTrue(events[0]['extra']['human_review_required'])
        self.assertGreater(events[0]['extra']['confirmation']['observations'], 4)
        self.assertEqual(detector.detect(self.frame, now=1000003.5), [])

    def test_nonhorizontal_bending_shape_never_accumulates_hold(self):
        detector = self.detector((20, 20, 60, 80))
        for n in range(20):
            self.assertEqual(detector.detect(self.frame, now=1000000 + n * .5), [])

    def test_long_frame_gap_resets_persistence(self):
        detector = self.detector()
        for timestamp in (0, .5, 1, 1.5, 2, 2.5, 4, 4.5):
            self.assertEqual(detector.detect(self.frame, now=1000000 + timestamp), [])

    def test_large_position_change_does_not_join_two_people(self):
        detector = self.detector()
        for timestamp in (0, .5, 1, 1.5, 2, 2.5):
            detector.detect(self.frame, now=1000000 + timestamp)
        detector._model = type('OtherPerson', (), {'best_fall': lambda self, frame: (.9, (120, 30, 190, 60))})()
        self.assertEqual(detector.detect(self.frame, now=1000003), [])
