from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'edge'))
from persistent_hazards import PersistentHazardDetector
from site_behaviors import SiteBehaviorSuite


class HazardTests(unittest.TestCase):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    def detector(self, kind='fire_smoke'):
        class Engine:
            rows = []
            def infer(self, frame): return self.rows
        engine = Engine()
        engine.rows = [{'label': 'fire' if kind == 'fire_smoke' else 'violence',
                        'conf': .9, 'box': [10, 10, 90, 70]}]
        return PersistentHazardDetector({'kind': kind}, engine), engine

    def test_fire_requires_persistent_observations_and_human_review(self):
        detector, _ = self.detector()
        for t in (1000, 1000.5, 1001, 1001.5):
            self.assertEqual(detector.detect(self.frame, now=t), [])
        event = detector.detect(self.frame, now=1002)[0]
        self.assertEqual(event['type'], 'Possible fire detected')
        self.assertTrue(event['extra']['human_review_required'])
        self.assertEqual(detector.detect(self.frame, now=1003), [])

    def test_long_gap_cannot_confirm_a_hazard(self):
        detector, _ = self.detector()
        for t in (1000, 1000.5, 1001, 1004, 1004.5):
            self.assertEqual(detector.detect(self.frame, now=t), [])

    def test_low_confidence_or_wrong_class_is_not_accepted(self):
        detector, engine = self.detector()
        engine.rows = [{'label': 'smoke', 'conf': .2, 'box': [10, 10, 90, 70]},
                       {'label': 'person', 'conf': .99, 'box': [10, 10, 90, 70]}]
        for t in range(10):
            self.assertEqual(detector.detect(self.frame, now=1000 + t), [])

    def test_violence_uses_the_same_timing_and_neutral_event_label(self):
        detector, _ = self.detector('violence')
        for t in (1000, 1000.5, 1001, 1001.5):
            detector.detect(self.frame, now=t)
        event = detector.detect(self.frame, now=1002)[0]
        self.assertEqual(event['severity'], 3)
        self.assertEqual(event['type'], 'Possible physical violence detected')

    def test_missing_model_is_a_visible_failure_not_a_quiet_healthy_detector(self):
        detector = PersistentHazardDetector({'weights': 'missing-file.pt'})
        with patch('persistent_hazards.log.exception'):
            self.assertEqual(detector.detect(self.frame, now=1000), [])
        self.assertTrue(detector._disabled)


class SiteTests(unittest.TestCase):
    def test_crowd_count_respects_region_and_requires_dwell(self):
        suite = SiteBehaviorSuite(config={'crowd': {'enabled': True, 'warn': 2, 'critical': 4, 'seconds': 2},
                    'crowd_region': [[0, 0], [.5, 0], [.5, 1], [0, 1]]}, engine=object())
        suite._fw = suite._fh = 100
        people = [SimpleNamespace(feet=lambda: (20, 50)), SimpleNamespace(feet=lambda: (30, 50)),
                  SimpleNamespace(feet=lambda: (80, 50)), SimpleNamespace(feet=lambda: (90, 50))]
        for t in (1000, 1000.5, 1001, 1001.5):
            self.assertEqual(suite._crowd(people, t), [])
        event = suite._crowd(people, 1002)[0]
        self.assertEqual(event['extra']['count'], 2)
        self.assertEqual(event['extra']['intent'], 'not_established')
        self.assertEqual(event['severity'], 2)

    def test_traffic_cannot_run_without_a_road_region(self):
        with self.assertRaisesRegex(ValueError, 'road region'):
            SiteBehaviorSuite(config={'vehicles': {'enabled': True}}, engine=object())

    def test_parking_outside_road_region_is_not_counted(self):
        suite = SiteBehaviorSuite(config={'vehicles': {'enabled': True, 'stopped_seconds': 2},
                      'road_region': [[0, 0], [.5, 0], [.5, 1], [0, 1]]}, engine=object())
        outside = [{'box': [70, 20, 90, 60], 'cls': 2}]
        for t in range(8):
            self.assertEqual(suite._vehicle(outside, 1000 + t, 100, 100), [])
        inside = [{'box': [10, 20, 30, 60], 'cls': 2}]
        events = []
        for t in range(8, 12):
            events += suite._vehicle(inside, 1000 + t, 100, 100)
        self.assertTrue(any('stopped' in e['type'] for e in events))
