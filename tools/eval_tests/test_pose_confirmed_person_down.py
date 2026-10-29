"""Pose confirmation requires a whole supported posture and an unbroken hold."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'edge'))
from pose_confirmed_person_down import PoseConfirmedPersonDownDetector


class PoseConfirmationTests(unittest.TestCase):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    def person(self, offset=0):
        points = [[0, 0, .9] for _ in range(17)]
        for index, x, y in ((5, 40, 50), (6, 50, 50), (11, 90, 50),
                            (12, 100, 50), (15, 140, 60), (16, 150, 60)):
            points[index] = [x + offset, y, .9]
        return {'box': [20 + offset, 30, 160 + offset, 80], 'conf': .9, 'cls': 0, 'kpts': points}

    def detector(self, rows=None):
        class Engine:
            def infer(self, frame): return self.rows
        engine = Engine()
        engine.rows = rows if rows is not None else [self.person()]
        return PoseConfirmedPersonDownDetector(engine=engine), engine

    def test_persistent_supported_pose_fires_once_with_human_review(self):
        detector, _ = self.detector()
        for n in range(8):
            self.assertEqual(detector.detect(self.frame, now=1000 + n * .5), [])
        event = detector.detect(self.frame, now=1004)[0]
        self.assertEqual(event['severity'], 4)
        self.assertTrue(event['extra']['human_review_required'])
        self.assertEqual(event['extra']['lead_seconds'], 0)
        self.assertEqual(detector.detect(self.frame, now=1005), [])

    def test_horizontal_torso_with_vertical_legs_does_not_confirm_bending(self):
        person = self.person()
        person['kpts'][15], person['kpts'][16] = [90, 90, .9], [100, 90, .9]
        detector, _ = self.detector([person])
        for n in range(12):
            self.assertEqual(detector.detect(self.frame, now=1000 + n), [])

    def test_missing_ankle_confidence_does_not_fabricate_a_lying_pose(self):
        person = self.person()
        person['kpts'][15][2] = .1
        detector, _ = self.detector([person])
        self.assertEqual(detector.detect(self.frame, now=1000), [])
        self.assertEqual(detector.last_decision['candidates'], 0)

    def test_candidate_alignment_allows_projected_lying_torso_without_changing_default(self):
        person = self.person()
        for a, b, x, y in ((5, 6, 45, 30), (11, 12, 85, 75), (15, 16, 135, 80)):
            person['kpts'][a], person['kpts'][b] = [x - 5, y, .9], [x + 5, y, .9]
        strict = PoseConfirmedPersonDownDetector()
        candidate = PoseConfirmedPersonDownDetector({'orientation_mode': 'aligned_segments'})
        self.assertFalse(strict._qualifies(person, 20000))
        self.assertTrue(candidate._qualifies(person, 20000))

    def test_candidate_alignment_rejects_folded_legs_and_missing_joint_evidence(self):
        candidate = PoseConfirmedPersonDownDetector({'orientation_mode': 'aligned_segments'})
        person = self.person()
        person['kpts'][15], person['kpts'][16] = [90, 90, .9], [100, 90, .9]
        self.assertFalse(candidate._qualifies(person, 20000))
        person = self.person()
        person['kpts'][16][2] = .1
        self.assertFalse(candidate._qualifies(person, 20000))

    def test_unknown_orientation_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'orientation mode'):
            PoseConfirmedPersonDownDetector({'orientation_mode': 'typo'})

    def test_tiny_hand_sized_box_cannot_be_treated_as_a_whole_body(self):
        person = self.person()
        person['box'] = [20, 20, 25, 23]
        detector, _ = self.detector([person])
        self.assertEqual(detector.detect(self.frame, now=1000), [])
        self.assertEqual(detector.last_decision['candidates'], 0)

    def test_long_observation_gap_resets_hold(self):
        detector, _ = self.detector()
        for t in (0, 1, 2, 5, 6, 7):
            self.assertEqual(detector.detect(self.frame, now=1000 + t), [])

    def test_separate_people_do_not_share_confirmation_time(self):
        detector, engine = self.detector()
        wide_frame = np.zeros((100, 400, 3), dtype=np.uint8)
        # Each person is observed for only two seconds. Pooling their timers
        # would incorrectly reach the four-second confirmation threshold.
        for t in range(6):
            engine.rows = [self.person(offset=0 if t < 3 else 150)]
            self.assertEqual(detector.detect(wide_frame, now=1000 + t), [])

    def test_cooldown_survives_a_short_detection_dropout(self):
        detector, engine = self.detector()
        for t in range(5):
            detector.detect(self.frame, now=1000 + t)
        engine.rows = []
        detector.detect(self.frame, now=1008)
        engine.rows = [self.person()]
        for t in range(9, 15):
            self.assertEqual(detector.detect(self.frame, now=1000 + t), [])

    def test_engine_failure_is_visible_and_disables_inference(self):
        detector, engine = self.detector()
        with patch.object(engine, 'infer', side_effect=RuntimeError('test model failure')), patch('pose_confirmed_person_down.log.exception'):
            self.assertEqual(detector.detect(self.frame, now=1000), [])
        self.assertTrue(detector._disabled)
        self.assertEqual(detector.last_decision['reason'], 'model_error')

    def test_replay_exports_confirmation_settings_and_builds_one_detector(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
        from evaluate_person_down import effective_settings, make_detectors
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'local-only.pt'
            model.write_bytes(b'test metadata; not loaded')
            args = SimpleNamespace(mode='pose-confirmed', config=None, camera_id=None,
                model=str(model), conf=.35, imgsz=640, seconds=4, resize_width=480, every=3,
                fall_weights=None, fall_conf=.35, fall_imgsz=640, fall_seconds=3)
            args.effective = effective_settings(args)
            settings = args.effective['pose_confirmation']
            self.assertTrue({'gap_tolerance', 'joint_conf', 'min_body_area', 'max_center_drift',
                             'horizontal_ratio', 'axis_ratio', 'min_observations', 'seconds'} <= settings.keys())
            detectors = make_detectors(args)
            self.assertEqual(len(detectors), 1)
            self.assertEqual(detectors[0].cfg, settings)
