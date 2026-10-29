"""Temporal observations remain conservative across lost frames and unknown causes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'edge'))
from temporal_posture import TemporalPostureObserver


def person(posture='upright', offset=0):
    points = [[0, 0, .95] for _ in range(17)]
    if posture == 'upright':
        box, shoulder, hip, ankle = [70, 10, 110, 210], [90, 60], [90, 110], [90, 200]
    elif posture == 'lying':
        box, shoulder, hip, ankle = [70, 160, 220, 210], [90, 180], [130, 180], [200, 190]
    else:
        box, shoulder, hip, ankle = [70, 90, 160, 210], [100, 130], [110, 160], [140, 190]
    for indices, position in (((5, 6), shoulder), ((11, 12), hip), ((15, 16), ankle)):
        for i in indices:
            points[i] = [position[0] + offset, position[1], .95]
    return {'box': [box[0] + offset, box[1], box[2] + offset, box[3]],
            'kpts': points, 'conf': .95, 'cls': 0, 'lying': posture == 'lying'}


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.observer = TemporalPostureObserver()

    def observe(self, posture, timestamp):
        row = person(posture)
        return self.observer.update([row], timestamp, lambda d: d['lying'])[id(row)]

    def upright_reference(self):
        for t in (0, .2, .4, .6):
            self.observe('upright', t)

    def test_observed_rapid_descent_is_not_a_medical_diagnosis(self):
        self.upright_reference()
        result = self.observe('lying', 1)
        self.assertEqual(result['transition']['kind'], 'abrupt_collapse_observed')
        self.assertEqual(result['medical_cause'], 'unknown')
        self.assertEqual(result['consciousness'], 'not_established')
        self.assertEqual(result['intent'], 'not_established')

    def test_gradual_lowering_has_its_own_observation(self):
        self.upright_reference()
        for t in (1, 1.5, 2, 2.5, 3):
            self.observe('transition', t)
        result = self.observe('lying', 3.5)
        self.assertEqual(result['transition']['kind'], 'gradual_lowering_observed')

    def test_person_already_down_does_not_invent_a_collapse(self):
        result = self.observe('lying', 0)
        self.assertEqual(result['transition']['kind'], 'already_down_transition_not_observed')

    def test_gap_before_descent_prevents_rapid_collapse_claim(self):
        self.upright_reference()
        result = self.observe('lying', 1.5)
        self.assertEqual(result['transition']['kind'], 'transition_uncertain')

    def test_missing_joints_before_descent_prevent_a_collapse_claim(self):
        self.upright_reference()
        row = person('transition')
        row['kpts'][15][2] = .1
        self.observer.update([row], .8, lambda d: d['lying'])
        self.assertEqual(self.observe('lying', 1)['transition']['kind'], 'transition_uncertain')

    def test_new_person_after_long_track_loss_has_no_inherited_transition(self):
        self.upright_reference()
        self.assertEqual(self.observe('lying', 8)['transition']['kind'], 'already_down_transition_not_observed')

    def test_stable_return_to_upright_records_recovery(self):
        self.upright_reference()
        self.observe('lying', 1)
        for t in (1.5, 1.8, 2.1):
            result = self.observe('upright', t)
        self.assertTrue(result['recovery_observed'])
        self.assertEqual(result['posture'], 'upright')
