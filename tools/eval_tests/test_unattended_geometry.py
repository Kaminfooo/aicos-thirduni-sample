"""Controlled reproductions for unattended-object association/proximity bugs."""
import unittest
import numpy as np
from test_detection_rule_coverage import suite,person,bag,START


class UnattendedGeometry(unittest.TestCase):
    frame=np.zeros((480,640,3),dtype=np.uint8)

    def test_bag_beside_standing_person_is_attended(self):
        detector,engine=suite('abandoned_object',{'seconds':2})
        engine.rows=[person(x=200,y=150,height=240,width=40),bag(x=230,y=275)]
        for second in (0,1,2,3):
            self.assertEqual(detector.detect(self.frame,now=START+second),[],
                'Bag beside a person must not become unattended because their torso center is far above the bag')

    def test_two_close_bags_keep_separate_state(self):
        detector,engine=suite('abandoned_object',{'seconds':2})
        engine.rows=[bag(x=200),bag(x=210)]
        detector.detect(self.frame,now=START)
        self.assertEqual(len(detector._objects),2,'One track must not absorb two detections in the same frame')

    def test_distant_person_does_not_block_unattended_alert(self):
        detector,engine=suite('abandoned_object',{'seconds':2})
        engine.rows=[person(x=500,y=150,height=240),bag(x=100,y=275)]
        for second in (0,1):self.assertFalse(detector.detect(self.frame,now=START+second))
        self.assertEqual(len(detector.detect(self.frame,now=START+2)),1)

    def test_nearby_person_leaves_then_full_hold_is_required(self):
        detector,engine=suite('abandoned_object',{'seconds':2})
        for second in (0,1,2):
            engine.rows=[person(x=200,y=150,height=240,width=40),bag(x=230,y=275)]
            self.assertFalse(detector.detect(self.frame,now=START+second))
        engine.rows=[bag(x=230,y=275)]
        self.assertFalse(detector.detect(self.frame,now=START+3))
        self.assertEqual(len(detector.detect(self.frame,now=START+4)),1)
