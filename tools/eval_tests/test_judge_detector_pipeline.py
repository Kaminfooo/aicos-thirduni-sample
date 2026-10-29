"""Adapter orchestration with controlled detections; not recognition accuracy."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.judge_detector_pipeline import Pipeline, weights as verified_weights
from deploy.judge_capabilities import IDS


class Engines:
    def __init__(self,rows=None):self.rows=rows or {};self.models={}
    def begin(self):pass
    def get(self,name):
        parent=self
        class Engine:
            def infer(self,frame):
                result=parent.rows.get(name,[])
                if isinstance(result,Exception):raise result
                return result
        return Engine()


def settings(keys):
    return dict(detectors=keys,region=[.1,.1,.9,.9],crowd_warn=8,
                loiter_seconds=120,stopped_seconds=45,direction='')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.zeros((270,480,3),np.uint8)
        # Controlled engines do not load weights. Keep these unit checks portable
        # without weakening the real resolver used by actual inference.
        substitute=patch('deploy.judge_detector_pipeline.weights',side_effect=lambda name:'unit-test-only/'+name)
        substitute.start();self.addCleanup(substitute.stop)

    def test_real_weight_resolver_rejects_missing_file(self):
        with self.assertRaisesRegex(RuntimeError,'Missing or changed'):
            verified_weights('not-a-reference-model.pt')

    def test_all_adapters_construct_and_monitor_without_fabricated_alerts(self):
        p=Pipeline(settings(list(IDS)),Engines())
        for second in (0,.34):
            result=p.detect(self.frame,second)
            self.assertEqual({r['id'] for r in result['detectors']},set(IDS))
            self.assertFalse([r for r in result['detectors'] if r['status']=='error'])
            self.assertEqual(result['events'],[])

    def test_fire_and_smoke_both_retained_after_real_hold(self):
        engine=Engines({'fire-smoke-yolo26n.pt':[
            {'label':k,'conf':.9,'box':[50,50,150,150]} for k in ('fire','smoke')]})
        p=Pipeline(settings(['fire_smoke']),engine)
        for second in (0,.5,1,1.5):self.assertFalse(p.detect(self.frame,second)['events'])
        events=p.detect(self.frame,2)['events']
        self.assertEqual({e['extra']['label'] for e in events},{'fire','smoke'})
        self.assertTrue(all(e['extra']['judge_capability']=='fire_smoke' for e in events))

    def test_sessions_do_not_share_temporal_state(self):
        engine=Engines({'fire-smoke-yolo26n.pt':[{'label':'fire','conf':.9,'box':[50,50,150,150]}]})
        first=Pipeline(settings(['fire_smoke']),engine)
        second=Pipeline(settings(['fire_smoke']),engine)
        for stamp in (0,.5,1,1.5,2):result=first.detect(self.frame,stamp)
        self.assertTrue(result['events'])
        self.assertFalse(second.detect(self.frame,2)['events'])

    def test_failed_model_is_visible_while_other_detector_continues(self):
        p=Pipeline(settings(['fire_smoke','intrusion']),Engines({
            'fire-smoke-yolo26n.pt':RuntimeError('Test failure'),
            'yolo26n.pt':[{'box':[100,50,200,200],'conf':.9,'cls':0,'kpts':None}]}))
        result=p.detect(self.frame,0)
        self.assertEqual(result['detectors'][0]['status'],'error')
        self.assertEqual(result['events'][0]['extra']['judge_capability'],'intrusion')

    def test_unknown_selection_rejected(self):
        with self.assertRaises(ValueError):Pipeline(settings(['unknown']),Engines())
