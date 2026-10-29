"""Compare preserved v4.1 rules with current rules using controlled detections.

These tests verify software behavior; they do not measure neural-model accuracy.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge'))
import detectors as current
from rtsp_worker import build_detectors, _model_detector_config


def reference(name):
    path = ROOT / '.runtime/starter-review/source/edge' / (name + '.py')
    if not path.is_file():
        path = ROOT / 'tests/fixtures/starter-source/edge' / (name + '.py')
    spec = importlib.util.spec_from_file_location('starter_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreservedRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = reference('detectors')
        cls.frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def compare(self, sequence, config, camera=None):
        outputs = []
        for module in (self.old, current):
            class Engine:
                rows = []
                def infer(self, frame): return self.rows
            engine = Engine()
            cfg = {key: {'enabled': False} for key in ('person_down', 'fight', 'crowd', 'panic',
                   'intrusion', 'loitering', 'abandoned_object', 'snatch', 'vehicles')}
            cfg.update(config)
            suite = module.YoloBehaviorSuite(camera=camera or {}, config=cfg, engine=engine)
            events = []
            for timestamp, rows in sequence:
                engine.rows = rows
                events += suite.detect(self.frame, now=1_000_000 + timestamp)
            outputs.append(events)
        self.assertEqual(outputs[0], outputs[1])
        return outputs[1]

    @staticmethod
    def people(count):
        return [{'box': (10 + n * 35, 20, 30 + n * 35, 150), 'cls': 0, 'conf': .9} for n in range(count)]

    def test_crowd_warn_and_cooldown_preserved(self):
        rows = self.people(8)
        events = self.compare([(0, rows), (1, rows)], {'crowd': {'enabled': True}})
        self.assertEqual([e['type'] for e in events], ['Crowd build-up'])
        self.assertEqual(events[0]['extra']['count'], 8)

    def test_large_crowd_preserved(self):
        events = self.compare([(0, self.people(15))], {'crowd': {'enabled': True}})
        self.assertEqual(events[0]['type'], 'Large crowd forming')

    def test_fewer_people_do_not_create_crowd_alert(self):
        self.assertEqual(self.compare([(0, self.people(4))], {'crowd': {'enabled': True}}), [])

    def test_person_down_transition_preserved(self):
        upright = [{'box': (100, 20, 140, 200), 'cls': 0, 'conf': .9}]
        down = [{'box': (60, 150, 200, 190), 'cls': 0, 'conf': .9}]
        seq = [(t, upright if t < 4 else down) for t in range(12)]
        events = self.compare(seq, {'person_down': {'enabled': True, 'seconds': 3}})
        self.assertTrue(any('Person down' in e['type'] for e in events))

    def test_standing_does_not_create_person_down_alert(self):
        self.assertEqual(self.compare([(t, self.people(1)) for t in range(12)],
                                     {'person_down': {'enabled': True}}), [])

    def test_intrusion_zone_and_pixel_location_preserved(self):
        zone = {'name': 'test zone', 'type': 'intrusion', 'points': [[0, 0], [1, 0], [1, 1], [0, 1]]}
        events = self.compare([(0, self.people(1))], {'intrusion': {'enabled': True}}, {'zones': [zone]})
        self.assertIn('Intrusion', events[0]['type'])
        self.assertEqual(len(events[0]['extra']['px']), 2)

    def test_loitering_timer_preserved(self):
        zone = {'name': 'test zone', 'type': 'loiter', 'points': [[0, 0], [1, 0], [1, 1], [0, 1]]}
        events = self.compare([(t, self.people(1)) for t in range(5)],
                              {'loitering': {'enabled': True, 'seconds': 2}}, {'zones': [zone]})
        self.assertTrue(any('Loitering' in e['type'] for e in events))

    def test_stopped_vehicles_and_congestion_preserved(self):
        vehicles = [{'box': (x, 180, x + 80, 240), 'cls': 2, 'conf': .9} for x in (100, 300)]
        events = self.compare([(t, vehicles) for t in range(7)],
                              {'vehicles': {'enabled': True, 'stopped_seconds': 2, 'congestion_count': 2}})
        self.assertTrue(any('stopped' in e['type'] for e in events))
        self.assertTrue(any('congestion' in e['type'] for e in events))

    def test_wrong_way_motion_preserved(self):
        seq = [(t, [{'box': (300 - t * 20, 180, 380 - t * 20, 240), 'cls': 2, 'conf': .9}])
               for t in range(4)]
        events = self.compare(seq, {'vehicles': {'enabled': True,
                              'wrong_way': {'enabled': True, 'direction': 'right', 'min_speed': .01}}})
        self.assertTrue(any('against traffic' in e['type'] for e in events))

    def test_violence_persistence_and_cooldown_preserved(self):
        import violence
        old = reference('violence')
        class Model:
            def best_score(self, frame, class_id): return .9
        outputs = []
        for module in (old, violence):
            detector = module.ViolenceDetector(model=Model())
            outputs.append([event for t in range(16)
                            for event in detector.detect(self.frame, now=1_000_000 + t)])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(outputs[0]), 1)


class ConfigurationContracts(unittest.TestCase):
    def test_original_nested_models_are_loaded(self):
        settings = {'detectors': {'motion': {'enabled': False}, 'yolo': {'enabled': False,
                    'fall_model': {'enabled': True}, 'violence': {'enabled': True}}}}
        self.assertEqual([d.name for d in build_detectors(settings, {})], ['violence-model', 'fall-model'])

    def test_explicit_motion_disable_is_respected(self):
        self.assertEqual(build_detectors({'detectors': {'motion': {'enabled': False}}}, {}), [])

    def test_unconfigured_demo_retains_original_default(self):
        self.assertEqual([d.name for d in build_detectors({}, {})], ['motion-demo'])

    def test_camera_overrides_global_and_canonical_overrides_legacy(self):
        settings = {'detectors': {'yolo': {'fall_model': {'conf': .2, 'seconds': 2}},
                                  'fall_model': {'conf': .4}}}
        camera = {'detectors': {'yolo': {'fall_model': {'seconds': 5}}, 'fall_model': {'conf': .7}}}
        self.assertEqual(_model_detector_config(settings, camera, 'fall_model'), {'conf': .7, 'seconds': 5})

    def test_pose_model_cannot_silently_enable_vehicle_classes(self):
        fake = types.SimpleNamespace(YOLO=lambda path: types.SimpleNamespace(names={0: 'person'}))
        with patch.dict(sys.modules, {'ultralytics': fake}):
            with self.assertRaisesRegex(ValueError, 'Model lacks requested class IDs'):
                current.UltralyticsEngine('local-test.pt', .35, 320, classes=[0, 2, 3, 5, 7])

    def test_compatible_object_model_accepts_vehicle_classes(self):
        fake = types.SimpleNamespace(YOLO=lambda path: types.SimpleNamespace(names={0: 'person', 2: 'car'}))
        with patch.dict(sys.modules, {'ultralytics': fake}):
            engine = current.UltralyticsEngine('local-test.pt', .35, 320, classes=[0, 2])
            self.assertEqual(engine.classes, [0, 2])

    def test_matching_numeric_ids_with_wrong_labels_are_rejected(self):
        fake = types.SimpleNamespace(YOLO=lambda path: types.SimpleNamespace(names={0: 'person', 2: 'fallen'}))
        with patch.dict(sys.modules, {'ultralytics': fake}):
            with self.assertRaisesRegex(ValueError, 'Model class labels do not match'):
                current.UltralyticsEngine('local-test.pt', .35, 320, classes=[0, 2])
