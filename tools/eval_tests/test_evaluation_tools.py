"""Targeted evaluation-tool regressions; no camera, models, or network."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'edge'))
import evaluate_person_down as evaluate
import capture_eval_clip as capture
from eval_support import clip_metadata, frame_time, sha256, sidecar


class EvaluationToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def video(self, frames=20):
        path = self.directory / 'clip.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 24))
        self.assertTrue(writer.isOpened())
        for n in range(frames):
            writer.write(np.full((24, 32, 3), n, dtype=np.uint8))
        writer.release()
        return path

    def args(self):
        return SimpleNamespace(every=1, stop_after_first=True, mode='heuristic',
                               effective={'rotate': 0, 'resize_width': 640})

    def test_unreadable_clip_is_error_without_loading_a_model(self):
        path = self.directory / 'broken.mp4'
        path.write_bytes(b'not a video')
        with patch.object(evaluate, 'make_detectors') as engine:
            result = evaluate.evaluate_video(path, self.args())
        engine.assert_not_called()
        self.assertIn('error', result)
        self.assertIsNone(result['alerted'])
        result['expected_positive'] = False
        summary = evaluate.metrics([result])
        self.assertEqual((summary['error_clips'], summary['valid_clips'], summary['tn']), (1, 0, 0))

    def test_total_miss_has_zero_f1(self):
        result = evaluate.metrics([{'expected_positive': True, 'alerted': False}])
        self.assertEqual(result['fn'], 1)
        self.assertEqual(result['f1'], 0)

    def test_missing_onset_does_not_invent_latency(self):
        result = evaluate.metrics([{'expected_positive': True, 'alerted': True,
                                    'first_alert_seconds': 5, 'onset_to_alert_seconds': None}])
        self.assertIsNone(result['mean_time_to_alert_seconds'])
        self.assertEqual(result['positive_clips_without_onset'], 1)
        self.assertEqual(result['mean_clip_start_to_alert_seconds'], 5)

    def test_capture_gaps_preserve_time_and_onset_latency(self):
        path = self.video(6)
        times = [0, 0.1, 0.2, 1.4, 1.5, 1.6]
        sidecar(path, 'capture').write_text(json.dumps({'status': 'valid', 'video_sha256': sha256(path),
                                                       'frame_times_seconds': times, 'max_gap_seconds': 1.2}))
        sidecar(path, 'labels').write_text(json.dumps({'video_sha256': sha256(path), 'onset_frame': 3}))
        class Detector:
            def detect(self, frame, previous, now):
                return [{'type': 'Person down', 'severity': 4}] if now >= evaluate.CLOCK_BASE + 1.5 else []
        with patch.object(evaluate, 'make_detectors', return_value=[Detector()]):
            row = evaluate.evaluate_video(path, self.args())
        self.assertNotIn('error', row)
        self.assertEqual(row['first_alert_seconds'], 1.5)
        self.assertAlmostEqual(row['onset_to_alert_seconds'], 0.1)
        self.assertEqual(row['frames_decoded'], 6)
        self.assertEqual(row['timeline_source'], 'capture_receipt_time')

    def test_pre_onset_alert_is_not_the_successful_alert(self):
        path = self.video()
        sidecar(path, 'labels').write_text(json.dumps({'video_sha256': sha256(path), 'onset_frame': 10}))
        class Detector:
            def detect(self, frame, previous, now):
                return [{'type': 'Person down'}]
        with patch.object(evaluate, 'make_detectors', return_value=[Detector()]):
            row = evaluate.evaluate_video(path, self.args())
        self.assertEqual(len(row['early_alerts']), 10)
        self.assertEqual(row['first_alert_seconds'], 1.0)
        self.assertEqual(row['onset_to_alert_seconds'], 0)

    def test_truncated_tail_is_not_hidden_by_stop_after_first(self):
        path = self.directory / 'truncated.mp4'
        path.write_bytes(b'fake')
        class FakeCapture:
            count = 0
            def isOpened(self): return True
            def get(self, field): return 10 if field == cv2.CAP_PROP_FPS else 4
            def read(self):
                self.count += 1
                return (True, np.zeros((24, 32, 3), dtype=np.uint8)) if self.count <= 2 else (False, None)
            def release(self): pass
        class Detector:
            def detect(self, frame, previous, now): return [{'type': 'Person down'}]
        with patch.object(cv2, 'VideoCapture', return_value=FakeCapture()), patch.object(evaluate, 'make_detectors', return_value=[Detector()]):
            row = evaluate.evaluate_video(path, self.args())
        self.assertIn('truncated', row['error'])
        self.assertIsNone(row['alerted'])

    def test_tampered_video_and_bad_timestamps_are_rejected(self):
        path = self.video(3)
        meta = {'status': 'valid', 'video_sha256': sha256(path), 'frame_times_seconds': [0, 0.1, 0.1]}
        sidecar(path, 'capture').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            clip_metadata(path, 10, 3)
        meta.update(video_sha256='wrong', frame_times_seconds=[0, 0.1, 0.2])
        sidecar(path, 'capture').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            clip_metadata(path, 10, 3)

    def test_phone_configuration_is_inherited_without_loading_models(self):
        weights = self.directory / 'local.pt'
        weights.write_bytes(b'only a path fixture; never loaded')
        args = SimpleNamespace(config=ROOT / 'config/cameras-phone.yaml', camera_id='CAM-LIVE',
                               model=str(weights), conf=None, imgsz=None, seconds=None, mode='heuristic',
                               fall_weights=None, fall_conf=.35, fall_imgsz=640, fall_seconds=3,
                               resize_width=None, every=3)
        result = evaluate.effective_settings(args)
        self.assertEqual(result['yolo']['conf'], .30)
        self.assertEqual(result['yolo']['imgsz'], 320)
        self.assertEqual(result['yolo']['person_down']['seconds'], 3)
        self.assertEqual(result['yolo']['person_down']['gap_tolerance'], 4)
        self.assertEqual(result['resize_width'], 480)
        self.assertFalse(result['yolo']['fight']['enabled'])
        args.model = str(self.directory / 'missing.pt')
        with self.assertRaisesRegex(ValueError, 'automatic download is disabled'):
            evaluate.effective_settings(args)

    def test_capture_writes_decodable_video_and_receipt_timestamps(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        class Camera:
            def isOpened(self): return True
            def read(self): return True, image
            def release(self): pass
        class Grabber:
            stop = threading.Event()
            _thread = SimpleNamespace(is_alive=lambda: False)
            def __init__(self, cap): pass
            def get(self): return image.copy()
            def close(self): pass
        actual_capture = cv2.VideoCapture
        def open_video(source, *args):
            return Camera() if source == 'synthetic-camera' else actual_capture(source, *args)
        clock = iter(n / 10 for n in range(100))
        args = ['capture', '--url', 'synthetic-camera', '--label', 'negative', '--name', 'synthetic',
                '--seconds', '1', '--lead-in', '0', '--fps', '10', '--dataset', str(self.directory)]
        with patch.object(sys, 'argv', args), patch.object(cv2, 'VideoCapture', side_effect=open_video), \
             patch('rtsp_worker.LatestFrame', Grabber), patch.object(capture.time, 'monotonic', side_effect=lambda: next(clock)), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(capture.main(), 0)
        video = next((self.directory / 'negative').glob('*.mp4'))
        meta = json.loads(sidecar(video, 'capture').read_text())
        self.assertEqual(meta['status'], 'valid')
        self.assertEqual(meta['frames_written'], meta['frames_decoded'])
        self.assertEqual(meta['video_sha256'], sha256(video))
        parsed = clip_metadata(video, 10, meta['frames_written'])
        self.assertEqual(frame_time(parsed, 1, 10), meta['frame_times_seconds'][1])
        self.assertNotAlmostEqual(meta['encoded_duration_seconds'], meta['capture_wall_seconds'])

    def test_dead_stream_aborts_countdown_without_recording_stale_frame(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        class Camera:
            def isOpened(self): return True
            def read(self): return True, image
            def release(self): pass
        class DeadGrabber:
            _thread = SimpleNamespace(is_alive=lambda: False)
            def __init__(self, cap):
                self.stop = threading.Event()
                self.stop.set()
            def get(self):
                raise AssertionError('Must not record stale data from a dead countdown stream')
            def close(self): pass
        actual_capture = cv2.VideoCapture
        def open_video(source, *args):
            return Camera() if source == 'synthetic-camera' else actual_capture(source, *args)
        argv = ['capture', '--url', 'synthetic-camera', '--label', 'negative', '--name', 'dead',
                '--seconds', '1', '--lead-in', '10', '--dataset', str(self.directory)]
        with patch.object(sys, 'argv', argv), patch.object(cv2, 'VideoCapture', side_effect=open_video), \
             patch('rtsp_worker.LatestFrame', DeadGrabber), patch.object(capture.time, 'sleep') as sleep, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(capture.main(), 5)
        sleep.assert_not_called()
        meta = json.loads(next((self.directory / 'negative').glob('*.capture.json')).read_text())
        self.assertEqual(meta['status'], 'invalid')
        self.assertTrue(meta['stream_ended_during_countdown'])
        self.assertEqual(meta['frames_written'], 0)

    def test_interval_onset_never_claims_exact_latency_or_credits_ambiguous_alert(self):
        path = self.video()
        sidecar(path, 'labels').write_text(json.dumps({'video_sha256': sha256(path),
            'onset_interval_frames': [5, 10], 'method': 'interval review'}))
        class Detector:
            def detect(self, frame, previous, now): return [{'type': 'Person down'}]
        with patch.object(evaluate, 'make_detectors', return_value=[Detector()]):
            row = evaluate.evaluate_video(path, self.args())
        self.assertIsNone(row['onset_seconds'])
        self.assertIsNone(row['onset_to_alert_seconds'])
        self.assertEqual(row['first_alert_seconds'], 1)
        self.assertEqual(row['onset_to_alert_interval_seconds'], [0, .5])
        self.assertEqual(len(row['early_alerts']), 5)
        self.assertEqual(len(row['ambiguous_alerts']), 5)

    def test_disabled_model_is_an_error_instead_of_a_false_negative(self):
        path = self.video()
        class BrokenDetector:
            _disabled = False
            def detect(self, frame, previous, now):
                self._disabled = True
                return []
        with patch.object(evaluate, 'make_detectors', return_value=[BrokenDetector()]):
            row = evaluate.evaluate_video(path, self.args())
        self.assertIsNone(row['alerted'])
        self.assertIn('disabled itself', row['error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
