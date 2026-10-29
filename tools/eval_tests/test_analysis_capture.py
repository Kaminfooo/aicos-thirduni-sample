"""Bounded capture preserves analysed frames/timing and does not gate detection."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'edge'), str(ROOT / 'tools')]
from analysis_capture import AnalysisCapture
from camera_health import ObservedCameraWorker
from eval_support import clip_metadata


class AnalysisCaptureTests(unittest.TestCase):
    def controller(self, directory):
        root = Path(directory)
        request = root / 'request.json'
        request.write_text(json.dumps({'requested_at': 100, 'lead_in': 5, 'seconds': 10}))
        controller = AnalysisCapture(request, root / 'diagnostic')
        controller.tick(now=100)
        return controller

    def test_analysed_video_decisions_and_real_timeline_remain_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.controller(directory)
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            c.observe(frame, 104, [{'reason': 'lead-in must not be saved'}])
            for index in range(10):
                c.observe(frame + index * 20, 105 + index, [{'reason': 'confirming', 'sample': index}])
            c.tick(now=115)
            self.assertEqual(c.state['status'], 'complete')
            meta = clip_metadata(c.path, 4, 10)
            self.assertEqual(meta['frame_times_seconds'], list(range(10)))
            rows = [json.loads(line) for line in c.path.with_suffix('.decisions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 10)
            self.assertEqual([r['frame'] for r in rows], list(range(10)))
            self.assertEqual([r['decisions'][0]['sample'] for r in rows], list(range(10)))
            self.assertEqual(c.end, None)

    def test_timeout_without_frames_is_explicitly_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.controller(directory)
            c.tick(now=115)
            self.assertEqual(c.state['status'], 'invalid')
            self.assertEqual(json.loads(c.path.with_suffix('.capture.json').read_text())['frames_decoded'], 0)
            self.assertFalse(c.path.exists())

    def test_interrupted_capture_is_not_a_valid_evaluation_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.controller(directory)
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            c.observe(frame, 105, [])
            c.close()
            self.assertEqual(c.state['status'], 'invalid')
            self.assertTrue(json.loads(c.path.with_suffix('.capture.json').read_text())['interrupted'])

    def test_old_request_cannot_start_an_unexpected_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / 'request.json'
            request.write_text(json.dumps({'requested_at': 100, 'lead_in': 5, 'seconds': 10}))
            c = AnalysisCapture(request, root / 'diagnostic')
            with self.assertRaisesRegex(ValueError, 'expired'):
                c.tick(now=111)
            self.assertEqual(c.state['status'], 'idle')

    def test_diagnostic_io_error_does_not_interrupt_live_analysis(self):
        worker = ObservedCameraWorker({'id': 'TEST'}, {'integrity_enabled': False, 'clip_seconds': 1},
                                      Mock(), [], threading.Event())
        worker.analysis_capture = Mock()
        worker.analysis_capture.observe.side_effect = OSError('test disk failure')
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch('rtsp_worker.CameraWorker._analyze', return_value=frame), patch('camera_health.logging.exception'):
            self.assertIs(worker._analyze(frame, None, 64), frame)
        worker.analysis_capture.close.assert_called_once()
