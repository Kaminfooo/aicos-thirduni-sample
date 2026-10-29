"""Optional diagnostics cannot break detection or write after their time limit."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'edge'))
from camera_health import ObservedDetector


class TraceTests(unittest.TestCase):
    class Detector:
        name = 'test'
        last_decision = {'reason': 'confirming'}

        def detect(self, frame, previous):
            return [{'type': 'test event'}]

    def test_trace_disk_error_preserves_detection(self):
        wrapper = ObservedDetector(self.Detector(), trace_path='unused.jsonl', trace_seconds=300)
        with patch('builtins.open', side_effect=PermissionError('busy')):
            self.assertEqual(wrapper.detect(None, None), [{'type': 'test event'}])
        self.assertEqual(wrapper.status, 'running')

    def test_expired_trace_does_not_write(self):
        wrapper = ObservedDetector(self.Detector(), trace_path='unused.jsonl', trace_seconds=0)
        with patch('builtins.open') as output:
            self.assertEqual(wrapper.detect(None, None), [{'type': 'test event'}])
            output.assert_not_called()
