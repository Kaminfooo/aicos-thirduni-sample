"""Process inspection failure must never authorize an unrelated process."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('judge_manager_lookup_test', ROOT / 'tools/manage_judge_workspace.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class ProcessLookupTests(unittest.TestCase):
    def test_wmi_denied_falls_back_to_exact_pid(self):
        value = {'ProcessId': 123, 'ExecutablePath': str(ROOT / '.venv/Scripts/python.exe'),
                 'CommandLine': f'{ROOT} {manager.MODULE} --port 8084'}
        with patch.object(manager.subprocess, 'run', side_effect=[
            subprocess.CompletedProcess([], 1, '', 'Access denied'),
            subprocess.CompletedProcess([], 0, json.dumps(value), '')]) as run:
            self.assertEqual(manager.identity(123), value)
            self.assertEqual(run.call_args.args[0][-1], '123')
            self.assertTrue(manager.owns(value))

    def test_missing_process_is_not_owned(self):
        with patch.object(manager.subprocess, 'run', side_effect=[
            subprocess.CompletedProcess([], 1, '', 'Access denied'),
            subprocess.CompletedProcess([], 0, 'null', '')]):
            self.assertIsNone(manager.identity(123))
            self.assertFalse(manager.owns(None))

    def test_both_inspection_methods_fail_closed(self):
        with patch.object(manager.subprocess, 'run', return_value=
                          subprocess.CompletedProcess([], 1, '', 'Access denied')):
            with self.assertRaisesRegex(RuntimeError, 'no process started or stopped'):
                manager.identity(123)

    def test_different_executable_is_not_owned(self):
        self.assertFalse(manager.owns({'ExecutablePath': 'C:/unrelated/python.exe',
            'CommandLine': f'{ROOT} {manager.MODULE} --port 8084'}))
