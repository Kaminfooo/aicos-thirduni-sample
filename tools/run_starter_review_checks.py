"""Run one isolated review suite and preserve its output under a new directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('application', 'focused', 'starter', 'completion', 'judge'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error('Choose a new project-local output directory')
    output.mkdir(parents=True)
    temp = output / 'tmp'
    temp.mkdir()
    for key in list(os.environ):
        if key.startswith('AICOS_') or key in ('DB_PATH', 'SNAPSHOT_DIR', 'BACKEND_URL'):
            del os.environ[key]
    os.environ.update(TEMP=str(temp), TMP=str(temp), PYTHONDONTWRITEBYTECODE='1',
                      DB_PATH=str(temp / 'fallback.db'), SNAPSHOT_DIR=str(temp / 'snapshots'),
                      AICOS_OUTBOX_PATH=str(temp / 'outbox.db'), AICOS_NOTIFY_ON='off')
    tempfile.tempdir = str(temp)
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    paths = {'application': 'tests', 'focused': 'tools/eval_tests', 'starter': 'tools/starter_tests',
             'completion': 'tools/completion_tests', 'judge': 'tests'}
    class Result(unittest.TextTestResult):
        passed = 0
        def addSuccess(self, test):
            self.passed += 1
            super().addSuccess(test)
    if args.suite == 'application':
        # The judge wrapper deliberately owns process-global backend configuration
        # and refuses import into a private backend process. Importing its test
        # during common discovery changes the original tests' environment before
        # their fixtures are created. Keep that application in its own process.
        suite = unittest.TestSuite()
        for test_file in sorted((ROOT / 'tests').glob('test_*.py')):
            if test_file.name != 'test_judge_workspace.py':
                suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=test_file.name))
    else:
        pattern = 'test_judge_workspace.py' if args.suite == 'judge' else 'test_*.py'
        suite = unittest.defaultTestLoader.discover(str(ROOT / paths[args.suite]), pattern=pattern)
    with (output / 'unittest.txt').open('x', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=Result).run(suite)
    summary = {'tests_run': result.testsRun, 'passed': result.passed, 'failures': len(result.failures),
               'errors': len(result.errors), 'skipped': len(result.skipped)}
    successful = result.wasSuccessful()
    if args.suite == 'application':
        (output / 'original-application-summary.json').write_text(json.dumps(summary, indent=2))
        command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--suite', 'judge',
                   '--output', str(output / 'judge')]
        with (output / 'judge.stdout.txt').open('x', encoding='utf-8') as stdout, \
                (output / 'judge.stderr.txt').open('x', encoding='utf-8') as stderr:
            child = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr, timeout=120)
        (output / 'judge.process-exit.txt').write_text(str(child.returncode))
        judge_summary = json.loads((output / 'judge/summary.json').read_text())
        summary = {key: summary[key] + judge_summary[key] for key in summary}
        successful = successful and child.returncode == 0
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'exit-code.txt').write_text('0' if successful else '1')
    print(json.dumps(summary))
    return 0 if successful else 1


if __name__ == '__main__':
    sys.exit(main())
