"""Repeat the frozen ThirdUni development replay without downloads or overwrites."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New project-local report directory')
    parser.add_argument('--profile', choices=('pose', 'legacy'), default='pose')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error('Reports must remain inside this project')
    if output.exists():
        parser.error('Choose a new directory to preserve previous reports')
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(TEMP=str(ROOT / '.runtime/tmp'), TMP=str(ROOT / '.runtime/tmp'),
               YOLO_CONFIG_DIR=str(ROOT / '.runtime/inference-setup/ultralytics'),
               MPLCONFIGDIR=str(ROOT / '.runtime/inference-setup/matplotlib'),
               YOLO_AUTOINSTALL='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    command = [str(ROOT / '.venv-inference/Scripts/python.exe'), '-B',
               str(ROOT / 'tools/evaluate_person_down.py'), str(ROOT / 'data/person_down_eval'),
               '--config', str(ROOT / 'config/cameras-phone.yaml'), '--camera-id', 'CAM-LIVE',
               '--json-out', str(output / 'evaluation.json')]
    if args.profile == 'pose':
        command += ['--mode', 'pose-confirmed', '--model', str(ROOT / 'models/pretrained/yolo26n-pose.pt'),
                    '--imgsz', '640', '--conf', '.35', '--seconds', '4']
    else:
        command += ['--mode', 'confirmed', '--fall-weights',
                    str(ROOT / 'models/pretrained/human-fall-yolo11.pt'), '--fall-seconds', '4']
    with (output / 'stdout.txt').open('x') as stdout, (output / 'stderr.txt').open('x') as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=300)
    (output / 'exit-code.txt').write_text(str(result.returncode))
    print('Replay report:', output)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
