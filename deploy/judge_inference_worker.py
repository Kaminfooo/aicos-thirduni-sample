"""Private local inference worker. JSON protocol on stdout; diagnostics on stderr."""
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'edge'))
sys.path.insert(0, str(ROOT))


def main():
    # Imports may print; stdout is reserved for replies to our parent process.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    import cv2
    import numpy as np
    import torch
    from deploy.judge_detector_pipeline import Engines, Pipeline, HASHES
    torch.set_num_threads(2)
    model = ROOT / 'models/pretrained/yolo26n-pose.pt'
    expected = 'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9'
    if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Existing reference model is missing or changed; no download attempted')
    engines = Engines()
    engines.get('yolo26n-pose.pt')
    states = {}

    def reply(value):
        protocol.write(json.dumps(value, allow_nan=False) + '\n')
        protocol.flush()

    reply({'ready': True, 'model_sha256': expected})
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            sid = msg['session']
            if msg.get('op') == 'drop':
                states.pop(sid, None)
                reply({'dropped': True})
                continue
            if sid not in states:
                if len(states) >= 8:
                    raise ValueError('Inference session capacity reached')
                states[sid] = Pipeline(msg['settings'], engines)
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(msg['image'], validate=True), np.uint8), cv2.IMREAD_COLOR)
            if frame is None or max(frame.shape[:2]) > 640:
                raise ValueError('Invalid or oversized frame')
            detector = states[sid]
            reply(detector.detect(frame, msg['seconds']))
        except Exception as exc:
            reply({'error': str(exc)})


if __name__ == '__main__':
    main()
