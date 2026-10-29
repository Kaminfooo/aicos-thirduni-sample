"""Bounded, explicitly requested diagnostic recording of analysed frames only."""
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
import cv2


class AnalysisCapture:
    def __init__(self, request_path, output_dir):
        self.request_path = Path(request_path)
        self.output_dir = Path(output_dir)
        self.lock = threading.RLock()
        self.writer = None
        self.trace = None
        self.times = []
        self.state = {'status': 'idle'}
        self.end = None

    def tick(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            if self.end is not None and now >= self.end:
                self._finish()
            if self.request_path.exists():
                request = json.loads(self.request_path.read_text())
                self.request_path.unlink()
                if self.end is not None:
                    raise ValueError('A diagnostic capture is already active')
                seconds, lead = float(request['seconds']), float(request['lead_in'])
                requested = float(request['requested_at'])
                if not (10 <= seconds <= 60 and 5 <= lead <= 30 and 0 <= now - requested <= 10):
                    raise ValueError('Invalid or expired diagnostic request')
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.path = self.output_dir / ('live_' + str(time.time_ns()) + '_' + uuid.uuid4().hex[:8] + '.mp4')
                self.start = now + lead
                self.end = self.start + seconds
                self.times = []
                self.requested_seconds = seconds
                self.state = {'status': 'lead-in', 'starts_at': self.start, 'ends_at': self.end,
                              'video': str(self.path), 'frames': 0}

    def observe(self, frame, receipt_time, decisions):
        with self.lock:
            if self.end is None or not self.start <= receipt_time < self.end:
                return
            offset = receipt_time - self.start
            if self.times and offset <= self.times[-1]:
                return
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), 4, (w, h))
                if not self.writer.isOpened():
                    self.writer.release()
                    self.writer = None
                    raise OSError('Diagnostic video writer could not open')
                self.trace = self.path.with_suffix('.decisions.jsonl').open('x', encoding='utf-8')
            self.writer.write(frame)
            self.trace.write(json.dumps({'frame': len(self.times), 'capture_seconds': offset,
                                         'decisions': decisions}) + '\n')
            self.trace.flush()
            self.times.append(offset)
            self.state.update(status='recording', frames=len(self.times))

    def _finish(self, interrupted=False):
        if self.end is None:
            return
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.trace is not None:
            self.trace.close()
            self.trace = None
        decoded = 0
        if self.path.exists():
            probe = cv2.VideoCapture(str(self.path))
            try:
                while probe.read()[0]:
                    decoded += 1
            finally:
                probe.release()
        gaps = [b - a for a, b in zip(self.times, self.times[1:])]
        span = self.times[-1] - self.times[0] if len(self.times) >= 2 else 0
        valid = not interrupted and decoded == len(self.times) and decoded >= 2 and span >= .8 * self.requested_seconds
        metadata = {'schema_version': 1, 'status': 'valid' if valid else 'invalid',
                    'label': 'unreviewed', 'scenario': 'live_diagnostic',
                    'video_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.exists() else None,
                    'encoded_fps': 4, 'frame_times_seconds': self.times,
                    'frames_written': len(self.times), 'frames_decoded': decoded,
                    'capture_wall_seconds': self.requested_seconds, 'observed_span_seconds': span,
                    'max_gap_seconds': max(gaps, default=0), 'interrupted': interrupted,
                    'timestamp_definition': 'PC analysis-input receipt time; not camera exposure time',
                    'note': 'Same frames as live inference, saved after inference; not a labeled accuracy test.'}
        self.path.with_suffix('.capture.json').write_text(json.dumps(metadata, indent=2))
        self.state.update(status='complete' if valid else 'invalid', frames=len(self.times),
                          max_gap_seconds=metadata['max_gap_seconds'])
        self.end = None

    def close(self):
        with self.lock:
            self._finish(interrupted=True)
