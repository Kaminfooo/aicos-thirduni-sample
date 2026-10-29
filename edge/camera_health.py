"""Observed camera/detector adapters; preserve existing detection and delivery behavior."""
import time
import json
import logging
from rtsp_worker import BackendClient, CameraWorker


class ObservedDetector:
    def __init__(self, detector, trace_path=None, trace_seconds=0):
        self.detector = detector
        self.name = detector.name
        self.status = 'initializing'
        self.trace_path = trace_path
        self.trace_deadline = time.monotonic() + min(300, max(0, trace_seconds))
        self.trace_rows = 0

    def detect(self, frame, previous):
        try:
            result = self.detector.detect(frame, previous)
        except Exception:
            self.status = 'failed'
            raise
        self.status = 'failed' if getattr(self.detector, '_disabled', False) else 'running'
        if self.trace_path and time.monotonic() < self.trace_deadline and self.trace_rows < 1200:
            try:
                with open(self.trace_path, 'a', encoding='utf-8') as stream:
                    stream.write(json.dumps(getattr(self.detector, 'last_decision', {})) + '\n')
                self.trace_rows += 1
            except OSError:
                pass  # Diagnostics must never interrupt incident detection/delivery.
        return result


class ObservedBackendClient(BackendClient):
    health_provider = None

    def post_heartbeat(self, payload):
        if self.health_provider:
            payload = {**payload, 'extra': {**payload.get('extra', {}), **self.health_provider()}}
        return super().post_heartbeat(payload)


class ObservedCameraWorker(CameraWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_frame_at = None
        self.processing_ms = None
        self.analysis_capture = None

    def _analyze(self, frame, previous, resize_w):
        self.last_frame_at = time.time()
        start = time.perf_counter()
        try:
            result = super()._analyze(frame, previous, resize_w)
            if self.analysis_capture:
                try:
                    decisions = [{'detector': d.name, **getattr(d.detector, 'last_decision', {})}
                                 for d in self.detectors if isinstance(d, ObservedDetector)]
                    self.analysis_capture.observe(result, self.last_frame_at, decisions)
                except Exception:
                    logging.exception('Diagnostic capture failed; live detection continues')
                    self.analysis_capture.close()
            return result
        finally:
            self.processing_ms = round((time.perf_counter() - start) * 1000, 2)

    def health(self):
        statuses = [{'name': detector.name, 'status': getattr(detector, 'status', 'unknown')}
                    for detector in self.detectors]
        extended = any(d['name'].startswith(('thirduni-site', 'thirduni-fire', 'thirduni-violence')) for d in statuses)
        return {'mode': 'multi-capability' if extended else 'person-down' if statuses else 'camera-only',
                'detectors': statuses,
                'detector_status': 'failed' if any(d['status'] == 'failed' for d in statuses)
                                   else 'initializing' if any(d['status'] == 'initializing' for d in statuses)
                                   else 'running' if statuses else 'off',
                'frame_age_seconds_at_heartbeat': max(0, time.time() - self.last_frame_at)
                                                if self.last_frame_at is not None else None,
                'last_processing_ms': self.processing_ms,
                'capture_fps_source': 'stream metadata hint'}
