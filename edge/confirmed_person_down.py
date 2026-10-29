"""ThirdUni optional side-view confirmation on top of pretrained fall detection.

Reject upright/crouched shapes, require a persistent horizontal person box,
multiple observations, and stable location before raising a human-review event.
This conservative side-view profile can miss bodies aligned with the lens; it
must not be represented as a universal camera-independent fall detector.
"""
import logging
import math
import time
from falldet import FallModelDetector

log = logging.getLogger('aicos.confirmed-person-down')


class ConfirmedPersonDownDetector(FallModelDetector):
    name = 'thirduni-confirmed-person-down'

    def __init__(self, config=None, model=None):
        super().__init__(config, model)
        self.min_ratio = float(self.cfg.get('horizontal_ratio', 1.2))
        self.min_observations = int(self.cfg.get('min_observations', 4))
        self.gap_tolerance = float(self.cfg.get('gap_tolerance', .75))
        self.max_drift = float(self.cfg.get('max_center_drift', .35))
        self._anchor = None
        self._last_valid = None
        self._observations = 0
        self.last_decision = {}

    def _decision(self, reason, now, score=0, box=None, events=None, resets=None):
        self.last_decision = {'time': now, 'reason': reason, 'score': float(score),
                              'box': list(box) if box else None,
                              'held_seconds': max(0, now - self._since) if self._since is not None else 0,
                              'observations': self._observations, 'resets': resets or []}
        return events or []

    def _reset(self):
        self._since = None
        self._anchor = None
        self._last_valid = None
        self._observations = 0
        self._peak = 0

    def detect(self, frame, prev_frame=None, now=None):
        if self._disabled:
            self.last_decision = {'reason': 'model_disabled', 'time': now or time.time()}
            return []
        now = time.time() if now is None else now
        try:
            score, box = self._get_model().best_fall(frame)
        except Exception:
            self._disabled = True
            log.exception('Confirmed person-down model disabled; inspection required')
            return self._decision('model_error', now)
        resets = []
        if self._last_valid is not None and now - self._last_valid > self.gap_tolerance:
            resets.append('observation_gap')
            self._reset()
        if not box or score < float(self.cfg['conf']):
            return self._decision('no_fallen_detection', now, score, box, resets=resets)
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0 or width / height < self.min_ratio:
            self._reset()
            return self._decision('not_horizontal', now, score, box, resets=resets)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if self._anchor is not None:
            distance = math.hypot(center[0] - self._anchor[0], center[1] - self._anchor[1])
            if distance > self.max_drift * math.hypot(width, height):
                resets.append('position_change')
                self._reset()
        if self._since is None:
            self._since = now
            self._anchor = center
        self._last_valid = now
        self._observations += 1
        self._peak = max(self._peak, score)
        held = now - self._since
        if (held < float(self.cfg['seconds']) or self._observations < self.min_observations
                or now - self._fired_at < float(self.cfg['cooldown_seconds'])):
            reason = 'cooldown' if now - self._fired_at < float(self.cfg['cooldown_seconds']) else 'confirming'
            return self._decision(reason, now, score, box, resets=resets)
        self._fired_at = now
        h, w = frame.shape[:2]
        events = [{'agent': 'emergency', 'type': 'Person down / possible medical emergency',
                 'severity': int(self.cfg['severity']), 'confidence': round(min(.95, self._peak), 2),
                 'extra': {'detector': self.name, 'down_seconds': round(held, 2),
                           'lead_seconds': 0,
                           'px': [round(center[0] / w, 4), round(y2 / h, 4)],
                           'confirmation': {'profile': 'side-view', 'pretrained_label': 'fallen',
                                            'horizontal_box_ratio': round(width / height, 3),
                                            'observations': self._observations,
                                            'persistence_seconds': round(held, 2)},
                           'human_review_required': True}}]
        return self._decision('alert', now, score, box, events, resets)
