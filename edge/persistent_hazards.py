"""Local class-verified fire/smoke and violence candidates with temporal holds."""
import logging
import math
from pathlib import Path
import threading
import time

log = logging.getLogger('aicos.hazards')


class LocalLabelEngine:
    def __init__(self, weights, labels, conf, imgsz):
        if not Path(weights).is_file():
            raise RuntimeError('Existing local model weights are required; no automatic download')
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.names = {int(k): str(v).strip().lower() for k, v in self.model.names.items()}
        self.labels = set(labels)
        if not self.labels <= set(self.names.values()):
            raise RuntimeError('Model label schema does not contain: ' + ', '.join(sorted(self.labels)))
        self.classes = [k for k, v in self.names.items() if v in self.labels]
        self.conf, self.imgsz = conf, imgsz
        self.lock = threading.Lock()

    def infer(self, frame):
        with self.lock:
            result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                        classes=self.classes, verbose=False)[0]
        return [{'label': self.names[int(b.cls.item())], 'conf': float(b.conf.item()),
                 'box': b.xyxy[0].cpu().tolist()} for b in result.boxes]


class PersistentHazardDetector:
    def __init__(self, config=None, engine=None):
        self.cfg = {'kind': 'fire_smoke', 'weights': '', 'conf': .5, 'imgsz': 640,
                    'seconds': 2, 'min_observations': 3, 'gap_tolerance': 1,
                    'cooldown_seconds': 60, **(config or {})}
        if self.cfg['kind'] not in ('fire_smoke', 'violence'):
            raise ValueError('Unsupported hazard kind')
        self.name = 'thirduni-' + self.cfg['kind']
        self.labels = ['fire', 'smoke'] if self.cfg['kind'] == 'fire_smoke' else ['violence']
        self._engine, self._disabled = engine, False
        self.state = {}
        self.last_decision = {}

    def detect(self, frame, prev_frame=None, now=None):
        now = time.time() if now is None else now
        if self._disabled:
            return []
        try:
            if self._engine is None:
                self._engine = LocalLabelEngine(self.cfg['weights'], self.labels, self.cfg['conf'], self.cfg['imgsz'])
            rows = self._engine.infer(frame)
        except Exception:
            self._disabled = True
            self.last_decision = {'time': now, 'reason': 'model_error'}
            log.exception('Hazard detector disabled; inspection required')
            return []
        events, holds = [], {}
        for label in self.labels:
            eligible = [d for d in rows if d['label'] == label and d['conf'] >= self.cfg['conf']
                        and len(d.get('box', [])) == 4 and d['box'][2] > d['box'][0] and d['box'][3] > d['box'][1]]
            old = self.state.get(label)
            if old and now - old['last'] > self.cfg['gap_tolerance']:
                old['since'], old['count'] = None, 0
            if not eligible:
                continue
            best = max(eligible, key=lambda d: d['conf'])
            x1, y1, x2, y2 = best['box']
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            if old is None:
                old = {'since': None, 'count': 0, 'fired_at': None, 'center': center, 'last': now}
                self.state[label] = old
            if math.dist(center, old['center']) > .75 * math.hypot(x2 - x1, y2 - y1):
                old['since'], old['count'] = None, 0
            if old['since'] is None:
                old['since'] = now
            old.update(last=now, center=center, count=old['count'] + 1)
            held = now - old['since']
            holds[label] = held
            if (held < self.cfg['seconds'] or old['count'] < self.cfg['min_observations'] or
                    old['fired_at'] is not None and now - old['fired_at'] < self.cfg['cooldown_seconds']):
                continue
            old['fired_at'] = now
            violence = label == 'violence'
            events.append({'agent': 'security' if violence else 'environmental',
                           'type': 'Possible physical violence detected' if violence else 'Possible ' + label + ' detected',
                           'severity': 3 if violence or label == 'smoke' else 4,
                           'confidence': round(min(.95, best['conf']), 2),
                           'extra': {'detector': self.name, 'human_review_required': True,
                                     'label': label, 'persistence_seconds': round(held, 2),
                                     'observations': old['count'], 'lead_seconds': 0,
                                     'px': [round(center[0] / frame.shape[1], 4), round(y2 / frame.shape[0], 4)],
                                     'validation_status': 'development_candidate'}})
        self.last_decision = {'time': now, 'reason': 'alert' if events else 'confirming' if holds else 'no_supported_hazard',
                              'holds': holds}
        return events
