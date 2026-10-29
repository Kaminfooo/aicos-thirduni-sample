"""Optional side-view lying-posture confirmation using a pretrained pose model.

Uses confident shoulders/hips/ankles rather than a fallen-class crop. This is
geometric decision support, not a trained medical diagnosis or fall classifier.
"""
import logging
import math
from pathlib import Path
import time
from detectors import _shared_engine
from temporal_posture import TemporalPostureObserver

log = logging.getLogger('aicos.pose-confirmed-person-down')


class PoseConfirmedPersonDownDetector:
    name = 'thirduni-pose-confirmed-person-down'

    def __init__(self, config=None, engine=None):
        self.cfg = {'model': '', 'conf': .35, 'imgsz': 640, 'seconds': 4,
                    'gap_tolerance': 2, 'min_observations': 4, 'horizontal_ratio': 1.2,
                    'axis_ratio': 1.15, 'joint_conf': .3, 'min_body_area': .015,
                    'max_center_drift': .35, 'cooldown_seconds': 90,
                    'orientation_mode': 'axes', 'alignment_min': .7, **(config or {})}
        if self.cfg['orientation_mode'] not in ('axes', 'aligned_segments'):
            raise ValueError('Unsupported pose orientation mode')
        self._engine = engine
        self._disabled = False
        self._sites = []
        self.last_decision = {}
        self.temporal = TemporalPostureObserver(self.cfg.get('temporal'))
        self.cfg['temporal'] = dict(self.temporal.cfg)

    def _get_engine(self):
        if self._engine is None:
            path = Path(self.cfg['model'])
            if not path.is_file():
                raise RuntimeError('Pose confirmation requires existing local weights; no automatic download')
            self._engine = _shared_engine(str(path), self.cfg['conf'], self.cfg['imgsz'], [0])
        return self._engine

    def _qualifies(self, detection, frame_area):
        box, points = detection['box'], detection.get('kpts')
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        if (detection.get('conf', 0) < self.cfg['conf'] or width <= 0 or height <= 0
                or width * height < self.cfg['min_body_area'] * frame_area):
            return False
        if width / height < self.cfg['horizontal_ratio'] or not points or len(points) < 17:
            return False
        if any(points[j][2] < self.cfg['joint_conf'] for j in (5, 6, 11, 12, 15, 16)):
            return False
        shoulder, hip, ankle = [[(points[a][i] + points[b][i]) / 2 for i in (0, 1)]
                                for a, b in ((5, 6), (11, 12), (15, 16))]
        if self.cfg['orientation_mode'] == 'aligned_segments':
            torso = [hip[i] - shoulder[i] for i in (0, 1)]
            legs = [ankle[i] - hip[i] for i in (0, 1)]
            lengths = math.hypot(*torso), math.hypot(*legs)
            if min(lengths) < 2:
                return False
            alignment = sum(a * b for a, b in zip(torso, legs)) / (lengths[0] * lengths[1])
            return alignment >= self.cfg['alignment_min']
        for a, b in ((shoulder, hip), (hip, ankle)):
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            if dx < 1 or dx < self.cfg['axis_ratio'] * dy:
                return False
        return True

    def detect(self, frame, prev_frame=None, now=None):
        now = time.time() if now is None else now
        if self._disabled:
            return []
        try:
            detections = self._get_engine().infer(frame)
        except Exception:
            self._disabled = True
            self.last_decision = {'time': now, 'reason': 'model_error'}
            log.exception('Pose confirmation disabled; inspection required')
            return []
        h, w = frame.shape[:2]
        contexts = self.temporal.update(detections, now, lambda d: self._qualifies(d, h * w))
        candidates = [d for d in detections if d.get('cls', 0) == 0 and self._qualifies(d, h * w)]
        self._sites = [s for s in self._sites if now - s['last'] <= max(120, self.cfg['cooldown_seconds'])]
        used, events, holds = set(), [], []
        for detection in sorted(candidates, key=lambda d: d['conf'], reverse=True):
            x1, y1, x2, y2 = detection['box']
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            radius = self.cfg['max_center_drift'] * math.hypot(x2 - x1, y2 - y1)
            choices = [(math.dist(center, s['center']), index) for index, s in enumerate(self._sites)
                       if index not in used and math.dist(center, s['center']) <= radius]
            if choices:
                _, index = min(choices)
                site = self._sites[index]
                if now - site['last'] > self.cfg['gap_tolerance']:
                    site.update(since=now, observations=0, peak=0)
            else:
                site = {'center': center, 'since': now, 'last': now, 'observations': 0,
                        'peak': 0, 'fired_at': None}
                self._sites.append(site)
                index = len(self._sites) - 1
            used.add(index)
            site['last'] = now
            site['observations'] += 1
            site['peak'] = max(site['peak'], detection['conf'])
            held = now - site['since']
            holds.append(held)
            if (held < self.cfg['seconds'] or site['observations'] < self.cfg['min_observations']
                    or site['fired_at'] is not None and now - site['fired_at'] < self.cfg['cooldown_seconds']):
                continue
            site['fired_at'] = now
            context = contexts.get(id(detection), {})
            abrupt = context.get('transition', {}).get('kind') == 'abrupt_collapse_observed'
            event_type = 'Suspected sudden collapse / person down' if abrupt else 'Person down / possible medical emergency'
            events.append({'agent': 'emergency', 'type': event_type,
                           'severity': 4, 'confidence': round(min(.95, site['peak']), 2),
                           'extra': {'detector': self.name, 'human_review_required': True,
                                     'down_seconds': round(held, 2), 'lead_seconds': 0,
                                     'px': [round(center[0] / w, 4), round(y2 / h, 4)],
                                     'transition_context': context,
                                     'confirmation': {'profile': 'side-view-pose',
                                         'method': ('confident shoulders/hips/ankles; wide body and aligned torso/legs'
                                                    if self.cfg['orientation_mode'] == 'aligned_segments' else
                                                    'confident shoulders/hips/ankles; horizontal torso and legs'),
                                         'observations': site['observations'], 'persistence_seconds': round(held, 2)}}})
        self.last_decision = {'time': now, 'reason': 'alert' if events else 'confirming' if candidates else 'no_supported_lying_pose',
                              'persons': len(detections), 'candidates': len(candidates),
                              'held_seconds': max(holds, default=0),
                              'score': max((d['conf'] for d in candidates), default=0)}
        self.last_decision['temporal_observations'] = list(contexts.values())
        return events
