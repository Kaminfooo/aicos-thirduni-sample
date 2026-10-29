"""Optional site-aware crowd persistence and road-region filtering."""
from pathlib import Path
from detectors import YoloBehaviorSuite, point_in_polygon


def valid_region(points):
    return (isinstance(points, list) and len(points) >= 3 and
            all(len(p) == 2 and all(isinstance(v, (float, int)) and 0 <= v <= 1 for v in p) for p in points))


class SiteBehaviorSuite(YoloBehaviorSuite):
    name = 'thirduni-site-behaviors'

    def __init__(self, camera=None, config=None, engine=None):
        config = {'person_down': {'enabled': False}, 'fight': {'enabled': False},
                  'panic': {'enabled': False}, **(config or {})}
        super().__init__(camera=camera, config=config, engine=engine)
        self.region = config.get('crowd_region')
        self.road_region = config.get('road_region')
        if self.region is not None and not valid_region(self.region):
            raise ValueError('Invalid crowd region')
        if self.cfg['vehicles']['enabled'] and not valid_region(self.road_region):
            raise ValueError('Traffic requires an explicit normalized road region')
        wrong_way = self.cfg['vehicles'].get('wrong_way', {})
        if wrong_way.get('enabled') and wrong_way.get('direction') not in ('left', 'right', 'up', 'down'):
            raise ValueError('Wrong-way detection requires an explicit allowed traffic direction')
        self.crowd_seconds = float(config.get('crowd', {}).get('seconds', 5))
        self.crowd_gap = float(config.get('crowd', {}).get('gap_tolerance', 1))
        self._crowd_holds = {}

    def _get_engine(self):
        if self._engine is None and not Path(self.cfg['model']).is_file():
            raise RuntimeError('Site behaviors require local compatible weights; no automatic download')
        return super()._get_engine()

    def _crowd(self, live, now):
        selected = [track for track in live if self.region is None or
                    point_in_polygon(track.feet()[0] / self._fw, track.feet()[1] / self._fh, self.region)]
        count = len(selected)
        level = 'critical' if count >= self.cfg['crowd']['critical'] else 'warn' if count >= self.cfg['crowd']['warn'] else None
        for key, state in self._crowd_holds.items():
            if key != level or now - state['last'] > self.crowd_gap:
                state['since'] = None
        if level is None:
            return []
        state = self._crowd_holds.setdefault(level, {'since': None, 'last': now, 'fired_at': None})
        if state['since'] is None:
            state['since'] = now
        state['last'] = now
        held = now - state['since']
        if held < self.crowd_seconds or state['fired_at'] is not None and now - state['fired_at'] < 120:
            return []
        state['fired_at'] = now
        return [{'agent': 'security', 'type': 'Large crowd occupancy' if level == 'critical' else 'Crowd occupancy buildup',
                 'severity': 3 if level == 'critical' else 2, 'confidence': .7,
                 'extra': {'detector': self.name, 'count': count, 'persistence_seconds': round(held, 2),
                           'human_review_required': True, 'intent': 'not_established',
                           'region': 'configured' if self.region is not None else 'whole_view',
                           'validation_status': 'development_candidate'}}]

    def _vehicle(self, vehicles, now, frame_w, frame_h):
        selected = [d for d in vehicles if point_in_polygon(
            (d['box'][0] + d['box'][2]) / (2 * frame_w), d['box'][3] / frame_h, self.road_region)]
        events = super()._vehicle(selected, now, frame_w, frame_h)
        for event in events:
            event.setdefault('extra', {}).update(detector=self.name, human_review_required=True,
                                                 road_region='configured', validation_status='development_candidate')
        return events
