"""Describe observed posture transitions; never infer consciousness or cause."""
import math

DEFAULTS = {'history_seconds': 12, 'max_observation_gap': .75,
            'max_track_gap': 3, 'rapid_seconds': 1.2, 'gradual_seconds': 2.5,
            'min_upright_samples': 3, 'min_upright_seconds': .5,
            'min_hip_drop_fraction': .25, 'joint_conf': .3}


class TemporalPostureObserver:
    def __init__(self, config=None):
        self.cfg = {**DEFAULTS, **(config or {})}
        self.tracks = []
        self.next_id = 1

    def _geometry(self, detection, lying):
        x1, y1, x2, y2 = detection['box']
        height = max(1, y2 - y1)
        points = detection.get('kpts')
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if not points or len(points) < 17:
            return center, height, 'unknown'
        if any(points[j][2] < self.cfg['joint_conf'] for j in (5, 6, 11, 12, 15, 16)):
            return center, height, 'unknown'
        shoulder, hip, ankle = [[(points[a][i] + points[b][i]) / 2 for i in (0, 1)]
                                for a, b in ((5, 6), (11, 12), (15, 16))]
        if lying:
            return tuple(hip), height, 'lying'
        upright = ((x2 - x1) / height < .9 and
                   hip[1] - shoulder[1] > abs(hip[0] - shoulder[0]) and
                   ankle[1] - hip[1] >= .3 * height)
        return tuple(hip), height, 'upright' if upright else 'transition'

    def _classify(self, track, now, position):
        reference = track.get('upright_reference')
        if reference is None:
            return {'kind': 'already_down_transition_not_observed', 'reason': 'no_stable_upright_reference'}
        elapsed = now - reference['time']
        drop = (position[1] - reference['hip_y']) / reference['height']
        evidence = {'duration_seconds': round(elapsed, 3), 'hip_drop_body_fraction': round(drop, 3)}
        if elapsed > self.cfg['history_seconds']:
            return {**evidence, 'kind': 'transition_uncertain', 'reason': 'upright_reference_too_old'}
        if track['uncertain']:
            return {**evidence, 'kind': 'transition_uncertain', 'reason': 'missing_frames_joints_or_ambiguous_track'}
        if drop < self.cfg['min_hip_drop_fraction']:
            return {**evidence, 'kind': 'transition_uncertain', 'reason': 'insufficient_observed_downward_displacement'}
        if elapsed <= self.cfg['rapid_seconds']:
            kind = 'abrupt_collapse_observed'
        elif elapsed >= self.cfg['gradual_seconds']:
            kind = 'gradual_lowering_observed'
        else:
            kind = 'transition_observed_speed_uncertain'
        return {**evidence, 'kind': kind, 'reason': 'observed_pose_sequence'}

    def update(self, detections, now, lying_predicate):
        self.tracks = [t for t in self.tracks if 0 <= now - t['last'] <= self.cfg['max_track_gap']]
        contexts, used = {}, set()
        for detection in sorted(detections, key=lambda d: d.get('conf', 0), reverse=True):
            position, height, posture = self._geometry(detection, lying_predicate(detection))
            choices = sorted((math.dist(position, t['position']), index) for index, t in enumerate(self.tracks)
                             if index not in used and math.dist(position, t['position']) <= .75 * max(height, t['height']))
            ambiguous = len(choices) > 1 and choices[1][0] - choices[0][0] < .1 * height
            if choices:
                _, index = choices[0]
                track = self.tracks[index]
                if now - track['last'] > self.cfg['max_observation_gap']:
                    track['uncertain'] = True
            else:
                track = {'id': self.next_id, 'position': position, 'height': height, 'last': now,
                         'up_count': 0, 'up_start': now, 'upright_reference': None,
                         'transition': None, 'seen_down': False, 'recovery': False, 'uncertain': False}
                self.next_id += 1
                self.tracks.append(track)
                index = len(self.tracks) - 1
            used.add(index)
            track['uncertain'] |= ambiguous or posture == 'unknown'
            if posture == 'upright':
                if not track['up_count']:
                    track['up_start'] = now
                track['up_count'] += 1
                if (track['up_count'] >= self.cfg['min_upright_samples'] and
                        now - track['up_start'] >= self.cfg['min_upright_seconds']):
                    track['recovery'] |= track['seen_down']
                    track['seen_down'] = False
                    track['upright_reference'] = {'time': now, 'hip_y': position[1], 'height': height}
                    track['uncertain'] = ambiguous
                    track['transition'] = None
            else:
                track['up_count'] = 0
                if posture == 'lying' and not track['seen_down']:
                    track['transition'] = self._classify(track, now, position)
                    track['seen_down'] = True
                    track['recovery'] = False
            track.update(position=position, height=height, last=now)
            contexts[id(detection)] = {
                'track_id': track['id'], 'posture': posture,
                'transition': dict(track['transition'] or {'kind': 'transition_not_yet_observed'}),
                'recovery_observed': track['recovery'], 'medical_cause': 'unknown',
                'consciousness': 'not_established', 'intent': 'not_established'}
        return contexts
