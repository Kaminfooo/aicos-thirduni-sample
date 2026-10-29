"""Session-isolated adapters around existing detectors. No downloads or dispatch."""
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'edge'))
from deploy.judge_capabilities import IDS

HASHES = {
 'yolo26n-pose.pt': 'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9',
 'yolo26n.pt': '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef',
 'fire-smoke-yolo26n.pt': 'd2b0be214f45fa01d2044b960c3cb4da4bf65c06a89cf0a1967b0d40e8db4b98',
 'violence-yolov8n.pt': '1714eef088cd4ae5fc8618f6f94890d476434298be6c7a2151faef18bef9d459'}
POSE = dict(conf=.35, imgsz=640, seconds=4, gap_tolerance=2, cooldown_seconds=90,
 min_observations=4, horizontal_ratio=1.2, joint_conf=.3, min_body_area=.015,
 max_center_drift=.35, orientation_mode='aligned_segments')


def weights(name):
    path = ROOT / 'models/pretrained' / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise RuntimeError('Missing or changed local weights: ' + name)
    return str(path)


class Engines:
    """Share weights, not tracking. Cache predictions only within one input frame."""
    def __init__(self):
        self.models = {}
        self.predictions = {}

    def begin(self):
        self.predictions.clear()

    def get(self, name):
        if name not in self.models:
            path = weights(name)
            if name in ('yolo26n.pt', 'yolo26n-pose.pt'):
                from detectors import UltralyticsEngine
                self.models[name] = UltralyticsEngine(path, .35 if 'pose' in name else .4,
                    640, classes=(0,) if 'pose' in name else (0,2,3,5,7,24,26,28))
            else:
                from persistent_hazards import LocalLabelEngine
                self.models[name] = LocalLabelEngine(path,
                    ['fire','smoke'] if 'fire' in name else ['violence'], .5, 640)
        owner = self
        class FrameEngine:
            def infer(self, frame):
                if name not in owner.predictions:
                    owner.predictions[name] = owner.models[name].infer(frame)
                return owner.predictions[name]
        return FrameEngine()


def polygon(rect):
    x1,y1,x2,y2 = rect
    return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


class Pipeline:
    def __init__(self, settings, engines):
        self.settings, self.engines = settings, engines
        self.selected = settings['detectors']
        if not self.selected or any(k not in IDS for k in self.selected):
            raise ValueError('Unknown/empty detector selection')
        self.detectors = {}
        self.errors = {}
        self.previous = None
        self.motion_at = None

    def build(self, key):
        from detectors import YoloBehaviorSuite
        from site_behaviors import SiteBehaviorSuite
        from pose_confirmed_person_down import PoseConfirmedPersonDownDetector
        from persistent_hazards import PersistentHazardDetector
        s = self.settings
        if key == 'person_down':
            return PoseConfirmedPersonDownDetector(dict(POSE, model=weights('yolo26n-pose.pt')),
                engine=self.engines.get('yolo26n-pose.pt'))
        if key in ('fire_smoke','violence'):
            name = 'fire-smoke-yolo26n.pt' if key == 'fire_smoke' else 'violence-yolov8n.pt'
            return PersistentHazardDetector(dict(kind=key, weights=weights(name)), engine=self.engines.get(name))
        if key == 'motion':
            from rtsp_worker import MotionDetector
            return MotionDetector()
        flags = {k: {'enabled': k == key} for k in ('person_down','fight','crowd','panic',
            'intrusion','loitering','vehicles','abandoned_object','snatch')}
        region = polygon(s['region']) if s.get('region') else None
        flags.update(model=weights('yolo26n-pose.pt' if key in ('fight','panic','snatch') else 'yolo26n.pt'),
            conf=.35 if key in ('fight','panic','snatch') else .4, imgsz=640)
        flags['crowd'].update(warn=s['crowd_warn'], critical=max(s['crowd_warn']+1,15),seconds=5)
        flags['loitering']['seconds'] = s['loiter_seconds']
        flags['vehicles'].update(stopped_seconds=s['stopped_seconds'],
            wrong_way={'enabled':bool(s['direction']), 'direction':s['direction']})
        flags.update(crowd_region=region,road_region=region)
        zones = [{'name':'Workspace test region','type':'intrusion' if key=='intrusion' else 'loiter','points':region}] if key in ('intrusion','loitering') else []
        engine = self.engines.get('yolo26n-pose.pt' if key in ('fight','panic','snatch') else 'yolo26n.pt')
        return SiteBehaviorSuite(camera={'zones':zones},config=flags,engine=engine)

    def detect(self, frame, seconds):
        self.engines.begin()
        events, observations = [], []
        for key in self.selected:
            start = time.perf_counter()
            try:
                if key in self.errors:
                    raise RuntimeError(self.errors[key])
                if key not in self.detectors:
                    self.detectors[key] = self.build(key)
                d = self.detectors[key]
                found = d.detect(frame,self.previous) if key=='motion' else d.detect(frame,now=1_000_000+seconds)
                if getattr(d,'_disabled',False):
                    raise RuntimeError('Detector disabled after inference failure')
                if key=='motion' and self.motion_at is not None and seconds-self.motion_at<30:
                    found=[]
                if key=='motion' and found:
                    self.motion_at=seconds
                for event in found:
                    model_name = ('yolo26n-pose.pt' if key in ('person_down','fight','panic','snatch') else
                        'fire-smoke-yolo26n.pt' if key=='fire_smoke' else 'violence-yolov8n.pt' if key=='violence' else 'yolo26n.pt')
                    event.setdefault('extra',{}).update(judge_capability=key,human_review_required=True,
                        model_sha256=HASHES[model_name] if key!='motion' else None,
                        validation_status='experimental' if key in ('violence','fight','panic','snatch','abandoned_object') else 'development_candidate')
                events.extend(found)
                detail = dict(getattr(d,'last_decision',{}) or {})
                detail.setdefault('reason','alert' if found else 'monitoring')
                detail.setdefault('persons',len(getattr(getattr(d,'tracker',None),'tracks',[])))
                observations.append(dict(id=key,status='alert' if found else 'monitoring',
                    detail=detail,elapsed_ms=round((time.perf_counter()-start)*1000)))
            except Exception as exc:
                self.errors[key] = str(exc)
                observations.append(dict(id=key,status='error',detail={'reason':'model_error','message':str(exc)},
                    elapsed_ms=round((time.perf_counter()-start)*1000)))
        self.previous = frame.copy()
        primary = next((r['detail'] for r in observations if r['id']=='person_down'),
            {'reason':'alert' if events else 'monitoring','persons':0})
        return dict(events=events,observation=primary,detectors=observations,
            model_sha256=HASHES['yolo26n-pose.pt'],model_hashes={k:HASHES[k] for k in self.engines.models})
