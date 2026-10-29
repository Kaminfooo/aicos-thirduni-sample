"""Preserved real-frame inference and annotation-based site-rule screening.

Research screening only. Publisher object labels are not crowd danger, criminal
intent, wrong-way road rules, or a claim of public-camera deployment accuracy.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/capability_validation_20261027/site'
WORK = ROOT / '.runtime/capability-validation-20261027/site-validation'
FULL = [[0, 0], [1, 0], [1, 1], [0, 1]]
EMPTY = [[0, 0], [.02, 0], [.02, .02], [0, .02]]
SELECTION = [('MOT17-04.mp4', 1, 1050), ('MOT17-09.mp4', 1, 525),
             ('Walk1.mpg', 1, None), ('LeftBag.mpg', 1, None),
             ('Meet_Crowd.mpg', 1, None), ('Fight_RunAway1.mpg', 1, None),
             ('sherbrooke.avi', 2754, 3754), ('rouen.avi', 20, 620)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def annotations(name):
    """Publisher IDs/boxes, without treating detections as ground truth."""
    frames = defaultdict(list)
    if name.startswith('MOT17'):
        sequence = Path(name).stem + '-SDP'
        with zipfile.ZipFile(DATA / 'mot17-labels.zip') as archive:
            text = archive.read('train/' + sequence + '/gt/gt.txt').decode()
        for line in text.splitlines():
            fields = list(map(float, line.split(',')))
            number, ident, x, y, w, h, valid, cls, visibility = fields
            if valid == 1 and cls == 1:
                frames[int(number)].append({'id': int(ident), 'box': [x, y, x+w, y+h], 'cls': 0})
    elif name.endswith('.mpg'):
        xmlname = {'Walk1.mpg': 'wk1gt.xml', 'LeftBag.mpg': 'lb1gt.xml',
                   'Meet_Crowd.mpg': 'mc1gt.xml', 'Fight_RunAway1.mpg': 'fra1gt.xml'}[name]
        for frame in ET.parse(DATA / xmlname).getroot().findall('frame'):
            for obj in frame.findall('./objectlist/object'):
                b = obj.find('box').attrib
                x, y, w, h = (float(b[key]) for key in ('xc', 'yc', 'w', 'h'))
                roles = [s.text for s in obj.findall('.//role')]
                # Annotated non-person objects cannot be counted as pedestrians.
                if any(s and ('object' in s.lower() or 'bag' in s.lower()) for s in roles):
                    continue
                frames[int(frame.attrib['number']) + 1].append({'id': int(obj.attrib['id']), 'box': [x-w/2,y-h/2,x+w/2,y+h/2], 'cls': 0})
    else:
        stem = Path(name).stem
        with zipfile.ZipFile(DATA / (stem + '-annotations.zip')) as archive:
            member = next(n for n in archive.namelist() if n.endswith('_gt_cars.sqlite'))
            target = WORK / (stem + '-publisher-cars.sqlite')
            payload = archive.read(member)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError('Changed extracted annotation DB')
            if not target.exists():
                target.write_bytes(payload)
        with sqlite3.connect('file:' + target.as_posix() + '?mode=ro', uri=True) as database:
            for ident, number, x1,y1,x2,y2 in database.execute('select * from bounding_boxes'):
                frames[number].append({'id': ident, 'box':[x1,y1,x2,y2], 'cls':2})
    return frames


def prepare(output):
    output.mkdir(parents=True)
    plan = {'selection_before_inference': SELECTION, 'sample_fps': 3, 'resize_width':480,
            'model':'models/pretrained/yolo26n.pt', 'conf': .4, 'imgsz':640,
            'annotation_scope':'Publisher pedestrian/car boxes. CAVIAR stationary people may be unlabelled. No publisher crowd-danger, site authorization, traffic direction or theft labels.',
            'policies': {'crowd':{'warn':8,'critical':15,'seconds':5}, 'intrusion':'all-view presence and empty top-left region controls',
                        'loitering':'explicit 5-second whole-view dwell demonstration; production default remains 120 seconds',
                        'vehicles':'default45-second stopping and six stationary vehicles; road region whole-view demo only',
                        'unattended':'default60-second bag hold; short LeftBag reference may not satisfy duration',
                        'fight_panic_snatch':'frozen existing heuristic output; crowd/runaway does not establish panic or theft'},
            'source_sha256':{p:sha(ROOT/p) for p in ('edge/detectors.py','edge/site_behaviors.py')},
            'model_sha256':sha(ROOT/'models/pretrained/yolo26n.pt')}
    (output/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return plan


def collect(output):
    if output.exists():
        raise ValueError('Choose a new output; prior incomplete and failed results stay preserved')
    plan = prepare(output)
    temp=output/'tmp'; temp.mkdir()
    os.environ.update(TEMP=str(temp),TMP=str(temp),YOLO_AUTOINSTALL='false',
        YOLO_CONFIG_DIR=str(output/'ultralytics'),MPLCONFIGDIR=str(output/'matplotlib'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    sys.path.insert(0,str(ROOT/'edge'))
    import cv2
    import torch
    from detectors import UltralyticsEngine
    torch.set_num_threads(2)
    engine=UltralyticsEngine(str(ROOT/plan['model']),plan['conf'],plan['imgsz'],classes=(0,2,3,5,7,24,26,28))
    receipts={}
    for file in ('acquisition.json','traffic-acquisition.json'):
        if (WORK/file).exists(): receipts.update(json.loads((WORK/file).read_text()))
    report={'plan':plan,'cases':[], 'field_accuracy_established':False}
    for name,start,end in SELECTION:
        row={'file':name,'samples':[], 'status':'incomplete'}
        cap=None; begun=time.monotonic()
        try:
            if name not in receipts or receipts[name].get('status')!='acquired' or sha(DATA/name)!=receipts[name]['sha256']:
                raise ValueError('Input missing or hash not receipted')
            gt=annotations(name)
            row.update(sha256=sha(DATA/name),annotation_frames=len(gt))
            cap=cv2.VideoCapture(str(DATA/name))
            if not cap.isOpened(): raise ValueError('Video failed to open')
            decoded=0; previous=-1; next_sample=0
            while True:
                if time.monotonic()-begun>240: raise TimeoutError('Four-minute per-clip processing bound')
                ok,frame=cap.read()
                if not ok: break
                decoded+=1
                if decoded<start: continue
                if end is not None and decoded>end: break
                t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if not math.isfinite(t) or t<previous: raise ValueError('Invalid presentation timestamps')
                previous=t
                if t+1e-7<next_sample: continue
                next_sample=t+1/3
                h,w=frame.shape[:2]; nh=max(1,int(h*480/w)); frame=cv2.resize(frame,(480,nh))
                found=engine.infer(frame)
                scaled=[dict(d,box=[d['box'][0]*480/w,d['box'][1]*nh/h,d['box'][2]*480/w,d['box'][3]*nh/h]) for d in gt.get(decoded,[])]
                row['samples'].append({'frame':decoded,'seconds':t,'size':[480,nh],'detections':found,'publisher_objects':scaled})
            if len(row['samples'])<2: raise ValueError('Too few analysis samples')
            row.update(status='complete',decoded_frames=decoded,first_seconds=row['samples'][0]['seconds'],last_seconds=previous)
        except Exception as exc:
            row.update(status='error',error=type(exc).__name__+': '+str(exc))
        finally:
            if cap is not None: cap.release()
        report['cases'].append(row)
        (output/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(name,row['status'],len(row['samples']),'samples',flush=True)
    return 0 if all(r['status']=='complete' for r in report['cases']) else 1


def replay(output):
    sys.path.insert(0,str(ROOT/'edge'))
    import numpy as np
    from detectors import YoloBehaviorSuite
    from site_behaviors import SiteBehaviorSuite
    source=json.loads((output/'predictions.json').read_text())
    base={k:{'enabled':False} for k in ('person_down','fight','crowd','panic','intrusion','loitering','abandoned_object','snatch','vehicles')}
    policies=[('crowd',SiteBehaviorSuite,{'crowd':{'enabled':True,'warn':8,'critical':15,'seconds':5}},{}),
              ('intrusion',YoloBehaviorSuite,{'intrusion':{'enabled':True}},{'zones':[{'name':'presence-demo','type':'intrusion','points':FULL}]}),
              ('empty_zone',YoloBehaviorSuite,{'intrusion':{'enabled':True}},{'zones':[{'name':'empty-control','type':'intrusion','points':EMPTY}]}),
              ('loitering_5s_demo',YoloBehaviorSuite,{'loitering':{'enabled':True,'seconds':5}},{'zones':[{'name':'dwell-demo','type':'loiter','points':FULL}]}),
              ('vehicles',SiteBehaviorSuite,{'vehicles':{'enabled':True},'road_region':FULL},{}),
              ('unattended_default',YoloBehaviorSuite,{'abandoned_object':{'enabled':True}},{}),
              ('unattended_5s_demo',YoloBehaviorSuite,{'abandoned_object':{'enabled':True,'seconds':5}},{}),
              ('fight',YoloBehaviorSuite,{'fight':{'enabled':True}},{}),
              ('panic',YoloBehaviorSuite,{'panic':{'enabled':True}},{}),
              ('snatch',YoloBehaviorSuite,{'snatch':{'enabled':True}}, {})]
    rows=[]
    for case in source['cases']:
        if case['status']!='complete': continue
        for label,implementation,cfg,camera in policies:
            class Engine:
                rows=[]
                def infer(self,frame): return self.rows
            engine=Engine(); suite=implementation(camera=camera,config={**base,**cfg},engine=engine)
            events=[]
            for sample in case['samples']:
                engine.rows=sample['detections']; w,h=sample['size']
                events.extend({'seconds':sample['seconds'],'event':e} for e in suite.detect(np.zeros((h,w,3),dtype=np.uint8),now=1_000_000+sample['seconds']))
            rows.append({'file':case['file'],'policy':label,'event_counts':dict(Counter(e['event']['type'] for e in events)), 'events':events})
    report={'results':rows,'note':'Real neural detections replayed through unchanged rules. No criminal intent or medical/danger truth is inferred. Object model has no pose keypoints; fight validation here is geometry-only; separate pose run required.',
            'source_sha256':{p:sha(ROOT/p) for p in ('edge/detectors.py','edge/site_behaviors.py')}}
    (output/'rule-outputs.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in row.items() if k!='events'} for row in rows],indent=2))


def score(output):
    """Independent policy oracles, using publisher IDs rather than YOLO tracks."""
    cached=json.loads((output/'predictions.json').read_text())
    results=json.loads((output/'rule-outputs.json').read_text())
    found={(r['file'],r['policy']):r for r in results['results']}
    scores=[]; observations=[]
    for case in cached['cases']:
        if case['status']!='complete': continue
        street=case['file'].endswith('.avi')
        if street:
            # Publisher traffic boxes are inside its evaluation mask; recall is
            # reported on those boxes only. No unmatched-object FP claim.
            total=matched=labelled_samples=0
            for sample in case['samples']:
                truth=[d for d in sample['publisher_objects'] if d['cls']==2]
                labelled_samples+=bool(truth)
                remaining=[d for d in sample['detections'] if d['cls'] in (2,3,5,7)]
                total+=len(truth)
                for item in truth:
                    a=item['box']; best=-1; overlap=0
                    for i,d in enumerate(remaining):
                        b=d['box']; intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                        union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
                        value=intersection/union if union>0 else 0
                        if value>overlap: best,overlap=i,value
                    if overlap>=.5:
                        matched+=1;remaining.pop(best)
            observations.append({'file':case['file'],'publisher_vehicle_boxes':total,'matched_iou_05':matched,
                'recall_at_iou_05':matched/total if total else None,'labelled_sample_frames':labelled_samples,
                'scope':'Publisher annotated car boxes only; no unmatched-detection precision or traffic-violation truth. Publisher frame numbers interpreted as one-based video frames; exact tracking metrics not claimed.'})
            continue
        crowd_since=None; crowd_expected=False; dwell={}; dwell_expected=False
        presence=False; empty_presence=False; errors=[]
        for sample in case['samples']:
            t=sample['seconds']; people=[d for d in sample['publisher_objects'] if d['cls']==0]
            w,h=sample['size']; model_count=sum(d['cls']==0 for d in sample['detections'])
            errors.append(abs(model_count-len(people)))
            presence=presence or bool(people)
            empty_presence=empty_presence or any(0 <= (d['box'][0]+d['box'][2])/2/w <= .02 and 0<=d['box'][3]/h<=.02 for d in people)
            if len(people)>=8:
                crowd_since=t if crowd_since is None else crowd_since
                crowd_expected=crowd_expected or t-crowd_since>=5
            else: crowd_since=None
            ids={d['id'] for d in people}
            dwell={ident:value for ident,value in dwell.items() if ident in ids}
            for ident in ids:
                dwell.setdefault(ident,t)
                dwell_expected=dwell_expected or t-dwell[ident]>=5
        expected={'crowd':crowd_expected,'intrusion':presence,'empty_zone':empty_presence,'loitering_5s_demo':dwell_expected}
        for policy,truth in expected.items():
            events=found[(case['file'],policy)]['events']; prediction=bool(events)
            scores.append({'file':case['file'],'policy':policy,'expected_alert':truth,'actual_alert':prediction,
                'outcome':('TP' if prediction else 'FN') if truth else ('FP' if prediction else 'TN'),
                'label_source':'publisher pedestrian IDs/boxes with predeclared artificial site policy; not real authorization/danger labels'})
        observations.append({'file':case['file'],'sampled_frames':len(case['samples']),
            'mean_absolute_pedestrian_count_difference':sum(errors)/len(errors),
            'max_absolute_pedestrian_count_difference':max(errors),
            'scope':'Annotation comparison. CAVIAR excludes some stationary people; it is not a fully annotated person-detection benchmark.'})
    report={'cases':scores,'policy_agreement':dict(Counter(r['outcome'] for r in scores)),
            'count_observations':observations,
            'unattended_default_positive':'Selected LeftBag recording is shorter than default60-second unattended hold; default positive acceptance not established.',
            'unsupported_labels':['real panic/intent','snatch theft','traffic violations','operational congestion','site authorization'],
            'note':'Policy agreement is development screening, not independent field accuracy. No operating thresholds were tuned to these results.'}
    (output/'annotation-scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['collect','replay','score'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); output=args.output.resolve()
    if not output.is_relative_to(WORK.resolve()): parser.error('Output must remain in the site-validation report root')
    sys.exit(collect(output) if args.action=='collect' else replay(output) if args.action=='replay' else score(output))
