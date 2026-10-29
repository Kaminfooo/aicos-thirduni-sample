"""Predeclared research experiment; never integrates a failed model into live inference."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import violence_temporal_eval as reference
WORK=reference.WORK


def prepare():
    data=json.loads((ROOT/'data/violence_improvement_20261028/ATTRIBUTION.json').read_text())['files']
    cases=[dict(file=r['file'],positive=r['positive'],sha256=r['sha256'],group='development') for r in data if r['split']=='development']
    cross=json.loads((WORK/'CROSS-SOURCE-CASES.json').read_text())
    cases.extend(dict(r,group='cross_source') for r in cross)
    plan={'prepared_before_predictions':True,'model':'x3d_once','threshold':.5,
        'window_seconds':4,'step_seconds':1,'frames':16,'confirmation':'two consecutive positive windows',
        'maximum_clip_seconds':180,'gate':'at least 75% sensitivity and at most one false-positive clip; no integration before independent holdout acceptance',
        'scope':'New development and cross-source clips; twelve AIRTLab holdout clips remain untouched. Training overlap unknown. No threshold search.',
        'cases':cases}
    target=WORK/'CAUSAL-WINDOW-PLAN.json'
    if target.exists():assert json.loads(target.read_text())==plan
    else:target.write_text(json.dumps(plan,indent=2))
    return plan


def evaluate():
    plan=prepare();output=WORK/'causal-window-evaluation'
    if output.exists():raise ValueError('Preserve previous evaluation')
    output.mkdir();reference.configure()
    import cv2,numpy as np,torch
    model,load=reference.build('x3d')
    report={'complete':False,'plan_sha256':reference.sha(WORK/'CAUSAL-WINDOW-PLAN.json'),'load':load,'cases':[]}
    start=time.monotonic()
    for row in plan['cases']:
        record=dict(row,windows=[],status='running',confirmed=False)
        path=ROOT/row['file'];cap=cv2.VideoCapture(str(path));buffer=deque();due=4;previous=-1;streak=0;decoded=0
        try:
            if reference.sha(path)!=row['sha256']:raise ValueError('Changed media hash')
            if not cap.isOpened():raise ValueError('Cannot open clip')
            while True:
                if time.monotonic()-start>840:raise RuntimeError('Research deadline exceeded')
                ok,frame=cap.read()
                if not ok:break
                decoded+=1;t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if not np.isfinite(t) or t<previous:raise ValueError('Invalid timeline')
                previous=t
                if t>180:raise ValueError('Clip exceeds predeclared duration; do not silently truncate labels')
                rgb=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(224,224))
                buffer.append((t,rgb))
                while buffer and buffer[0][0]<t-4:buffer.popleft()
                if t+1e-6<due:continue
                if len(buffer)<16:raise ValueError('Insufficient distinct decoded frames')
                indices=np.linspace(0,len(buffer)-1,16).astype(int)
                x=torch.from_numpy(np.stack([buffer[i][1] for i in indices])).float().permute(3,0,1,2)/255
                x=((x-.45)/.225).unsqueeze(0);tick=time.monotonic()
                with torch.inference_mode():raw=model(x)
                if raw.shape!=(1,2) or not torch.isfinite(raw).all():raise ValueError('Invalid output')
                score=raw[0,1].item();streak=streak+1 if score>=.5 else 0
                record['confirmed']|=streak>=2
                record['windows'].append(dict(end=t,first_pts=buffer[0][0],score=score,
                    confirmed=streak>=2,inference_seconds=time.monotonic()-tick))
                due+=1
            if not record['windows']:raise ValueError('No complete causal window')
            record.update(status='complete',decoded_frames=decoded,last_pts=previous,
                outcome=('TP' if record['confirmed'] else 'FN') if row['positive'] else ('FP' if record['confirmed'] else 'TN'))
        except Exception as exc:record.update(status='unscorable',error=str(exc))
        finally:cap.release()
        report['cases'].append(record);(output/'report.json').write_text(json.dumps(report,indent=2))
        print(Path(row['file']).name,record.get('outcome',record.get('error')),len(record['windows']),flush=True)
    counts={k:sum(r.get('outcome')==k for r in report['cases']) for k in ('TP','FN','FP','TN')}
    report.update(complete=all(r['status']=='complete' for r in report['cases']),summary=counts,
        elapsed_seconds=time.monotonic()-start,holdout_evaluated=False,live_integration=False)
    recall=counts['TP']/max(1,counts['TP']+counts['FN'])
    report['development_gate_passed']=report['complete'] and recall>=.75 and counts['FP']<=1
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('cases','load')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate']);args=p.parse_args()
    prepare() if args.action=='prepare' else evaluate()
