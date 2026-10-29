"""Matched real-video comparison for pose-based movement cues; not intent proof."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.runtime/detector-upgrade-20261028'
sys.path.insert(0,str(ROOT/'edge'))


def main():
    cases=json.loads((ROOT/'.runtime/violence-improvement-20261028/CROSS-SOURCE-CASES.json').read_text())
    cases=[dict(r,expected_fight=r['positive']) for i,r in enumerate(cases) if i in (0,1,2,3,7)]
    official=next(r for r in json.loads((WORK/'OFFICIAL-HF-ACQUISITION.json').read_text()) if r['name']=='yolo26s-pose.pt')
    models={'nano':dict(path='models/pretrained/yolo26n-pose.pt',sha256='eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9'),
        'small':dict(path='.runtime/detector-upgrade-20261028/models/yolo26s-pose.pt',sha256=official['sha256'])}
    plan=dict(prepared_before_predictions=True,cases=cases,models=models,settings={'conf':.35,'imgsz':640,'analysis_fps':3},
        policy='Original fight/panic/snatch thresholds and holds. Shared pose inference, separate temporal state.',
        scope='Publisher staged fights vs ordinary walking/meeting/solo boxing. No labelled theft/panic positives; report those as movement-cue observations, not validated intent recall.',
        gate='No promotion without improved fight sensitivity, no ordinary-activity alerts, and complete model/CPU/pipeline checks. This comparison alone cannot establish theft/panic capability.')
    p=WORK/'INTERACTION-POSE-PLAN.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:p.write_text(json.dumps(plan,indent=2))
    out=WORK/'interaction-pose-evaluation'
    if out.exists():raise ValueError('Preserve previous run')
    out.mkdir();tmp=out/'tmp';tmp.mkdir();os.environ.update(TEMP=str(tmp),TMP=str(tmp),YOLO_CONFIG_DIR=str(tmp/'yolo'),YOLO_AUTOINSTALL='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    import cv2,torch
    from detectors import UltralyticsEngine,YoloBehaviorSuite
    torch.set_num_threads(2);engines={}
    for key,model in models.items():
        path=ROOT/model['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==model['sha256']
        engines[key]=UltralyticsEngine(str(path),.35,640,classes=(0,))
    class Cache:
        rows=[]
        def infer(self,frame):return self.rows
    report=dict(complete=False,cases=[],plan_sha256=hashlib.sha256(p.read_bytes()).hexdigest());started=time.monotonic()
    for row in cases:
        path=ROOT/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];cap=cv2.VideoCapture(str(path));due=0;prev=-1
        record=dict(file=row['file'],positive=row['positive'],status='running',models={key:dict(samples=[],events={k:[] for k in ('fight','panic','snatch')}) for key in models})
        states={};cache=Cache()
        for kind in models:
            states[kind]={}
            for selected in ('fight','panic','snatch'):
                cfg={k:{'enabled':k==selected} for k in ('person_down','fight','crowd','panic','intrusion','loitering','vehicles','abandoned_object','snatch')}
                states[kind][selected]=YoloBehaviorSuite(config=cfg,engine=cache)
        try:
            if not cap.isOpened():raise ValueError('Cannot open source')
            while True:
                if time.monotonic()-started>720:raise RuntimeError('Twelve-minute comparison budget exceeded')
                ok,frame=cap.read()
                if not ok:break
                t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if t<prev or t>180:raise ValueError('Invalid/out-of-budget timeline')
                prev=t
                if t+1e-6<due:continue
                due=t+1/3;h,w=frame.shape[:2];scale=min(480/w,480/h,1);frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
                for kind,engine in engines.items():
                    tick=time.monotonic();cache.rows=engine.infer(frame);latency=time.monotonic()-tick
                    record['models'][kind]['samples'].append(dict(seconds=t,inference_seconds=latency,persons=len(cache.rows)))
                    for key,detector in states[kind].items():
                        events=detector.detect(frame,now=1_000_000+t)
                        record['models'][kind]['events'][key].extend(dict(seconds=t,event=e) for e in events)
            record.update(status='complete',last_pts=prev)
        except Exception as exc:record.update(status='error',error=str(exc))
        finally:cap.release()
        report['cases'].append(record);(out/'report.json').write_text(json.dumps(report,indent=2));print(Path(row['file']).name,{kind:{k:len(v) for k,v in r['events'].items()} for kind,r in record['models'].items()},flush=True)
    report.update(complete=all(r['status']=='complete' for r in report['cases']),elapsed_seconds=time.monotonic()-started,live_replacement=False)
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
