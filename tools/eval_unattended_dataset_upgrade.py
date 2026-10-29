"""Real ABODA replay: compare bag recognition and timing, no invented intent labels."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.runtime/detector-upgrade-20261028'
sys.path.insert(0,str(ROOT/'edge'))
sys.path.insert(0,str(ROOT/'tools'))
from avi_reference_clock import read_clock


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    acquired=json.loads((WORK/'OBJECT-DATA-ACQUISITION.json').read_text())
    cases=[r for r in acquired if r['file'].endswith('.avi')]
    assert len(cases)==3 and all(r['status']=='verified' for r in cases)
    baseline=ROOT/'models/pretrained/yolo26n.pt'
    official=next(r for r in json.loads((WORK/'official-object-assets.json').read_text()) if r['name']=='yolo26s.pt')
    plan=dict(prepared_before_predictions=True,cases=cases,models={
        'nano':dict(path=str(baseline.relative_to(ROOT)),sha256='9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'),
        'small':dict(path='.runtime/detector-upgrade-20261028/models/yolo26s.pt',sha256=official['digest'].removeprefix('sha256:'))},
        conf=.4,imgsz=640,sampling='At most1fps at480px long edge; sequential decoded frame index timed by native AVI stream dwScale/dwRate; actual decode count must match the native header.',
        clocks={r['file']:read_clock(ROOT/r['file']) for r in cases},
        holds=[60,5],scope='Publisher abandoned-object references. Report observed bag frames and alerts at60s original/5s explicit demo policy separately. No exact onset/owner/box ground truth supplied; no precision/recall or automatic promotion claim.')
    path=WORK/'UNATTENDED-CFR-EVAL-PLAN.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    return plan


def evaluate(kind):
    plan=prepare();out=WORK/('unattended-'+kind+'-cfr')
    if out.exists():raise ValueError('Preserve previous run')
    out.mkdir();tmp=out/'tmp';tmp.mkdir();os.environ.update(TEMP=str(tmp),TMP=str(tmp),YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(tmp/'yolo'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    import cv2,torch
    from detectors import UltralyticsEngine,YoloBehaviorSuite,BAG_CLASSES
    torch.set_num_threads(2);model=plan['models'][kind];path=ROOT/model['path'];assert sha(path)==model['sha256']
    engine=UltralyticsEngine(str(path),.4,640,classes=(0,24,26,28))
    class Cache:
        rows=[]
        def infer(self,frame):return self.rows
    cache=Cache();report=dict(model=kind,model_sha256=sha(path),complete=False,cases=[])
    for row in plan['cases']:
        path=ROOT/row['file'];assert sha(path)==row['sha256'];cap=cv2.VideoCapture(str(path));due=0;prev=-1;decoded=0
        clock=plan['clocks'][row['file']]
        record=dict(file=row['file'],sha256=row['sha256'],clock=clock,status='running',samples=[],events={str(hold):[] for hold in plan['holds']})
        detectors={}
        for hold in plan['holds']:
            flags={key:{'enabled':key=='abandoned_object'} for key in ('person_down','fight','crowd','panic','intrusion','loitering','vehicles','abandoned_object','snatch')}
            flags['abandoned_object']['seconds']=hold;detectors[hold]=YoloBehaviorSuite(config=flags,engine=cache)
        try:
            if not cap.isOpened():raise ValueError('Cannot open video')
            while True:
                ok,frame=cap.read()
                if not ok:break
                t=decoded*clock['scale']/clock['rate'];decoded+=1
                if t<prev or t>180:raise ValueError('Invalid/out-of-budget timestamp')
                prev=t
                if t+1e-6<due:continue
                due=t+1;h,w=frame.shape[:2];scale=min(480/w,480/h,1);frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
                tick=time.monotonic();cache.rows=engine.infer(frame);latency=time.monotonic()-tick
                record['samples'].append(dict(seconds=t,decoded_frame_index=decoded-1,detections=cache.rows,inference_seconds=latency))
                for hold,detector in detectors.items():
                    for event in detector.detect(frame,now=1_000_000+t):record['events'][str(hold)].append(dict(seconds=t,event=event))
            if decoded!=clock['frames']:raise ValueError('Actual decode count differs from native stream length')
            record.update(status='complete',decoded_frames=decoded,last_pts=prev,
                bag_observation_frames=sum(any(d['cls'] in BAG_CLASSES for d in r['detections']) for r in record['samples']))
        except Exception as exc:record.update(status='error',error=str(exc))
        finally:cap.release()
        report['cases'].append(record);(out/'report.json').write_text(json.dumps(report,indent=2));print(Path(row['file']).name,record.get('bag_observation_frames'),{k:len(v) for k,v in record['events'].items()},flush=True)
    report['complete']=all(r['status']=='complete' for r in report['cases']);report['automatic_promotion']=False
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate']);p.add_argument('--model',choices=['nano','small']);a=p.parse_args();prepare() if a.action=='prepare' else evaluate(a.model)
