"""Pinned YOLO26 violence candidate: causal replay, fixed gates, no live replacement."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.runtime/detector-upgrade-20261028'
sys.path.insert(0,str(ROOT/'edge'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    data=json.loads((ROOT/'data/violence_improvement_20261028/ATTRIBUTION.json').read_text())['files']
    rows=[dict(file=r['file'],sha256=r['sha256'],positive=r['positive'],group='development') for r in data if r['split']=='development']
    cross=json.loads((ROOT/'.runtime/violence-improvement-20261028/CROSS-SOURCE-CASES.json').read_text())
    rows.extend(dict(r,group='cross_source') for r in cross)
    model=next(r for r in json.loads((WORK/'ACQUISITION-PLAN.json').read_text())['files'] if r['name']=='violence-yolo26s.pt')
    plan=dict(prepared_before_predictions=True,model=model,cases=rows,
        settings=dict(kind='violence',conf=.5,imgsz=512,seconds=2,min_observations=3,gap_tolerance=1,cooldown_seconds=60),
        gate='At least75% clip sensitivity and at most1 false positive among11 negatives; then separate holdout and CPU acceptance before promotion.',
        sampling='Causal actual decoded timestamps, at most3 analysis fps,480-pixel long edge. No threshold tuning or padding.',
        training_overlap='Unknown. Research comparison, not field accuracy.')
    p=WORK/'YOLO26-EVAL-PLAN.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:p.write_text(json.dumps(plan,indent=2))
    return plan


def evaluate():
    plan=prepare();out=WORK/'yolo26-violence-evaluation'
    if out.exists():raise ValueError('Preserve previous report')
    out.mkdir();tmp=out/'tmp';tmp.mkdir();os.environ.update(TEMP=str(tmp),TMP=str(tmp),YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(tmp/'yolo'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    import cv2,torch
    from persistent_hazards import PersistentHazardDetector,LocalLabelEngine
    torch.set_num_threads(2);path=WORK/'models/violence-yolo26s.pt'
    assert sha(path)==plan['model']['sha256']
    globals_=torch.serialization.get_unsafe_globals_in_checkpoint(path)
    # The inspected checkpoint contains only installed PyTorch/Ultralytics model
    # classes and NumPy metadata intrinsics; refuse any other module before load.
    allowed=('torch.nn.modules.','ultralytics.nn.modules.','ultralytics.nn.tasks.DetectionModel',
        'ultralytics.utils.tal.TaskAlignedAssigner','ultralytics.utils.IterableSimpleNamespace',
        'ultralytics.utils.loss.','numpy.dtype','numpy._core.multiarray.scalar')
    if any(not name.startswith(allowed) for name in globals_):raise ValueError('Unexpected checkpoint global')
    engine=LocalLabelEngine(str(path),['violence'],.5,512)
    report=dict(complete=False,model_sha256=sha(path),class_names=engine.names,cases=[],plan_sha256=sha(WORK/'YOLO26-EVAL-PLAN.json'))
    start=time.monotonic()
    for row in plan['cases']:
        record=dict(row,status='running',events=[],samples=[]);media=ROOT/row['file'];cap=cv2.VideoCapture(str(media));due=0;previous=-1;decoded=0
        detector=PersistentHazardDetector(plan['settings'],engine=engine)
        try:
            assert sha(media)==row['sha256'],'Changed media'
            if not cap.isOpened():raise ValueError('Cannot decode media')
            while True:
                if time.monotonic()-start>600:raise RuntimeError('Ten-minute inference budget exceeded')
                ok,frame=cap.read()
                if not ok:break
                decoded+=1;t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if t<previous or t>180:raise ValueError('Invalid/out-of-budget timeline')
                previous=t
                if t+1e-8<due:continue
                due=t+1/3;h,w=frame.shape[:2];scale=min(480/w,480/h,1);frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
                tick=time.monotonic();events=detector.detect(frame,now=1_000_000+t)
                if detector._disabled:raise RuntimeError('Candidate model error')
                record['samples'].append(dict(seconds=t,decision=detector.last_decision,inference_seconds=time.monotonic()-tick))
                record['events'].extend(dict(seconds=t,event=e) for e in events)
            if not record['samples']:raise ValueError('No analysis frames')
            record.update(status='complete',decoded_frames=decoded,
                outcome=('TP' if record['events'] else 'FN') if row['positive'] else ('FP' if record['events'] else 'TN'))
        except Exception as exc:record.update(status='error',error=str(exc))
        finally:cap.release()
        report['cases'].append(record);(out/'report.json').write_text(json.dumps(report,indent=2));print(Path(row['file']).name,record.get('outcome',record.get('error')),flush=True)
    counts={k:sum(r.get('outcome')==k for r in report['cases']) for k in ('TP','FN','FP','TN')}
    report.update(complete=all(r['status']=='complete' for r in report['cases']),summary=counts,
        gate_passed=counts['TP']>=.75*(counts['TP']+counts['FN']) and counts['FP']<=1,
        holdout_evaluated=False,live_replacement=False,elapsed_seconds=time.monotonic()-start)
    report['gate_passed'] &= report['complete']
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('cases','class_names')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate']);a=p.parse_args();prepare() if a.action=='prepare' else evaluate()
