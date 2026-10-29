"""Strict offline VideoMAE candidate screen; no remote code or live replacement."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.runtime/detector-upgrade-20261028'
sys.path.insert(0,str(WORK/'deps-recovered'))
sys.path.insert(0,str(ROOT/'tools'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    rows=json.loads((WORK/'YOLO26-EVAL-PLAN.json').read_text())['cases']
    model=next(r for r in json.loads((WORK/'ACQUISITION-PLAN.json').read_text())['files'] if r['name']=='videomae.safetensors')
    plan=dict(prepared_before_predictions=True,model=model,cases=rows,frames=16,threshold=.5,
        preprocessing='Publisher RGB resize shortest edge224, center crop224, ImageNet mean/std, uniform actual decoded whole-clip sample; no padding.',
        gate='At least75% sensitivity, at most1 false positive among11 negatives. Separate causal/holdout/CPU gates before integration.',
        classification='Class0 nonviolent, class1 violent per pinned author README. Single softmax.',
        scope='Whole-clip model screen, not causal live alerts or field accuracy. No threshold search. Holdout untouched.',
        compatibility='Explicit query/key/value bias layout retained; all three learned biases strict-loaded. Pre-load original-runtime equivalence checked per layer.')
    p=WORK/'VIDEOMAE-COMPAT-EVAL-PLAN.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:p.write_text(json.dumps(plan,indent=2))
    return plan


def evaluate():
    plan=prepare();out=WORK/'videomae-evaluation-compatible'
    if out.exists():raise ValueError('Preserve prior evaluation')
    out.mkdir();tmp=out/'tmp';tmp.mkdir();os.environ.update(TEMP=str(tmp),TMP=str(tmp),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    import cv2,numpy as np,torch
    from safetensors.torch import load_file
    from transformers import VideoMAEConfig,VideoMAEForVideoClassification,VideoMAEImageProcessor
    torch.set_num_threads(2);path=WORK/'models/videomae.safetensors';assert sha(path)==plan['model']['sha256']
    config=json.loads((WORK/'videomae-config.json').read_text());config['num_labels']=2
    model=VideoMAEForVideoClassification(VideoMAEConfig(**config))
    from videomae_qkv_compat import adapt
    compatibility=adapt(model)
    state=load_file(str(path),device='cpu');result=model.load_state_dict(state,strict=True);del state;model.eval()
    processor=VideoMAEImageProcessor(**json.loads((WORK/'videomae-preprocessor_config.json').read_text()))
    report=dict(complete=False,load=dict(missing=list(result.missing_keys),unexpected=list(result.unexpected_keys),compatibility=compatibility),
        model_sha256=sha(path),plan_sha256=sha(WORK/'VIDEOMAE-COMPAT-EVAL-PLAN.json'),cases=[])
    started=time.monotonic()
    for row in plan['cases']:
        record=dict(row,status='running');media=ROOT/row['file'];cap=cv2.VideoCapture(str(media));times=[]
        try:
            assert sha(media)==row['sha256'],'Changed media'
            if not cap.isOpened():raise ValueError('Cannot open video')
            while True:
                ok,frame=cap.read()
                if not ok:break
                t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if not np.isfinite(t) or times and t<times[-1] or t>180:raise ValueError('Invalid/out-of-budget timeline')
                times.append(t)
            cap.release()
            if len(times)<16:raise ValueError('Too few actual frames; padding forbidden')
            indices=np.linspace(0,len(times)-1,16,dtype=int);wanted=set(indices.tolist());images=[];number=0
            cap=cv2.VideoCapture(str(media))
            while True:
                ok,frame=cap.read()
                if not ok:break
                if number in wanted:images.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                number+=1
            if number!=len(times) or len(images)!=16:raise ValueError('Inconsistent sequential decode')
            inputs=processor(images,return_tensors='pt');tick=time.monotonic()
            with torch.inference_mode():logits=model(**inputs).logits
            if logits.shape!=(1,2) or not torch.isfinite(logits).all():raise ValueError('Invalid logits')
            score=torch.softmax(logits,dim=-1)[0,1].item();positive=score>=.5
            record.update(status='complete',score=score,decoded_frames=number,sampled_indices=indices.tolist(),
                sampled_pts=[times[i] for i in indices],inference_seconds=time.monotonic()-tick,
                outcome=('TP' if positive else 'FN') if row['positive'] else ('FP' if positive else 'TN'))
        except Exception as exc:record.update(status='error',error=str(exc))
        finally:cap.release()
        report['cases'].append(record);(out/'report.json').write_text(json.dumps(report,indent=2));print(Path(row['file']).name,record.get('outcome',record.get('error')),flush=True)
        if time.monotonic()-started>600:break
    counts={k:sum(r.get('outcome')==k for r in report['cases']) for k in ('TP','FN','FP','TN')}
    report.update(complete=len(report['cases'])==len(plan['cases']) and all(r['status']=='complete' for r in report['cases']),
        summary=counts,holdout_evaluated=False,live_replacement=False,elapsed_seconds=time.monotonic()-started)
    report['gate_passed']=report['complete'] and counts['TP']>=.75*(counts['TP']+counts['FN']) and counts['FP']<=1
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('cases','load')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate']);a=p.parse_args();prepare() if a.action=='prepare' else evaluate()
