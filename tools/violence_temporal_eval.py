"""Offline, strict-load temporal candidate evaluation; no live ingestion."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.runtime/violence-improvement-20261028'
sys.path.insert(0,str((WORK/'deps').resolve()))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def configure():
 temp=WORK/'tmp';temp.mkdir(exist_ok=True);os.environ.update(TEMP=str(temp),TMP=str(temp),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 import torch
 torch.set_num_threads(2)

def build(kind):
 import torch,numpy as np
 from torch import nn
 name='final_x3d_realtime.pt' if kind=='x3d' else 'final_best.pt';path=WORK/'models'/name
 manifest=json.loads((WORK/'MODEL-DOWNLOAD-PLAN.json').read_text());expected=next(r for r in manifest['files'] if Path(r['filename']).name==name)
 if sha(path)!=expected['sha256']:raise ValueError('Candidate weight hash changed')
 # Only the two documented intrinsic NumPy metadata types need allowlisting;
 # never deserialize arbitrary remote model classes or turn weights_only off.
 with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType]):c=torch.load(path,map_location='cpu',weights_only=True)
 if kind=='x3d':
  from pytorchvideo.models.hub import x3d_m
  model=x3d_m(pretrained=False);model.blocks[-1].proj=nn.Sequential(nn.Dropout(.3),nn.Linear(2048,2))
  if not isinstance(model.blocks[-1].activation,nn.Softmax):raise ValueError('Unexpected publisher head activation')
 else:
  from torchvision.models.video import r2plus1d_18
  model=r2plus1d_18(weights=None);model.fc=nn.Sequential(nn.Dropout(.5),nn.Linear(512,2))
 state=c['model']
 if not all(k.startswith('backbone.') for k in state):raise ValueError('Unexpected state prefix')
 result=model.load_state_dict({k[len('backbone.'):]:v for k,v in state.items()},strict=True)
 model.eval();return model,{'checkpoint':str(path.relative_to(ROOT)),'sha256':sha(path),'missing':list(result.missing_keys),'unexpected':list(result.unexpected_keys),'chosen_config':c.get('chosen_config')}

def frame_tensor(path,num_frames):
 import cv2,numpy as np,torch
 # Count actual decoded frames, not unverified container frame-count hints.
 cap=cv2.VideoCapture(str(path));frames=[];times=[];previous=-1
 try:
  if not cap.isOpened():raise ValueError('Cannot open video')
  while True:
   ok,f=cap.read()
   if not ok:break
   t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
   if not math.isfinite(t) or t<previous:raise ValueError('Invalid decoded timestamps')
   previous=t;frames.append(cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),(224,224),interpolation=cv2.INTER_LINEAR));times.append(t)
   if len(frames)>18000:raise ValueError('Frame budget exceeded')
 finally:cap.release()
 if len(frames)<num_frames:raise ValueError('Too few real frames; no padding/repetition permitted')
 indices=np.linspace(0,len(frames)-1,num_frames).astype(int)
 array=np.stack([frames[i] for i in indices]);x=torch.from_numpy(array).float().permute(3,0,1,2)/255
 x=(x-.45)/.225
 return x.unsqueeze(0),{'decoded_frames':len(frames),'sampled_indices':indices.tolist(),'sampled_pts':[times[i] for i in indices],'last_pts':times[-1]}

def prepare():
 p=WORK/'CANDIDATE-EVAL-PLAN.json'
 value={'prepared_before_predictions':True,'modes':{'r2plus1d':{'frames':32,'threshold':.5,'score':'single softmax of logits'},
  'x3d_reference':{'frames':16,'threshold':.4,'score':'publisher wrapper head softmax plus demo second softmax'},
  'x3d_once':{'frames':16,'threshold':.5,'score':'existing model head probability once; separate predeclared compatibility variant'}},
  'preprocess':'Actual decoded-frame uniform whole-clip sample, RGB,224x224 bilinear, mean.45/std.225. No repeated padding. Short-clip classification, not causal alerts or onset latency.',
  'source_issue':'Pinned author X3D wrapper retains Softmax head and its temporal_analyzer applies F.softmax again. Reference reproduced; once-only variant fixed before evaluation. Publisher threshold is not transferred to the variant.',
  'selection':'Known24 + development12; select best balanced accuracy with at most1 false-positive among18 negatives, else no promotion. Evaluate selected mode only once on untouched12. No threshold search.',
  'data_plan_sha256':sha(WORK/'DATA-PLAN.json'),'model_plan_sha256':sha(WORK/'MODEL-DOWNLOAD-PLAN.json')}
 if p.exists():assert json.loads(p.read_text())==value
 else:p.write_text(json.dumps(value,indent=2))

def cases(split):
 if split=='selection':
  old=json.loads((ROOT/'.runtime/capability-validation-20261027/violence-followup/FINAL-SUMMARY-20261028.json').read_text())
  rows=[{'file':r['file'],'sha256':r['sha256'],'positive':r['expected_positive'],'split':'known-development'} for r in old['cases']]
 else:rows=[]
 new=json.loads((ROOT/'data/violence_improvement_20261028/ATTRIBUTION.json').read_text())['files']
 rows.extend(r for r in new if r['split']==('development' if split=='selection' else 'untouched_validation'))
 return rows

def evaluate(kind,split):
 import torch
 if split=='validation':
  decision=json.loads((WORK/'SELECTION-DECISION.json').read_text())
  if decision.get('selected_model')!=kind or not decision.get('gate_passed'):
   raise ValueError('No passing selection decision for this model; validation remains untouched')
 output=WORK/(split+'-'+kind)
 if output.exists():raise ValueError('Prior run preserved; do not overwrite')
 output.mkdir();plan=json.loads((WORK/'CANDIDATE-EVAL-PLAN.json').read_text());modes=['r2plus1d'] if kind=='r2plus1d' else ['x3d_reference','x3d_once']
 n=32 if kind=='r2plus1d' else 16
 model,load=build(kind);rows=cases(split);report={'kind':kind,'split':split,'load':load,'plan_sha256':sha(WORK/'CANDIDATE-EVAL-PLAN.json'),'cases':[],'requested_cases':len(rows),'complete':False}
 for row in rows:
  path=ROOT/row['file'];record={k:row[k] for k in ('file','positive','split')}
  try:
   if sha(path)!=row['sha256']:raise ValueError('Media hash mismatch')
   x,meta=frame_tensor(path,n);start=time.perf_counter()
   with torch.inference_mode():raw=model(x);prob=torch.softmax(raw,dim=1)[0,1].item()
   if raw.shape!=(1,2) or not torch.isfinite(raw).all():raise ValueError('Invalid model output')
   scores={'r2plus1d':prob} if kind=='r2plus1d' else {'x3d_reference':prob,'x3d_once':raw[0,1].item()}
   results={mode:{'score':score,'predicted_positive':score>=plan['modes'][mode]['threshold'],
      'outcome':('TP' if score>=plan['modes'][mode]['threshold'] else 'FN') if row['positive'] else ('FP' if score>=plan['modes'][mode]['threshold'] else 'TN')} for mode,score in scores.items()}
   record.update(status='complete',sha256=row['sha256'],media=meta,results=results,inference_seconds=time.perf_counter()-start,model_output=raw[0].tolist())
  except Exception as e:record.update(status='error',error=type(e).__name__+': '+str(e))
  report['cases'].append(record);(output/'report.json').write_text(json.dumps(report,indent=2));print(Path(row['file']).name,record.get('results',record.get('error')),flush=True)
  if record['status']=='error':break
 report['summary']={mode:{k:sum(r.get('results',{}).get(mode,{}).get('outcome')==k for r in report['cases']) for k in ('TP','FN','FP','TN')} for mode in modes}
 report['complete']=len(report['cases'])==len(rows) and all(r['status']=='complete' for r in report['cases'])
 (output/'report.json').write_text(json.dumps(report,indent=2));(output/'exit.txt').write_text('0' if report['complete'] else '1');print(json.dumps(report['summary']));return 0 if report['complete'] else 1

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate']);p.add_argument('--model',choices=['x3d','r2plus1d']);p.add_argument('--split',choices=['selection','validation'],default='selection');a=p.parse_args();configure();sys.exit(prepare() if a.action=='prepare' else evaluate(a.model,a.split))
