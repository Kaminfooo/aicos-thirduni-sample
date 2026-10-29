"""Predeclare disjoint new clip sets before candidate predictions."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.runtime/violence-improvement-20261028';DATA=ROOT/'data/violence_improvement_20261028'
sys.path.insert(0,str(ROOT/'tools'));from acquire_capability_validation import get

def prepare():
 tree=json.loads((ROOT/'.runtime/capability-validation-20261027/airtlab-tree.json').read_text());rows=[]
 for split,numbers in [('development',range(13,19)),('untouched_validation',range(19,25))]:
  for label in ('violent','non-violent'):
   for number in numbers:
    path=f'violence-detection-dataset/{label}/cam1/{number}.mp4';item=next(r for r in tree['tree'] if r['path']==path)
    rows.append({'split':split,'label':label,'positive':label=='violent','publisher_path':path,'bytes':item['size'],'git_blob_sha1':item['sha'],
     'file':str((DATA/(label+'-'+str(number)+'.mp4')).relative_to(ROOT)),'url':f"https://raw.githubusercontent.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/{tree['sha']}/{path}"})
 plan={'declared_before_new_predictions':True,'revision':tree['sha'],'rows':rows,'selection':'cam1 scenarios13–18/class development,19–24/class untouched validation; cam2 excluded as correlated views.',
  'thresholds':'Publisher defaults only: X3D16 frames0.4, R(2+1)D32 frames0.5. No validation outcomes used for selection.',
  'promotion_gate':'Compare against frozen baseline on existing24 and development; then evaluate selected candidate once on untouched validation and cross-source confusions. No automatic live enablement.',
  'measurement':'Clip classification on whole short recordings first; causal-window alerting is a separate gate. Preserve source labels, including slaps/pushes/weapons; no relabeling to improve scores.',
  'limits':'Research/educational local use only. Same room/actors and unknown training overlap; not independent public-camera accuracy.'}
 p=WORK/'DATA-PLAN.json'
 if p.exists():assert json.loads(p.read_text())==plan
 else:p.write_text(json.dumps(plan,indent=2),encoding='utf-8')
 print('Predeclared',len(rows),'clips, bytes',sum(r['bytes'] for r in rows))

def acquire():
 DATA.mkdir(parents=True,exist_ok=True);plan=json.loads((WORK/'DATA-PLAN.json').read_text());records=[]
 for row in plan['rows']:
  target=ROOT/row['file'];data=target.read_bytes() if target.exists() else get(row['url'],row['bytes'])[0]
  actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  if len(data)!=row['bytes'] or actual!=row['git_blob_sha1']:raise ValueError('Pinned publisher blob mismatch')
  if not target.exists():target.write_bytes(data)
  records.append({**row,'sha256':hashlib.sha256(data).hexdigest()});print(row['split'],target.name,len(data),'verified',flush=True)
  (WORK/'DATA-ACQUISITION-PROGRESS.json').write_text(json.dumps(records,indent=2))
 (DATA/'ATTRIBUTION.json').write_text(json.dumps({'source':'https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos','revision':plan['revision'],'terms':'Publisher research/educational use; no redistribution','files':records},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','acquire']);a=p.parse_args();globals()[a.action]()
