"""Pinned, bounded candidate downloads; no production model replacement."""
import argparse,hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.runtime/violence-improvement-20261028';DATA=WORK/'models'
sys.path.insert(0,str(ROOT/'tools'));from acquire_capability_validation import CTX

def worker(filename):
 plan=json.loads((WORK/'MODEL-DOWNLOAD-PLAN.json').read_text());row=next(r for r in plan['files'] if r['filename']==filename)
 target=DATA/Path(filename).name;partial=target.with_suffix('.download.partial')
 if target.exists():
  assert target.stat().st_size==row['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'];print('Already verified',target.name);return
 if partial.exists():raise RuntimeError('Prior partial preserved; choose explicit recovery after inspection')
 request=urllib.request.Request(row['url'],headers={'User-Agent':'AICOS-local-candidate-evaluation/1.0','Accept-Encoding':'identity'})
 count=0;started=time.monotonic();digest=hashlib.sha256()
 with urllib.request.urlopen(request,context=CTX,timeout=35) as response,partial.open('xb') as stream:
  while True:
   block=response.read1(256*1024)
   if not block:break
   count+=len(block)
   if count>row['bytes'] or time.monotonic()-started>300:raise RuntimeError('Download bound exceeded')
   stream.write(block);digest.update(block)
 if count!=row['bytes'] or digest.hexdigest()!=row['sha256']:raise RuntimeError('Publisher LFS size/SHA256 mismatch')
 partial.rename(target);print('Verified',target.name,count,flush=True)

def acquire():
 DATA.mkdir(parents=True,exist_ok=True);metadata=json.loads((WORK/'school-metadata.json').read_text());files=[]
 for name in ['final/final_x3d_realtime.pt','final/final_best.pt']:
  r=next(x for x in metadata['siblings'] if x['rfilename']==name)
  files.append({'filename':name,'bytes':r['size'],'sha256':r['lfs']['sha256'],'url':'https://huggingface.co/visionlab-ai/school-violence-detection-models/resolve/'+metadata['sha']+'/'+name+'?download=true'})
 plan={'repository':'visionlab-ai/school-violence-detection-models','revision':metadata['sha'],'files':files,'purpose':'isolated temporal-model comparison; publication and commercial licensing not inferred','publisher_thresholds':{'final_x3d_realtime.pt':.4,'final_best.pt':.5}}
 p=WORK/'MODEL-DOWNLOAD-PLAN.json'
 if p.exists():assert json.loads(p.read_text())==plan
 else:p.write_text(json.dumps(plan,indent=2))
 for row in files:
  name=Path(row['filename']).name
  with (WORK/(name+'.download.stdout.txt')).open('x') as out,(WORK/(name+'.download.stderr.txt')).open('x') as err:
   try:r=subprocess.run([sys._base_executable,'-I','-B',str(Path(__file__).resolve()),'worker','--file',row['filename']],cwd=ROOT,stdout=out,stderr=err,timeout=320,creationflags=subprocess.CREATE_NO_WINDOW);code=r.returncode
   except subprocess.TimeoutExpired:code=124
  (WORK/(name+'.download.exit.txt')).write_text(str(code));print(name,'exit',code,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['acquire','worker']);p.add_argument('--file');a=p.parse_args();worker(a.file) if a.action=='worker' else acquire()
