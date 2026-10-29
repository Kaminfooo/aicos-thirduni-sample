"""Acquire pinned research candidates with publisher size/SHA checks."""
import argparse,hashlib,json,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.runtime/detector-upgrade-20261028'
sys.path.insert(0,str(ROOT/'tools'))
from acquire_capability_validation import CTX


def worker(name):
    plan=json.loads((WORK/'ACQUISITION-PLAN.json').read_text());row=next(r for r in plan['files'] if r['name']==name)
    path=WORK/'models'/name
    if path.exists():
        assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];return
    partial=path.with_suffix('.partial')
    if partial.exists():raise ValueError('Prior partial preserved')
    start=time.monotonic();count=0;digest=hashlib.sha256()
    req=urllib.request.Request(row['url'],headers={'User-Agent':'AICOS-bounded-research/1.0','Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,context=CTX,timeout=35) as response,partial.open('xb') as stream:
        while True:
            block=response.read1(256*1024)
            if not block:break
            count+=len(block)
            if count>row['bytes'] or time.monotonic()-start>360:raise RuntimeError('Download bound exceeded')
            digest.update(block);stream.write(block)
    assert count==row['bytes'] and digest.hexdigest()==row['sha256'],'Hash/size mismatch'
    partial.rename(path);print(name,'verified',count,flush=True)


def acquire():
    (WORK/'models').mkdir(exist_ok=True);files=[]
    for key,filename,name in [('yolo26_violence','best.pt','violence-yolo26s.pt'),('videomae','model.safetensors','videomae.safetensors')]:
        meta=json.loads((WORK/(key+'-metadata.json')).read_text());row=next(r for r in meta['siblings'] if r['rfilename']==filename)
        files.append(dict(name=name,repository=meta['id'],revision=meta['sha'],bytes=row['size'],sha256=row['lfs']['sha256'],
            url='https://huggingface.co/'+meta['id']+'/resolve/'+meta['sha']+'/'+filename+'?download=true'))
    plan=dict(purpose='Local comparative research only; no automatic production replacement',files=files)
    target=WORK/'ACQUISITION-PLAN.json'
    if target.exists():assert json.loads(target.read_text())==plan
    else:target.write_text(json.dumps(plan,indent=2))
    for row in files:
        name=row['name']
        with (WORK/(name+'.stdout.txt')).open('x') as out,(WORK/(name+'.stderr.txt')).open('x') as err:
            try:result=subprocess.run([sys._base_executable,'-I','-B',str(Path(__file__).resolve()),'worker','--name',name],cwd=ROOT,stdout=out,stderr=err,timeout=390,creationflags=subprocess.CREATE_NO_WINDOW);code=result.returncode
            except subprocess.TimeoutExpired:code=124
        (WORK/(name+'.exit.txt')).write_text(str(code));print(name,code,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['acquire','worker']);p.add_argument('--name');a=p.parse_args()
    acquire() if a.action=='acquire' else worker(a.name)
