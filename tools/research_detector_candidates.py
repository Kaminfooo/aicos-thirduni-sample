"""Bounded primary-source metadata discovery. No live model changes."""
import json
from pathlib import Path
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from acquire_capability_validation import CTX
WORK=ROOT/'.runtime/detector-upgrade-20261028'
REPOS={'yolo26_violence':'Manas-Maahir/violence-detection-yolov26',
       'cuenet':'manhdungcr7/cuenet-rwf2000',
       'videomae':'HappyGook/videomae-violence-detector'}


def get(url,limit=2_000_000):
    req=urllib.request.Request(url,headers={'User-Agent':'AICOS-research/1.0'})
    with urllib.request.urlopen(req,context=CTX,timeout=30) as response:
        data=response.read(limit+1)
    if len(data)>limit:raise ValueError('Metadata bound exceeded')
    return data


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    for name,repo in REPOS.items():
        path=WORK/(name+'-metadata.json')
        if path.exists():meta=json.loads(path.read_text())
        else:
            meta=json.loads(get('https://huggingface.co/api/models/'+repo+'?blobs=true'))
            path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print(name,meta['sha'],[(x['rfilename'],x.get('size')) for x in meta['siblings']],flush=True)
        for row in meta['siblings']:
            filename=row['rfilename']
            if (filename.endswith(('.md','.json','.yaml','.py')) and '/' not in filename and
                row.get('size',0)<500_000):
                target=WORK/(name+'-'+filename)
                if not target.exists():target.write_bytes(get('https://huggingface.co/'+repo+'/resolve/'+meta['sha']+'/'+filename))


if __name__=='__main__':main()
