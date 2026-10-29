"""Fetch pinned public reference weights; verify bytes before installing them.

No credentials, auto-discovery or inference. Model weights are not source files.
Publisher provenance/labels are recorded in the manifests; retain their notices.
"""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    tree=ast.parse((ROOT/'deploy/judge_detector_pipeline.py').read_text(encoding='utf-8'))
    hashes=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='HASHES' for t in n.targets))
    entries=json.loads((ROOT/'models/pretrained/MANIFEST.json').read_text())['models']
    entries+=json.loads((ROOT/'models/pretrained/THIRDUNI-EXTENSIONS.json').read_text())['assets']
    catalog={entry['name']:entry for entry in entries}
    directory=ROOT/'models/pretrained';directory.mkdir(parents=True,exist_ok=True)
    for name,expected in hashes.items():
        path=directory/name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:
            print(name+': verified existing reference');continue
        entry=catalog[name]
        if entry['sha256']!=expected or not entry['url'].startswith(('https://github.com/ultralytics/','https://huggingface.co/')):
            raise RuntimeError('Unexpected reference source: '+name)
        temp=directory/(name+'.download')
        try:
            request=urllib.request.Request(entry['url'],headers={'User-Agent':'AICOS-reference-model-setup/1.0'})
            digest=hashlib.sha256();size=0
            with urllib.request.urlopen(request,timeout=120) as response,temp.open('wb') as output:
                while chunk:=response.read(1024*1024):
                    size+=len(chunk)
                    if size>50_000_000:raise RuntimeError('Reference size limit exceeded')
                    digest.update(chunk);output.write(chunk)
            if digest.hexdigest()!=expected or size!=entry['bytes']:
                raise RuntimeError('Reference hash/size mismatch: '+name)
            temp.replace(path);print(name+': downloaded and SHA-256 verified')
        finally:
            temp.unlink(missing_ok=True)


if __name__=='__main__':main()
