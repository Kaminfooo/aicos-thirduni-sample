"""Prepare the owner-requested full-code release, excluding private runtime data.

This creates a NEW local directory and an auditable manifest; it never uploads,
changes repository visibility, assigns a code license or deploys a service.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'.runtime') or out.exists():
        parser.error('Choose a new project-local .runtime output directory')
    out.mkdir(parents=True);release=out/'release';release.mkdir()
    files={}
    groups={'backend':{'.py','.txt','.json'},'edge':{'.py','.txt'},
            'dashboard':{'.html','.js','.jsx','.css','.txt'},
            'deploy':{'.py','.txt','.yaml'},'tools':{'.py','.cjs'},'tests':{'.py'}}
    def add(source,target=None):
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(ROOT):
            raise RuntimeError('Invalid release source')
        name=target or source.relative_to(ROOT).as_posix()
        if name.startswith(('.runtime/','data/','results/','runs/')):
            raise RuntimeError('Private runtime tree cannot be published')
        files[name]=source.read_bytes()
    for group,extensions in groups.items():
        for path in (ROOT/group).rglob('*'):
            if path.is_file() and path.suffix in extensions and not any(p.startswith('.') or p=='__pycache__' for p in path.relative_to(ROOT).parts):
                add(path)
    for name in ('.env.example','VERSION','docker-compose.yml','backend/Dockerfile','deploy/Dockerfile.workspace',
                 'config/cameras.example.yaml','runtime/inference-requirements.txt',
                 'models/pretrained/MANIFEST.json','models/pretrained/THIRDUNI-EXTENSIONS.json',
                 'docs/REVIEWER-GUIDE.md','docs/CAMERA-WALL-20261029.md','docs/CAMERA-MAP-AND-CAPACITY-20261029.md',
                 'docs/PLATFORM-READINESS-20261029.md','docs/THIRD_PARTY_MODEL_INVENTORY.md'):
        add(ROOT/name)
    # Publicly attributed laboratory media only; no owner recordings/screenshots.
    for path in (ROOT/'deploy/free-demo-assets').iterdir():
        if path.is_file() and path.name in {'controlled-fall.mp4','walking.mp4','trigger.jpg','context.gif','seed.json','LICENSE.txt','credits.html','evaluation-summary.json'}:
            add(path)
    add(ROOT/'submission/full-platform-release/README.md','README.md')
    add(ROOT/'README.md','docs/LEGACY-PLATFORM-README.md')
    add(ROOT/'submission/thirduni-final-draft/VIDEO-RECORDING-CURRENT-20261029.md','docs/VIDEO-RECORDING-GUIDE.md')
    for name in ('detectors.py','violence.py'):
        add(ROOT/'.runtime/starter-review/source/edge'/name,'tests/fixtures/starter-source/edge/'+name)
    old=ROOT/'.runtime/full-source-portability-20261026/candidate'
    for path in (old/'docs/dependencies').rglob('*'):
        if path.is_file():add(path,path.relative_to(old).as_posix())
    notices=(old/'THIRD-PARTY-NOTICES.md').read_text(encoding='utf-8')
    notices+='\n\n## Current release preparation — 29 October 2026\n\nThe owner requested full-code publication for ThirdUni. Source, example configuration and attributed laboratory assets are prepared; private credentials, environments, databases and personal footage are excluded. No new blanket source license is assigned by this preparation. Model binaries are not redistributed in Git; the setup script obtains pinned reference files from their publishers and verifies SHA-256. Publisher labels and underlying license obligations remain applicable. Leaflet 1.9.4 is now locally served with its BSD license at dashboard/map-assets/LEAFLET-LICENSE.txt. Claude, ChatGPT and Codex assistance is disclosed in the README.\n'
    files['THIRD-PARTY-NOTICES.md']=notices.encode('utf-8')
    # This historical phone configuration is only a settings fixture. Never ship
    # the owner's live URL, GPS point, named site or device configuration.
    phone=yaml.safe_load((ROOT/'config/cameras-phone.yaml').read_text(encoding='utf-8'))
    def portable(value):
        if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
        if isinstance(value,list):return [portable(v) for v in value]
        if isinstance(value,str) and re.match(r'^[A-Za-z]:[/\\]',value):return 'models/pretrained/'+Path(value.replace('\\','/')).name
        return value
    fixture={'settings':portable(phone['settings']),'cameras':[{'id':'CAM-LIVE','name':'Example camera','district':'Example site','lat':0.0,'lon':0.0,'rtsp_url':'rtsp://REPLACE_USER:REPLACE_PASSWORD@camera.example.invalid/stream'}]}
    files['config/cameras-phone.yaml']=yaml.safe_dump(fixture,sort_keys=False).encode()
    files['.gitignore']=b'.env\n.env.*\n!.env.example\n.runtime/\n.venv/\n.venv-inference/\n__pycache__/\n*.pyc\n*.db\n*.db-*\n*.pem\n*.key\n*.dpapi\n*.pt\n*.onnx\n*.engine\ndata/\nresults/\nruns/\nsnapshots/\n*.download\n'
    files['.dockerignore']=b'.git\n.runtime\n.venv\n.venv-inference\n__pycache__\n*.pyc\ndata\nresults\nruns\n*.db\n*.pem\n*.key\n*.dpapi\n'
    files['render.yaml']=files['deploy/render-workspace.yaml']
    files['docs/VALIDATION-STATUS.md']=('''# Validation status — 29 October 2026

The working tree audit passed 207 Python checks and 92 JavaScript checks before Linux-host portability changes. Release tests are recorded separately; do not infer a fresh installation test from an existing-environment run.

The fresh API reference cases produced person-down, fire, smoke and configured zone-entry alerts with verified evidence/review/approval. The short walking reference stayed quiet. These are selected development/reference examples, not a representative accuracy benchmark.

Two browser-played real-time surveillance tests missed the controlled person-down event (39 and 46 frames), despite a passing sampled reference replay. The known violence reference also remained a miss. Physical camera reliability and broad scene performance are not established. No model thresholds or weights were changed to hide those outcomes. Follow the supplied recording workflow for the demonstrated core and retain its operating limits.

The older public sample uses recorded output and a simplified audit; the full operations workspace runs fresh inference and uses the private application's evidence implementation against an isolated demo database. Only an actual successful cloud deployment establishes that the full current workspace is online.
''').encode()
    flags=[]
    token_patterns=[rb'(?m)^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{30,}',rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}']
    for name,data in files.items():
        if any(re.search(pattern,data) for pattern in token_patterns):flags.append({'path':name,'kind':'credential/private-key pattern'})
        if name.endswith('.py'):ast.parse(data,filename=name)
        path=release/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    if flags:
        (out/'SCAN-BLOCKED.json').write_text(json.dumps(flags,indent=2));raise RuntimeError('Release scan requires review; filenames only in SCAN-BLOCKED.json')
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}
    manifest=''.join(sha+'  '+name+'\n' for name,sha in hashes.items())
    (release/'RELEASE-SHA256SUMS.txt').write_text(manifest,encoding='utf-8')
    archive=out/'AICOS-full-platform-source.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        for name,data in sorted(files.items()):bundle.writestr(name,data)
        bundle.writestr('RELEASE-SHA256SUMS.txt',manifest)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert all(hashlib.sha256(bundle.read(name)).hexdigest()==sha for name,sha in hashes.items())
    result={'file_count':len(files),'source_bytes':sum(map(len,files.values())),'archive_bytes':archive.stat().st_size,
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'manifest_verified':True,'credential_pattern_findings':flags,
        'excluded':['live camera configuration','home coordinates','personal recordings/screenshots','databases and signing keys','environment folders','runtime logs and traces','model binaries','unrelated downloaded candidate models'],
        'published':False,'visibility_changed':False,'deployed':False,'source_sha256':hashes}
    (out/'RELEASE-REVIEW.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
