"""Package only the independent sample. Never exports the private AICOS project."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'thirduni-public-sample'
FILES = ('app.py', 'web/index.html', 'test_sample.py', 'README.md', 'NOTICE.md',
         'requirements.txt', 'render.yaml', '.gitignore', 'assets/record.json',
         'assets/trigger.jpg', 'assets/context.gif', 'assets/controlled-fall.mp4',
         'assets/walking.mp4', 'assets/LICENSE.txt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / '.runtime'):
        parser.error('Choose a new project .runtime output directory')
    contents = {}
    for name in FILES:
        source = SAMPLE / name
        if source.is_symlink() or not source.resolve().is_relative_to(SAMPLE):
            raise RuntimeError('Linked or outside sample source')
        contents[name] = source.read_bytes()
    imported = set()
    for node in ast.walk(ast.parse(contents['app.py'])):
        if isinstance(node, ast.Import):
            imported.update(item.name.split('.')[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or '').split('.')[0])
    if imported - {'contextlib', 'hashlib', 'json', 'os', 'pathlib', 're', 'secrets', 'sqlite3',
                   'tempfile', 'threading', 'time', 'urllib', 'fastapi', 'pydantic', 'starlette'}:
        raise RuntimeError('Unexpected dependency in standalone application')
    record = json.loads(contents['assets/record.json'])
    for name, sha in record['media_sha256'].items():
        if hashlib.sha256(contents['assets/' + name]).hexdigest() != sha:
            raise RuntimeError('Licensed media hash mismatch')
    output.mkdir(parents=True)
    repo = output / 'repository'
    repo.mkdir()
    for name, data in contents.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = ''.join(hashlib.sha256(data).hexdigest() + '  ' + name + '\n' for name, data in sorted(contents.items()))
    (repo / 'SHA256SUMS.txt').write_text(manifest, encoding='utf-8')
    archive = output / 'aicos-thirduni-minimal-sample.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for name, data in contents.items():
            bundle.writestr(name, data)
        bundle.writestr('SHA256SUMS.txt', manifest)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError('Archive integrity failure')
        for name, data in contents.items():
            if bundle.read(name) != data:
                raise RuntimeError('Archive content mismatch')
    result = {'files': len(contents), 'application_files': 2, 'proposed_repository': 'kaminfooo/aicos-thirduni-sample',
              'archive_bytes': archive.stat().st_size, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'private_backend_or_detector_included': False, 'private_credentials_or_recordings_included': False,
              'model_weights_included': False, 'published': False, 'cloud_deployment_tested': False,
              'competition_sample_acceptance_confirmed': False, 'application_imports': sorted(imported)}
    (output / 'publication-review.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
