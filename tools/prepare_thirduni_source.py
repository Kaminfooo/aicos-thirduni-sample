"""Prepare a local source-only review archive; never publishes or includes runtime data."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def selected_sources():
    allowed = {'backend': {'.py', '.txt'}, 'edge': {'.py', '.txt'},
               'dashboard': {'.html', '.js', '.jsx', '.css'}, 'tests': {'.py'}, 'tools': {'.py'}}
    selected = set()
    for directory, suffixes in allowed.items():
        for path in (ROOT / directory).rglob('*'):
            relative = path.relative_to(ROOT)
            if path.is_file() and path.suffix in suffixes and not any(
                    part.startswith('.') or part == '__pycache__' for part in relative.parts):
                if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                    raise RuntimeError('Source links are excluded: ' + str(relative))
                selected.add(path)
    for relative in ('README.md', '.gitignore', 'VERSION', 'backend/triage_demo_sops.json',
                     'config/cameras.example.yaml', 'docs/THIRDUNI-COMPETITION-README.md',
                     'docs/THIRDUNI-DEMO-VIDEO-SCRIPT.md', 'docs/THIRDUNI-SUBMISSION-PLAN-20261025.md',
                     'docs/THIRDUNI-PROFESSIONAL-UPDATE-20261025.md',
                     'docs/THIRDUNI-PUBLIC-MEDIA-VALIDATION-20261025.md',
                     'docs/THIRDUNI-COMPETITOR-REVIEW-20261025.md',
                     'docs/THIRDUNI-PERSON-DOWN-EVAL-PROTOCOL.md'):
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError('Missing or linked required source: ' + relative)
        selected.add(path)
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / '.runtime') or output.exists():
        parser.error('Choose a new output directory under project .runtime')
    sources = selected_sources()
    # Snapshot bytes once so archive and manifest describe exactly the same contents.
    contents = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sources}
    for name, data in contents.items():
        if re.search(rb'(?m)^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s*$', data):
            raise RuntimeError('Private key marker found; no archive written: ' + name)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    output.mkdir(parents=True)
    archive = output / 'AICOS-ThirdUni-source-review.zip'
    manifest = ''.join(value + '  ' + name + '\n' for name, value in hashes.items())
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for name, data in contents.items():
            bundle.writestr(name, data)
        bundle.writestr('SOURCE-REVIEW-SHA256SUMS.txt', manifest)
        bundle.writestr('PUBLICATION-REVIEW-REQUIRED.txt',
                        'LOCAL REVIEW CANDIDATE ONLY. Not published.\n'
                        'Check ownership/licenses, sample configuration and attribution before publication.\n'
                        'Runtime, databases, credentials, environments, models and videos are excluded.\n'
                        'Portable hosted-demo bootstrap and final owner contribution information remain pending.\n')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
        for name, expected in hashes.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise RuntimeError('Archive manifest mismatch: ' + name)
    result = {'files': len(contents), 'archive': archive.name,
              'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'archive_integrity_verified': True, 'published': False,
              'publication_review_required': True,
              'excluded': ['runtime', '.runtime', 'data', 'models', 'credentials', 'databases',
                           'environment folders', 'videos', 'private keys']}
    (output / 'review-report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
