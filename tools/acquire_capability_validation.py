"""Bounded acquisition of documented educational evaluation data; no model downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import ssl
import shutil
import io
import zipfile
import urllib.request
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.runtime/capability-validation-20261027'
DATA = ROOT / 'data/capability_validation_20261027'
REPO = 'airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos'


def context():
    ctx = ssl.create_default_context()
    # Windows' installed trusted roots, without modifying trust or disabling TLS.
    if hasattr(ssl, 'enum_certificates'):
        for cert, encoding, trust in ssl.enum_certificates('ROOT'):
            if encoding == 'x509_asn' and (trust is True or '1.3.6.1.5.5.7.3.1' in trust):
                ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(cert))
    return ctx


CTX = context()


def get(url, maximum=2_000_000, headers=None):
    request = urllib.request.Request(url, headers={'User-Agent': 'AICOS-ThirdUni-local-evaluation/1.0', **(headers or {})})
    with urllib.request.urlopen(request, context=CTX, timeout=45) as response:
        data = response.read(maximum + 1)
        if len(data) > maximum:
            raise RuntimeError('Response exceeded the declared acquisition bound')
        return data, dict(response.headers), response.status


def metadata():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in [('firesense-metadata.json', 'https://zenodo.org/api/records/836749'),
                      ('airtlab-tree.json', f'https://api.github.com/repos/{REPO}/git/trees/master?recursive=1')]:
        target = OUT / name
        if target.exists():
            continue
        data, _, _ = get(url)
        value = json.loads(data)
        target.write_text(json.dumps(value, indent=2), encoding='utf-8')
    zen = json.loads((OUT / 'firesense-metadata.json').read_text())
    tree = json.loads((OUT / 'airtlab-tree.json').read_text())
    print(json.dumps({'firesense_license': zen['metadata'].get('license'),
        'firesense_files': [{'key': f['key'], 'size': f['size'], 'checksum': f['checksum'], 'links': f['links']} for f in zen['files']],
        'airtlab_revision': tree['sha'], 'airtlab_examples': [x for x in tree['tree'] if x['path'].endswith('.mp4')][:3]}, indent=2), flush=True)


def airtlab():
    tree = json.loads((OUT / 'airtlab-tree.json').read_text())
    # First six numeric scenarios from cam1 in each class, selected BEFORE inference.
    # cam2 is the SAME activity, so it is not counted as independent evidence.
    chosen = []
    for label in ('violent', 'non-violent'):
        for number in range(1, 7):
            path = f'violence-detection-dataset/{label}/cam1/{number}.mp4'
            item = next(x for x in tree['tree'] if x['path'] == path)
            chosen.append({**item, 'label': label})
    total = sum(x['size'] for x in chosen)
    if total > 100_000_000:
        raise RuntimeError('Selected AIRTLab subset exceeds its 100 MB budget')
    plan = {'dataset': REPO, 'revision': tree['sha'], 'selection': 'cam1 scenarios 1Ã¢â‚¬â€œ6 per publisher class; fixed before predictions',
            'license': 'Publisher permits research and educational use; no public redistribution here',
            'source_page': 'https://github.com/' + REPO, 'files': chosen, 'bytes': total,
            'training_overlap_with_third_party_model': 'unknown; not an independent benchmark claim'}
    plan_path = OUT / 'airtlab-predeclared.json'
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    folder = DATA / 'airtlab'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in chosen:
        url = f"https://raw.githubusercontent.com/{REPO}/{tree['sha']}/{item['path']}"
        target = folder / (item['label'] + '-' + Path(item['path']).name)
        data = target.read_bytes() if target.exists() else get(url, item['size'])[0]
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != item['size'] or actual != item['sha']:
            raise RuntimeError('Pinned Git blob integrity mismatch: ' + item['path'])
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(data)
        rows.append({'file': str(target.relative_to(ROOT)), 'label': item['label'], 'bytes': len(data),
                     'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_sha1': actual, 'url': url})
        print(target.name, len(data), 'verified', flush=True)
    (folder / 'ATTRIBUTION.json').write_text(json.dumps({**plan, 'files': rows}, indent=2), encoding='utf-8')


def firesense():
    meta = json.loads((OUT / 'firesense-metadata.json').read_text())
    if meta['metadata'].get('license', {}).get('id') != 'cc-by-4.0':
        raise RuntimeError('Unexpected dataset license; inspect before acquisition')
    folder = DATA / 'firesense'
    folder.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < 3_000_000_000:
        raise RuntimeError('Less than 3 GB free; refusing the bounded archive download')
    for item in meta['files']:
        name, size = item['key'], item['size']
        if name not in ('fire_videos.1406.zip', 'smoke_videos.1407.zip') or size > 650_000_000:
            raise RuntimeError('Unexpected archive or archive size')
        target = folder / name
        if target.exists():
            actual = hashlib.md5(target.read_bytes()).hexdigest()
            if target.stat().st_size != size or 'md5:' + actual != item['checksum']:
                raise RuntimeError('Existing archive does not match publisher checksum; preserved for inspection')
            print(name, 'already verified', flush=True)
            continue
        partial = target.with_suffix('.zip.partial')
        if partial.exists():
            raise RuntimeError('Partial archive exists; inspect before deciding a recovery action')
        request = urllib.request.Request(item['links']['self'], headers={'User-Agent': 'AICOS-ThirdUni-local-evaluation/1.0'})
        count, next_progress, digest = 0, 32_000_000, hashlib.md5()
        with urllib.request.urlopen(request, context=CTX, timeout=45) as response, partial.open('xb') as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > size:
                    raise RuntimeError('Archive exceeded publisher size')
                stream.write(chunk)
                digest.update(chunk)
                if count >= next_progress:
                    print(name, count, '/', size, 'bytes', flush=True)
                    next_progress += 32_000_000
        if count != size or 'md5:' + digest.hexdigest() != item['checksum']:
            raise RuntimeError('Downloaded archive integrity mismatch')
        partial.rename(target)
        print(name, 'publisher MD5 verified', flush=True)
    (folder / 'ATTRIBUTION.json').write_text(json.dumps({'source_page': 'https://zenodo.org/records/836749',
        'doi': '10.5281/zenodo.836749', 'license': 'CC-BY-4.0',
        'authors': ['Nikos Grammalidis', 'Kosmas Dimitropoulos', 'Enis Cetin'],
        'funding': 'European Commission FIRESENSE FP7-ENV-244088',
        'purpose': 'Local evaluation only; no redistribution performed',
        'files': meta['files']}, indent=2), encoding='utf-8')


class RemoteArchive(io.RawIOBase):
    """Read exact bounded ranges; never silently accept a full or truncated response."""
    def __init__(self, item):
        self.item, self.position, self.transferred = item, 0, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.item['size'] + offset
        if not 0 <= self.position <= self.item['size']:
            raise ValueError('Archive seek outside declared bounds')
        return self.position

    def read(self, size=-1):
        size = min(self.item['size'] - self.position, size if size >= 0 else self.item['size'])
        if size > 55_000_000:
            raise RuntimeError('Archive read exceeds 55 MB component limit')
        parts = []
        while size:
            count = min(size, 1_000_000)
            end = self.position + count - 1
            expected = f"bytes {self.position}-{end}/{self.item['size']}"
            # Some intermediaries cache an earlier range response under the same
            # URL. A unique query is safe only because every returned range and
            # byte count is still validated before updating our position.
            for attempt in range(3):
                url = self.item['links']['self'] + f'?aicos_range={self.position}-{end}&attempt={time.time_ns()}-{attempt}'
                try:
                    data, headers, status = get(url, count, {
                        'Range': f'bytes={self.position}-{end}',
                        'Cache-Control': 'no-cache', 'Accept-Encoding': 'identity'})
                    received = next((v for k, v in headers.items() if k.lower() == 'content-range'), None)
                    if status != 206 or received != expected or len(data) != count:
                        raise RuntimeError(f'Server range mismatch: status={status}, expected={expected}, received={received}, length={len(data)}, wanted={count}')
                    break
                except (OSError, RuntimeError) as exc:
                    if attempt == 2:
                        raise
                    print(f'Range retry {attempt + 1}/2: {self.item["key"]} {expected}: {exc}', flush=True)
                    time.sleep(attempt + 1)
            self.position += count
            self.transferred += count
            size -= count
            parts.append(data)
        return b''.join(parts)


def firesense_index():
    meta = json.loads((OUT / 'firesense-metadata.json').read_text())
    rows = []
    for item in meta['files']:
        with zipfile.ZipFile(RemoteArchive(item)) as archive:
            rows.extend({'archive': item['key'], 'name': info.filename, 'bytes': info.file_size,
                         'compressed': info.compress_size, 'crc32': info.CRC}
                        for info in archive.infolist() if not info.is_dir())
    (OUT / 'firesense-archive-index.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2), flush=True)


def firesense_subset():
    meta = json.loads((OUT / 'firesense-metadata.json').read_text())
    rows = json.loads((OUT / 'firesense-archive-index.json').read_text())
    chosen = []
    for family in ('fire', 'smoke'):
        for label in ('positive', 'negative'):
            candidates = [r for r in rows if r['archive'].startswith(family + '_') and
                          r['name'].lower().startswith(('pos/' if label == 'positive' else 'neg/')) and Path(r['name']).suffix.lower() in ('.avi', '.mpg', '.mpeg', '.mp4', '.wmv')]
            candidates.sort(key=lambda r: (r['compressed'], r['name'].casefold()))
            if len(candidates) < 4:
                raise RuntimeError('Inspect publisher folders before assigning labels: ' + family + '/' + label)
            for row in candidates[:4]:
                chosen.append({**row, 'family': family, 'publisher_label': label})
    if sum(r['compressed'] for r in chosen) > 300_000_000 or any(r['bytes'] > 150_000_000 or r['compressed'] > 50_000_000 for r in chosen):
        raise RuntimeError('Predeclared 16-video subset exceeds its size bound; no videos downloaded')
    plan = {'selection': 'Four smallest compressed video members per publisher positive/negative family; fixed before inference; size-biased bounded screening subset',
            'files': chosen, 'compressed_bytes': sum(r['compressed'] for r in chosen), 'license': 'CC-BY-4.0',
            'label_scope': 'Flame-family negatives label flame only; smoke-family negatives label smoke only. Other hazard class is unlabelled.',
            'training_overlap': 'unknown; external-reference evaluation, not proven independent holdout'}
    target_plan = OUT / 'firesense-predeclared.json'
    if not target_plan.exists():
        target_plan.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    folder = DATA / 'firesense/subset'
    folder.mkdir(parents=True, exist_ok=True)
    verified = []
    for item in meta['files']:
        with zipfile.ZipFile(RemoteArchive(item)) as archive:
            for row in chosen:
                if row['archive'] != item['key']:
                    continue
                name = row['family'] + '-' + row['publisher_label'] + '-' + Path(row['name']).name
                target = folder / name
                if target.exists():
                    data = target.read_bytes()
                else:
                    data = archive.read(row['name'])  # zipfile also checks member CRC.
                import zlib
                if len(data) != row['bytes'] or zlib.crc32(data) != row['crc32']:
                    raise RuntimeError('Selected archive member integrity mismatch')
                if not target.exists():
                    with target.open('xb') as stream:
                        stream.write(data)
                verified.append({**row, 'file': target.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(data).hexdigest()})
                print(name, len(data), 'member CRC verified', flush=True)
    (folder / 'ATTRIBUTION.json').write_text(json.dumps({**plan, 'files': verified,
        'source_page': 'https://zenodo.org/records/836749', 'authors': ['Nikos Grammalidis','Kosmas Dimitropoulos','Enis Cetin'],
        'integrity_scope': 'Verified TLS, exact HTTP byte ranges, ZIP member sizes/CRC and local SHA256; entire archive MD5 not verified',
        'purpose': 'Local educational evaluation only; no public redistribution'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['metadata', 'airtlab', 'firesense', 'firesense-index', 'firesense-subset'])
    args = parser.parse_args()
    {'metadata': metadata, 'airtlab': airtlab, 'firesense': firesense,
     'firesense-index': firesense_index, 'firesense-subset': firesense_subset}[args.action]()
