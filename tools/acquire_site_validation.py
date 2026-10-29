"""Acquire a bounded, predeclared official-source site-rule evaluation subset.

No models, accounts, production configuration or external application are changed.
Original bytes and acquisition receipts remain local. Local hashes establish
repeatability, not a publisher signature. Interrupted files are never trusted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from acquire_capability_validation import CTX

DATA = ROOT / 'data/capability_validation_20261027/site'
WORK = ROOT / '.runtime/capability-validation-20261027/site-validation'
CAVIAR = 'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/'
FILES = [
    ('mot17-labels.zip', 'https://motchallenge.net/data/MOT17Labels.zip', 15_000_000, 'MOTChallenge', 'publisher pedestrian boxes; not incident labels'),
    ('MOT17-04.mp4', 'https://motchallenge.net/sequenceVideos/MOT17-04-SDP-raw.mp4', 60_000_000, 'MOTChallenge', 'stationary elevated night crowd; publisher annotated boxes'),
    ('MOT17-09.mp4', 'https://motchallenge.net/sequenceVideos/MOT17-09-SDP-raw.mp4', 60_000_000, 'MOTChallenge', 'pedestrian street; publisher annotated boxes'),
]
for folder, annotation in [('Walk1', 'wk1gt.xml'), ('LeftBag', 'lb1gt.xml'),
                           ('Meet_Crowd', 'mc1gt.xml'), ('Fight_RunAway1', 'fra1gt.xml')]:
    FILES.extend([(folder + '.mpg', CAVIAR + folder + '/' + folder + '.mpg', 25_000_000, 'CAVIAR', 'acted publisher scenario ' + folder),
                  (annotation, CAVIAR + folder + '/' + annotation, 8_000_000, 'CAVIAR', 'publisher frame annotations')])
TRAFFIC_FILES = []
for sequence in ('sherbrooke', 'rouen'):
    base = 'https://www.jpjodoin.com/urbantracker/dataset/' + sequence + '/' + sequence
    TRAFFIC_FILES.extend([(sequence + '.avi', base + '_video.avi', 100_000_000, 'UrbanTracker', 'publisher annotated fixed-camera mixed traffic'),
                          (sequence + '-annotations.zip', base + '_annotations.zip', 12_000_000, 'UrbanTracker', 'publisher object trajectories; no incident labels')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traffic-only', action='store_true')
    args = parser.parse_args()
    selected = TRAFFIC_FILES if args.traffic_only else FILES
    DATA.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    plan = {'selection_before_inference': True, 'files': selected,
            'sources': {'MOTChallenge': {'page': 'https://motchallenge.net/data/MOT17/',
                        'terms': 'CC BY-NC-SA 3.0; original sequence terms may also apply; local educational evaluation only; no redistribution'},
                        'CAVIAR': {'page': CAVIAR, 'terms': 'Publisher Creative Commons BY-SA; acknowledge EC CAVIAR IST 2001 37540; no publication in this task'},
                        'UrbanTracker': {'page': 'https://www.jpjodoin.com/urbantracker/dataset.html',
                        'terms': 'Publisher downloadable research evaluation data; cite Jodoin, Bilodeau, Saunier WACV14; no redistribution or broad license inferred'}},
            'scope': 'Bounded development screening; unknown training overlap; no field-accuracy claim. Crowd/zone/duration are explicit test configurations, not publisher incident labels.'}
    plan_path = WORK / ('TRAFFIC-DATA-PLAN.json' if args.traffic_only else 'DATA-PLAN.json')
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    receipts_path = WORK / ('traffic-acquisition.json' if args.traffic_only else 'acquisition.json')
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    for name, url, limit, source, purpose in selected:
        target = DATA / name
        prior = receipts.get(name, {})
        if target.exists():
            value = target.read_bytes()
            if prior.get('sha256') != hashlib.sha256(value).hexdigest():
                raise RuntimeError('Unreceipted or changed input preserved: ' + name)
            print(name, 'already verified', flush=True)
            continue
        errors = []
        for attempt in range(3):
            partial = DATA / (name + '.attempt-' + str(attempt + 1) + '.partial')
            if partial.exists():
                errors.append('Historical partial preserved: ' + partial.name)
                continue
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'AICOS-local-educational-evaluation/1.0', 'Accept-Encoding': 'identity'})
                count = 0
                with urllib.request.urlopen(request, context=CTX, timeout=30) as response, partial.open('xb') as stream:
                    headers = dict(response.headers)
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > limit:
                            raise ValueError('Declared byte budget exceeded')
                        stream.write(block)
                if count < 50 or (headers.get('Content-Length') and int(headers['Content-Length']) != count):
                    raise ValueError('Empty or truncated input')
                value = partial.read_bytes()
                if name.endswith('.zip'):
                    with zipfile.ZipFile(partial) as archive:
                        if archive.testzip() is not None:
                            raise ValueError('ZIP CRC failed')
                elif name.endswith('.xml'):
                    import xml.etree.ElementTree as ET
                    ET.fromstring(value)
                elif 'html' in headers.get('Content-Type', '').lower() or value.lstrip().lower().startswith(b'<!doctype html'):
                    raise ValueError('HTML instead of video')
                partial.rename(target)
                receipts[name] = {'url': url, 'file': str(target.relative_to(ROOT)), 'source': source,
                                  'purpose': purpose, 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest(),
                                  'headers': headers, 'prior_attempt_errors': errors, 'status': 'acquired'}
                print(name, len(value), 'acquired', flush=True)
                break
            except Exception as exc:
                errors.append(type(exc).__name__ + ': ' + str(exc))
                print(name, 'attempt', attempt + 1, errors[-1], flush=True)
                time.sleep(1)
        else:
            receipts[name] = {'url': url, 'status': 'acquisition_failed', 'errors': errors}
        receipts_path.write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    return 0 if all(receipts.get(row[0], {}).get('status') == 'acquired' for row in selected) else 1


if __name__ == '__main__':
    sys.exit(main())
