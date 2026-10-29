"""Prepare attributed public-demo candidate media from one frozen offline replay.

No backend is contacted and no personal video, credentials or database is copied.
"""
import hashlib
import html
import json
from pathlib import Path
import shutil
import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / '.runtime/public-demo-media-20261025/frozen-replay/report.json'
SOURCES = ROOT / 'data/public_eval/assist_iot'
OUTPUT = ROOT / 'submission/public-demo-media'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise RuntimeError('Preserving existing public media candidate directory')
    report = json.loads(REPORT.read_text())
    attribution = json.loads((SOURCES / 'ATTRIBUTION.json').read_text(encoding='utf-8'))
    expected = {Path(item['local_file']).name: item['sha256'] for item in attribution['files']}
    rows = []
    for record in report['reports']:
        video = Path(record['video'])
        if video.parent != SOURCES or digest(video) != record['video_sha256'] or digest(video) != expected[video.name]:
            raise RuntimeError('Source mismatch; no output prepared')
        if record['status'] != 'complete':
            raise RuntimeError('Incomplete source replay')
        rows.append(record)
    positive = [r for r in rows if '__fall__' in Path(r['video']).name]
    negatives = [r for r in rows if '__adl__' in Path(r['video']).name]
    selected = next(r for r in rows if Path(r['video']).name == 'videos__fall__03_lateral_fall_walking.mp4')
    if len(selected['events']) != 1:
        raise RuntimeError('Expected selected source replay event is unavailable')
    OUTPUT.mkdir(parents=True)
    copied = {}
    for record in rows:
        source = Path(record['video'])
        target = OUTPUT / source.name
        shutil.copyfile(source, target)
        if digest(target) != record['video_sha256']:
            raise RuntimeError('Copied source integrity failed')
        copied[target.name] = digest(target)
    for name in ('LICENSE', 'README.md', 'notes.md', 'ATTRIBUTION.json'):
        shutil.copyfile(SOURCES / name, OUTPUT / ('SOURCE-' + name))
    alert = selected['events'][0]
    cap = cv2.VideoCapture(selected['video'])
    index, poster, frames = 0, None, []
    next_sample = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if index == alert['frame']:
            poster = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if pts >= next_sample and pts <= alert['seconds'] + 2:
            picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            picture.thumbnail((640, 400))
            banner = Image.new('RGB', (picture.width, picture.height + 28), '#071014')
            banner.paste(picture, (0, 28))
            text = f"CONTROLLED REPLAY {pts:.1f}s | " + ('PERSON-DOWN ALERT' if pts >= alert['seconds'] else 'Recorded context')
            ImageDraw.Draw(banner).text((8, 7), text, fill='white')
            frames.append(banner)
            next_sample = pts + .5
        index += 1
    cap.release()
    if index != selected['decoded_frames'] or poster is None or not frames:
        raise RuntimeError('Media decode mismatch; prepared directory requires inspection')
    poster.thumbnail((960, 600))
    poster.save(OUTPUT / 'person-down-trigger.jpg', quality=90)
    frames[0].save(OUTPUT / 'person-down-context.gif', save_all=True, append_images=frames[1:],
                   duration=500, loop=0)
    Image.open(OUTPUT / 'person-down-trigger.jpg').verify()
    with Image.open(OUTPUT / 'person-down-context.gif') as image:
        if image.n_frames != len(frames):
            raise RuntimeError('Generated context frame count mismatch')
    result = {'source_report_sha256': digest(REPORT), 'profile_sha256': report['profile_sha256'],
              'model_sha256': report['model_sha256'], 'sampling_fps': report['sample_fps'],
              'selected_demo_source': Path(selected['video']).name,
              'selected_after_testing': True, 'selected_event': alert,
              'positive_examples': len(positive), 'positive_examples_with_alert': sum(bool(r['events']) for r in positive),
              'positive_examples_without_alert': [Path(r['video']).name for r in positive if not r['events']],
              'ordinary_activity_examples': len(negatives), 'ordinary_examples_with_alert': sum(bool(r['events']) for r in negatives),
              'copied_source_sha256': copied, 'personal_recording_included': False, 'published': False,
              'derived_media_changes': 'JPEG extraction; resized silent GIF with replay/status captions at nominal 2 FPS. Original MP4 sources unchanged.',
              'limits': ['Five short reference clips only; no population accuracy estimate.',
                         'Controlled laboratory simulations with two views composed into one frame; not outdoor CCTV.',
                         'Alert location is in the inset view; the same participant is also shown in the main view.',
                         'Vertical-collapse example missed by frozen profile; failure retained.',
                         'No stable upright transition was observed by the detector; no sudden-collapse or medical-cause claim.',
                         'This gallery is media review, not the deployed interactive AICOS backend.']}
    (OUTPUT / 'MEDIA-REVIEW.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    credits = ('Multimodal fall detection dataset — Anastasiya Danilenka, Piotr Sowiński, Monika Kobus, '
               'Anna Dąbrowska, Kajetan Rachwał, Karolina Bogacka, Krzysztof Baszczyński. '
               'ASSIST-IoT, EU Horizon 2020 grant 957258; CIOP-PIB facilities/equipment. '
               'CC BY 4.0. Not endorsed by the dataset authors.')
    parts = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
             '<title>AICOS public-media review candidate</title><style>body{max-width:1000px;margin:30px auto;padding:20px;font:17px system-ui;background:#071014;color:#e7f4f3}video,img{max-width:100%}a{color:#69e6c5}article{border-top:1px solid #31545c;padding:16px 0}</style>',
             '<h1>AICOS — licensed replay media review</h1><p>Local preparation only. This gallery is not the interactive AICOS product.</p>',
             '<p>Frozen profile: one of two controlled fall examples alerted; the vertical example was missed. No alerts on three short ordinary activities. This is not an accuracy benchmark.</p>',
             '<p>Selected lateral replay: alert at 8.767 seconds; transition not established. Sources contain a main view plus an inset of the same activity.</p>',
             '<img src="person-down-trigger.jpg" alt="Triggering frame from the controlled lateral-fall simulation">']
    for row in rows:
        name = Path(row['video']).name
        parts.append('<article><h2>' + html.escape(name) + '</h2><p>Frozen replay alerts: ' + str(len(row['events'])) + '</p><video controls preload="metadata" src="' + name + '"></video></article>')
    parts.extend(['<p>' + html.escape(credits) + '</p>',
                  '<p><a href="https://github.com/ASSIST-IoT-SRIPAS/multimodal-fall-detection-dataset">Original dataset</a> · <a href="https://creativecommons.org/licenses/by/4.0/">License</a> · <a href="MEDIA-REVIEW.json">Full local review record</a></p>'])
    (OUTPUT / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'positive_alerts': result['positive_examples_with_alert'],
                      'positive_examples': len(positive), 'negative_alerts': result['ordinary_examples_with_alert'],
                      'ordinary_examples': len(negatives), 'selected_alert_seconds': alert['seconds'],
                      'personal_footage_included': False, 'published': False}))


if __name__ == '__main__':
    main()
