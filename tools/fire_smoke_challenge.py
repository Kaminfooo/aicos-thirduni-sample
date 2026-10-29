"""Fixed-model cross-source fire/smoke challenge; never changes live settings."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.runtime/fire-smoke-challenge-20261028'
DATA=ROOT/'data/fire_smoke_challenge_20261028'
COHORT='all'
sys.path.insert(0,str(ROOT/'tools'))
from acquire_capability_validation import CTX


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable(path,value):
    value=json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text())!=value: raise ValueError('Existing record differs: '+str(path))
    else: path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def flame_labels(path):
    import xml.etree.ElementTree as ET
    labels={}
    for entry in ET.parse(path).getroot().findall('./frames/_'):
        index=int(entry.findtext('frameNumber'))
        if index<0 or index in labels: raise ValueError('Invalid or duplicate annotation index')
        positive=False
        for rectangle in entry.findall('./annotations/_'):
            values=list(map(float,(rectangle.text or '').split()))
            if len(values)!=4 or not all(math.isfinite(v) for v in values):raise ValueError('Malformed annotated rectangle')
            positive=positive or values[2]>0 and values[3]>0
        labels[index]=positive
    if not labels:raise ValueError('No publisher frame annotations')
    return labels


def download(name,url,limit,blob=None):
    target=DATA/name;receipt=WORK/(name+'.receipt.json')
    if target.exists():
        r=json.loads(receipt.read_text())
        if sha(target)!=r['sha256']: raise ValueError('Existing content changed')
        return target
    errors=[]
    for attempt in range(1,4):
        partial=DATA/(name+'.attempt'+str(attempt)+'.partial')
        if partial.exists(): continue
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'AICOS-local-research-evaluation/1.0','Accept-Encoding':'identity'})
            count=0;begun=time.monotonic()
            with urllib.request.urlopen(request,context=CTX,timeout=35) as response,partial.open('xb') as stream:
                headers=dict(response.headers)
                while True:
                    if time.monotonic()-begun>240: raise TimeoutError('Four-minute download bound')
                    # Return available bytes rather than waiting to fill a large
                    # buffer, so the total-time bound is checked during slow sends.
                    block=response.read1(64*1024)
                    if not block: break
                    count+=len(block)
                    if count>limit: raise ValueError('Byte bound exceeded')
                    stream.write(block)
            content=partial.read_bytes()
            if len(content)<20 or content.lstrip().lower().startswith((b'<!doctype html',b'<html')): raise ValueError('Not expected media')
            if headers.get('Content-Length') and int(headers['Content-Length'])!=len(content): raise ValueError('Truncated content')
            if blob and hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()!=blob: raise ValueError('Pinned Git blob mismatch')
            if name.endswith('.zip'):
                with zipfile.ZipFile(partial) as z:
                    if z.testzip(): raise ValueError('ZIP member CRC failed')
            partial.rename(target)
            immutable(receipt,{'url':url,'bytes':len(content),'sha256':sha(target),'git_blob':blob,'headers':headers,'errors':errors})
            print('acquired',name,len(content),flush=True);return target
        except Exception as exc:
            errors.append(type(exc).__name__+': '+str(exc));print(name,errors[-1],flush=True)
    raise RuntimeError('Acquisition incomplete; partials preserved: '+name+' '+str(errors))


def acquire():
    DATA.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    tree=json.loads((WORK/'furg-tree.json').read_text());revision=tree['sha']
    chosen=['house6.mp4','house6.xml','hand_held_batteries_night.mp4','hand_held_batteries_night.xml',
            'case2_car.mp4','case2_car.xml','non_fire_patrolbot_onboard.mp4','non_fire_patrolbot_onboard.xml','LICENSE','README.md']
    archives=[('kmu-short-smoke.zip','Indoor_outdoor%28short_distance%29smoke.zip','smoke_positive'),
              ('kmu-wildfire-smoke.zip','Wildfire_smoke.zip','smoke_positive'),
              ('kmu-confusions.zip','Smoke_flame_like_moving_object.zip','confusion_negative')]
    immutable(WORK/'ACQUISITION-PLAN.json',{'fixed_before_inference':True,'furg_revision':revision,'furg_files':chosen,'kmu_archives':archives,
        'sources':{'FURG':'https://github.com/steffensbola/furg-fire-dataset','KMU':'https://cvpr.kmu.ac.kr/Dataset/Dataset.htm'},
        'terms':{'FURG':'Publisher CC0 repository; source/license retained.','KMU':'Copyright KMU CVPR Lab; academic research download only; no redistribution.'},
        'selection':'FURG fixed house, night battery, car flame and non-fire robot scenarios; every video in three KMU archives. No outcome-based exclusion.',
        'training_overlap':'unknown; new-to-project cross-source screening is not proof of training independence'})
    for name in chosen:
        item=next(x for x in tree['tree'] if x['path']==name)
        download('furg-'+name,'https://raw.githubusercontent.com/steffensbola/furg-fire-dataset/'+revision+'/'+name,item['size'],item['sha'])
    members=[]
    for name,remote,family in archives:
        path=download(name,'https://cvpr.kmu.ac.kr/Dataset/'+remote,70_000_000)
        with zipfile.ZipFile(path) as z:
            for info in sorted(z.infolist(),key=lambda i:i.filename):
                if info.is_dir() or Path(info.filename).suffix.lower() not in ('.avi','.mp4','.mpg','.mpeg','.mov','.wmv'): continue
                if info.file_size>150_000_000: raise ValueError('Unexpected uncompressed component size')
                target=DATA/(Path(name).stem+'--'+Path(info.filename).name)
                content=z.read(info)
                if target.exists() and target.read_bytes()!=content: raise ValueError('Changed extracted member')
                if not target.exists(): target.write_bytes(content)
                members.append({'file':str(target.relative_to(ROOT)),'archive':name,'member':info.filename,'crc32':info.CRC,'sha256':sha(target),'family':family})
    immutable(WORK/'KMU-MEMBERS.json',members)
    print('All acquisition complete:',len(members),'KMU videos +4 FURG videos.',flush=True)


def prepare():
    members=[] if COHORT=='furg' else json.loads((ROOT/'.runtime/fire-smoke-challenge-20261028/KMU-MEMBERS.json').read_text())
    rows=[]
    for name in (() if COHORT=='kmu' else ('house6','hand_held_batteries_night','case2_car','non_fire_patrolbot_onboard')):
        path=DATA/('furg-'+name+'.mp4')
        labels=flame_labels(DATA/('furg-'+name+'.xml'))
        if any(labels.values()) != (name!='non_fire_patrolbot_onboard'):raise ValueError('Publisher annotation conflicts with planned FURG label: '+name)
        rows.append({'file':str(path.relative_to(ROOT)),'sha256':sha(path),'labels':{'fire':name!='non_fire_patrolbot_onboard'},
            'group':'FURG','annotation':str((DATA/('furg-'+name+'.xml')).relative_to(ROOT))})
    for row in members:
        rows.append({**row,'labels':{'smoke':True} if row['family']=='smoke_positive' else {'fire':False,'smoke':False},'group':'KMU'})
    # These existing videos are new to this detector screen; source attribution
    # and previous purposes remain visible rather than claiming new acquisition.
    for name in (() if COHORT=='kmu' else ('site/sherbrooke.avi','site/rouen.avi')):
        path=ROOT/'data/capability_validation_20261027'/name
        rows.append({'file':str(path.relative_to(ROOT)),'sha256':sha(path),'labels':{'fire':False,'smoke':False},'group':'UrbanTracker ordinary traffic control',
            'note':'Previously acquired traffic video, unseen by fire model; review montage before scoring negative; not newly acquired dataset.'})
    base=json.loads((ROOT/'.runtime/capability-validation-20261027/fire-followup/evaluation/VALIDATION-PLAN.json').read_text())
    for name,digest in base['source_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Frozen detector source changed')
    immutable(WORK/'PREDECLARED.json',{'cases':rows,'frozen_settings':base['frozen_hazard_settings'],'sampling_fps':3,'resize_width':480,
        'model_sha256':sha(ROOT/'models/pretrained/fire-smoke-yolo26n.pt'),'no_threshold_or_model_changes':True,
        'acceptance':'Every scored positive alerts labelled class; no scored negative alerts labelled class. Other classes unlabelled, not inferred negative.',
        'onset_policy':'No onset latency without verified frame annotation alignment; alert timestamp is not onset latency.',
        'duration':'Score whole decoded videos, original encoded PTS; no repeated/extended/frozen frames to inflate exposure.',
        'selection_before_inference':True})
    immutable(WORK/'VALIDATION-PLAN.json',base)
    print('Fixed',len(rows),'videos before any challenge inference.')


def evaluate():
    plan=json.loads((WORK/'PREDECLARED.json').read_text())
    if sha(ROOT/'models/pretrained/fire-smoke-yolo26n.pt')!=plan['model_sha256']: raise ValueError('Model changed')
    spec=importlib.util.spec_from_file_location('frozen_fire_evaluator',ROOT/'tools/run_capability_validation.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator);evaluator.WORK=WORK
    reviews={r['file']:r for r in json.loads((WORK/'DECODE-REVIEW.json').read_text())}
    exclusions=[r for r in plan['cases'] if not reviews[r['file']]['timestamps_valid']]
    immutable(WORK/'UNSCORABLE-TIMELINES.json',exclusions)
    evaluator.cases=lambda _: [{'path':ROOT/r['file'],'label':next(iter(r['labels'])),'positive':next(iter(r['labels'].values())),
        'source_sha256':r['sha256'],'source_group':r['group']} for r in plan['cases'] if r not in exclusions]
    # Base evaluator's provisional single-class result is retained, then the
    # final report below scores all explicitly labelled classes from the same run.
    return evaluator.evaluate('fire_smoke')


def review():
    import cv2
    from PIL import Image,ImageDraw
    plan=json.loads((WORK/'PREDECLARED.json').read_text());records=[];tiles=[]
    for case in plan['cases']:
        path=ROOT/case['file'];cap=cv2.VideoCapture(str(path));count=0;last=-1;selected=[];onset=None;labels={};timeline_errors=[]
        if not cap.isOpened():raise ValueError('Cannot decode review video '+str(path))
        if case.get('annotation'):
            labels=flame_labels(ROOT/case['annotation'])
        next_tile=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            stamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if not math.isfinite(stamp) or stamp<last:timeline_errors.append({'frame0':count,'previous':last,'current':stamp})
            if labels.get(count) and onset is None:
                onset={'first_labelled_positive_frame0':count,'first_labelled_positive_pts':stamp,
                    'already_positive_at_clip_start':count==0,'preceding_frame_explicitly_negative':labels.get(count-1) is False,
                    'onset_interval_pts':[last,stamp] if count>0 and labels.get(count-1) is False else None}
            if stamp>=next_tile:
                pic=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));pic.thumbnail((200,130))
                tile=Image.new('RGB',(200,160),'white');tile.paste(pic,(0,25));draw=ImageDraw.Draw(tile)
                draw.text((2,2),path.name[:29],fill='black');draw.text((2,145),str(round(stamp,2))+' seconds',fill='black');selected.append((stamp,tile));next_tile=stamp+15
            last=stamp;count+=1
        cap.release()
        # Every15s is a coarse scene review, not an exhaustive hazard label.
        tiles.extend(tile for _,tile in selected)
        records.append({'file':case['file'],'decoded_frames':count,'last_pts':last,'review_pts':[t for t,_ in selected],
            'timestamps_valid':not timeline_errors,'timeline_errors':timeline_errors[:20],
            'annotation_frames':len(labels),'annotation_frame_count_matches':set(labels)==set(range(count)) if labels else None,
            'onset':onset,'negative_annotation_has_any_flame':any(labels.values()) if labels and case['labels'].get('fire') is False else None})
    for page,start in enumerate(range(0,len(tiles),30)):
        batch=tiles[start:start+30];canvas=Image.new('RGB',(1000,160*((len(batch)+4)//5)),'white')
        for i,tile in enumerate(batch):canvas.paste(tile,((i%5)*200,(i//5)*160))
        canvas.save(WORK/('review-'+str(page+1)+'.jpg'))
    immutable(WORK/'DECODE-REVIEW.json',records)
    print(json.dumps(records,indent=2))


def summarize():
    plan=json.loads((WORK/'PREDECLARED.json').read_text());raw=json.loads((WORK/'fire_smoke-frozen-screen/report.json').read_text())
    reviews={r['file']:r for r in json.loads((WORK/'DECODE-REVIEW.json').read_text())};cases=[]
    for case in plan['cases']:
        row=next((r for r in raw['cases'] if Path(r['file'])==Path(case['file'])),None)
        if row is None:
            unscorable=not reviews[case['file']]['timestamps_valid']
            for label,positive in case['labels'].items():cases.append({'file':case['file'],'source_group':case['group'],'label':label,'positive':positive,
                'status':'unscorable_timestamps' if unscorable else 'not_run','outcome':'ERROR' if unscorable else 'UNTESTED','event_count':None,'first_alert_pts':None,'decoded_seconds':None})
            continue
        for label,positive in case['labels'].items():
            events=[e for e in row['events'] if e['event']['extra']['label']==label];predicted=bool(events)
            result={'file':case['file'],'source_group':case['group'],'label':label,'positive':positive,'status':row['status'],
                'outcome':(('TP' if predicted else 'FN') if positive else ('FP' if predicted else 'TN')) if row['status']=='complete' else 'ERROR',
                'decoded_seconds':row['decoded_seconds'],'event_count':len(events),'first_alert_pts':events[0]['seconds'] if events else None,
                'onset_latency_seconds':None,'onset_note':'No verified onset interval for this class.'}
            reviewrow=reviews[case['file']];onset=reviewrow['onset'] if label=='fire' else None
            if onset:
                result['onset']=onset
                if onset['already_positive_at_clip_start']:result['onset_note']='Already fire-positive at frame0; first alert time is within-clip response, not ignition latency.'
                elif not reviewrow['annotation_frame_count_matches']:result['onset_note']='Annotation/video frame counts differ; onset mapping unverified.'
                elif onset['onset_interval_pts'] and events:
                    start,end=onset['onset_interval_pts'];t=events[0]['seconds'];result['onset_note']='Relative to publisher first visible-flame interval, not physical ignition.'
                    result['onset_delay_interval_seconds']=[t-end,t-start] if t>=end else None
                    result['early_relative_to_annotation']=t<end
            cases.append(result)
    summary={}
    for label in ('fire','smoke'):
        rows=[r for r in cases if r['label']==label];counts={k:sum(r['outcome']==k for r in rows) for k in ('TP','FN','FP','TN','ERROR')}
        if any(r['outcome']=='UNTESTED' for r in rows):counts['UNTESTED']=sum(r['outcome']=='UNTESTED' for r in rows)
        negatives=[r for r in rows if not r['positive'] and r['status']=='complete'];duration=sum(r['decoded_seconds'] for r in negatives)
        summary[label]={**counts,'negative_encoded_seconds':duration,'false_events':sum(r['event_count'] for r in negatives),
            'false_events_per_encoded_negative_hour':sum(r['event_count'] for r in negatives)*3600/duration if duration else None}
    unique={r['file'] for r in cases};passed=len(unique)==len(plan['cases']) and all(r['outcome'] in ('TP','TN') for r in cases)
    result={'unique_videos':len(unique),'requested_videos':len(plan['cases']),'summary':summary,'cases':cases,'screening_passed':passed,
        'thresholds_or_model_changed':False,'training_overlap':'unknown','ordinary_controls':'UrbanTracker negative labels are coarse visual review, not publisher fire annotations.',
        'negative_hours_note':'Per-class exposures overlap on dual-labelled negative videos; do not add them as independent camera-hours.',
        'source_plan_sha256':sha(WORK/'PREDECLARED.json'),'inference_report_sha256':sha(WORK/'fire_smoke-frozen-screen/report.json')}
    immutable(WORK/'FINAL-RESULT.json',result)
    lines=['# Fire/smoke cross-source challenge — 28 October 2026','',
        'Frozen existing model, confidence0.5, two-second persistence,3fps,480-pixel input. No tuning, live activation, package installation or model download. Previous18-clip passing result remains separate.','',
        '| Class | Detected positives | Missed positives | Negatives with alerts | Quiet negatives | Errors |',
        '|---|---:|---:|---:|---:|---:|']
    for label,s in summary.items():lines.append(f"| {label} | {s['TP']} | {s['FN']} | {s['FP']} | {s['TN']} | {s['ERROR']} |")
    lines+=['',f"Unique videos: {len(unique)}/{len(plan['cases'])}. Challenge acceptance: {'PASS' if passed else 'FAIL'}.",'',
        'These are new-to-project source scenarios, not proof of independence from model training. FURG flame labels, KMU smoke/confusion family labels, and coarse reviewed ordinary traffic controls are distinguished in the JSON. Other hazard classes stay unlabelled. Negative exposure overlaps across classes and is not a field false-alarm-rate estimate.','',
        'Only a verified first-visible-flame onset interval can produce a delay interval. Already-positive clips report within-clip alert time only; smoke clip-level labels cannot establish smoke onset. Raw annotations and decode/frame-count checks are preserved.','',
        'Sources: [FURG](https://github.com/steffensbola/furg-fire-dataset), [KMU](https://cvpr.kmu.ac.kr/Dataset/Dataset.htm), [UrbanTracker](https://www.jpjodoin.com/urbantracker/dataset.html). Publisher terms and acquisition receipts retained locally; no media published.','',
        '## Cases','', '| Video | Class | Outcome | First alert (encoded seconds) |','|---|---|---|---|']
    if COHORT=='rise':
        position=lines.index('## Cases')
        lines[position:position]=['## RISE scope and timing','',
            'This cohort uses researcher-confirmed industrial-smoke labels from [CMU RISE](https://github.com/CMU-CREATE-Lab/deep-smoke-machine). Its [primary paper](https://arxiv.org/html/2005.06111) distinguishes smoke from steam and includes steam distractions. Each selected file has36 frames and2.9167 encoded seconds. These are timelapses, not regular camera video. Treat the results as encoded-input transfer failures; do not interpret the computed encoded-hour ratio as a real-time false-alarm rate or the two-second confirmation as physical time. There are no frame-level onset labels.','']
    for r in cases:lines.append(f"| {Path(r['file']).name} | {r['label']} | {r['outcome']} | {r['first_alert_pts']} |")
    suffix='-'+COHORT.upper() if COHORT!='all' else ''
    (ROOT/('docs/FIRE-SMOKE-CHALLENGE'+suffix+'-20261028.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'unique_videos':len(unique),'summary':summary,'screening_passed':passed},indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['acquire','prepare','review','evaluate','summarize'])
    parser.add_argument('--cohort',choices=['all','furg','kmu','rise'],default='all')
    args=parser.parse_args();COHORT=args.cohort
    if COHORT!='all':
        if args.action=='acquire':parser.error('Acquisition always retains its complete original plan')
        WORK=WORK/COHORT;WORK.mkdir(exist_ok=True)
    sys.exit(globals()[args.action]())
