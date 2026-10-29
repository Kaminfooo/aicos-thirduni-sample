"""Deliver cross-source challenge evidence without claiming field accuracy."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.runtime/fire-smoke-challenge-20261028'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    phases={};inputs=[];selected=[];cases=[];miss_diagnostics=[]
    for name,count in [('furg',6),('rise',9),('kmu',16)]:
        final=WORK/name/'FINAL-RESULT.json';planpath=WORK/name/'PREDECLARED.json'
        report=json.loads(final.read_text());plan=json.loads(planpath.read_text())
        assert report['unique_videos']==count and len(plan['cases'])==count
        assert report['source_plan_sha256']==sha(planpath)
        assert report['inference_report_sha256']==sha(WORK/name/'fire_smoke-frozen-screen/report.json')
        for row in plan['cases']:
            path=ROOT/row['file'];assert sha(path)==row['sha256']
            selected.append({'phase':name,**row})
        phases[name]={'unique_videos':count,'summary':report['summary'],'screening_passed':report['screening_passed']}
        cases.extend({'phase':name,**r} for r in report['cases'])
        raw=json.loads((WORK/name/'fire_smoke-frozen-screen/report.json').read_text())
        for index,row in enumerate(raw['cases']):
            if row.get('outcome')!='FN':continue
            trace=WORK/name/'fire_smoke-frozen-screen'/(f'{index:02d}-'+Path(row['file']).stem+'.trace.json')
            observations=json.loads(trace.read_text());label=row['label']
            miss_diagnostics.append({'phase':name,'file':row['file'],'label':label,
                'samples_with_qualifying_label':sum(label in o.get('holds',{}) for o in observations),
                'maximum_confirmation_seconds':max([o.get('holds',{}).get(label,0) for o in observations]+[0]),
                'scope':'Cached detector decisions, not raw confidence/logits; no model rerun or threshold selection.'})
        inputs.append({'file':str(final.relative_to(ROOT)),'sha256':sha(final)})
    baseline=[]
    for rel in ['evaluation/fire_smoke-frozen-screen/report.json','additional-negatives/fire_smoke-frozen-screen/report.json']:
        p=ROOT/'.runtime/capability-validation-20261027/fire-followup'/rel
        baseline.extend(json.loads(p.read_text())['cases'])
    old={r['sha256']:r['file'] for r in baseline}
    previous_duplicates=[{'new':r['file'],'previous':old[r['sha256']]} for r in selected if r['sha256'] in old]
    cohorts_by_hash={}
    for row in selected:cohorts_by_hash.setdefault(row['sha256'],[]).append(row['file'])
    cross_duplicates=[files for files in cohorts_by_hash.values() if len(files)>1]
    model=ROOT/'models/pretrained/fire-smoke-yolo26n.pt'
    manifest=json.loads((ROOT/'models/pretrained/THIRDUNI-EXTENSIONS.json').read_text())
    assert sha(model)==next(r['sha256'] for r in manifest['assets'] if r['name']==model.name)
    original_plan=json.loads((ROOT/'.runtime/capability-validation-20261027/fire-followup/evaluation/VALIDATION-PLAN.json').read_text())
    assert all(sha(ROOT/n)==digest for n,digest in original_plan['source_sha256'].items())
    checks={'previous_detector_source_hashes_match':True,'previous_model_hash_matches':True,
        'cohort_report_and_media_hashes_match':True,'tool_contract_tests':7,'production_changes':False,
        'new_selected_videos':len(selected),'new_unique_media_hashes':len(cohorts_by_hash),
        'scored_unique_videos':len({r['file'] for r in cases if r['status']=='complete'}),
        'unscored_unique_videos':len({r['file'] for r in cases if r['status']!='complete'}),
        'previous_screen_exact_duplicates':previous_duplicates,'cross_cohort_exact_duplicates':cross_duplicates,
        'not_proof_of_training_independence':True}
    summary={'date':'2026-10-28 Africa/Algiers','phases':phases,'cases':cases,'inputs':inputs,'checks':checks,'miss_diagnostics':miss_diagnostics,
        'overall_challenge_passed':all(v['screening_passed'] for v in phases.values()),
        'limitations':['Single previous passing dataset is insufficient for general reliability.',
            'RISE is timelapse; encoded results cannot be converted to real-time camera-hours or physical latency.',
            'FURG annotation counts differ from actual decode; precise onset mapping remains unverified.',
            'One UrbanTracker control has a non-monotonic final PTS and is unscored.',
            'No model or threshold tuning, public activation, package installation or camera/service changes.']}
    output=WORK/'CHALLENGE-SUMMARY.json'
    if output.exists():raise ValueError('Preserve completed summary; choose a new delivery version')
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Fire/smoke challenge results — 28 October 2026','',
        '**The cross-source challenge does not support general detection reliability.** The previously passing18-clip FIRESENSE result is retained. New sources are scored separately, including failures and an invalid control.','',
        '| Cohort | Class | Detected positives | Missed positives | Negatives with alerts | Quiet negatives | Unscored/errors |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,phase in phases.items():
        for label,s in phase['summary'].items():
            if not sum(s[k] for k in ('TP','FN','FP','TN','ERROR')):continue
            lines.append(f"| {name.upper()} | {label} | {s['TP']} | {s['FN']} | {s['FP']} | {s['TN']} | {s['ERROR']} |")
    lines+=['',f"Selected videos: {len(selected)}; scored: {checks['scored_unique_videos']}; unscored: {checks['unscored_unique_videos']}; distinct media hashes: {len(cohorts_by_hash)}. Exact matches to the prior18-clip screen: {len(previous_duplicates)}. Different hashes do not exclude re-encoded shared source scenes or model-training overlap.",'',
        '## Protocol and source records','',
        'The source families, FURG scenarios and RISE camera/label selection were fixed before their challenge predictions. Every video in the three KMU archives was selected; no failed prediction changed the list. The frozen model SHA256, detector source hashes, confidence0.5, two-second persistence,3fps sampling and480-pixel resize were retained. Plans, original downloads, receipts, failed/interrupted downloads, raw timestamps, per-frame decisions, decode reviews and actual process exits are under `.runtime/fire-smoke-challenge-20261028/`.','',
        'Original partial downloads were copied before resuming. KMU archives passed verified TLS, ETag continuity, exact Content-Range/size and complete ZIP member CRC checks. FURG media/XML were verified against a pinned Git tree/blob. RISE metadata was Git-blob verified and media hashes recorded after verified HTTPS. Publisher terms remain attached locally; no media was published.','',
        '## Timing and label limits','',
        'KMU is ordinary video; its smoke-positive and smoke/flame-like-negative families are scored under their publisher labels. Other hazard classes in a positive clip stay unlabelled. UrbanTracker controls are coarse reviewed ordinary traffic, not publisher fire annotations. One control (Rouen) returns a decreasing final decoded timestamp and is explicitly unscored, not a quiet negative.','',
        'FURG includes house, car and night-battery flame scenes plus a non-fire robot view. All four annotation/video frame counts differ. Exact onset delay is withheld; already-burning clips report only first-alert time within the recording. Smoke clips have no frame-level onset labels.','',
        'RISE is a separate industrial-smoke transfer stress test. Each selected clip has36 frames and2.9167 encoded seconds. The publisher distinguishes smoke from steam and includes steam distractions. Its compressed playback and domain differ from regular CCTV. Do not use the encoded-hour figure in the raw report as a field false-alarm rate, or the two-second encoded hold as physical time. Two missed positives showed short confirmations; one never formed a qualifying hold.','',
        'KMU negative encoded exposure per class: '+str(round(phases['kmu']['summary']['smoke']['negative_encoded_seconds'],2))+' seconds. The same negative clips are scored for both classes; those exposures overlap and cannot be added as independent camera-hours. This remains limited reference footage, not an installation acceptance test.','',
        '## Miss diagnostics','']
    for row in miss_diagnostics:
        lines.append('- '+row['phase'].upper()+' / '+Path(row['file']).name+': '+str(row['samples_with_qualifying_label'])+' sampled decisions with a qualifying '+row['label']+' candidate; longest confirmation '+str(round(row['maximum_confirmation_seconds'],3))+' encoded seconds. This uses saved decisions, not raw model logits.')
    lines+=['', '## Disposition','',
        'Keep the prior passing result and this failed/mixed challenge visible together. Fire success on selected scenes does not establish general smoke reliability. Preserve the baseline and investigate smoke misses/false alarms on a separate development set before evaluating any changed model or threshold on untouched material. No production model, profile, threshold, service or camera was changed in this challenge. Seven validation-tool contract checks pass; the historical186 application/detection checks were not needlessly rerun for tooling-only work.','',
        '## Evidence','',
        '- [FURG cases](FIRE-SMOKE-CHALLENGE-FURG-20261028.md)',
        '- [KMU cases](FIRE-SMOKE-CHALLENGE-KMU-20261028.md)',
        '- [RISE cases and timing warning](FIRE-SMOKE-CHALLENGE-RISE-20261028.md)','',
        'Primary sources: [FURG](https://github.com/steffensbola/furg-fire-dataset), [KMU CVPR Lab](https://cvpr.kmu.ac.kr/Dataset/Dataset.htm), [CMU RISE](https://github.com/CMU-CREATE-Lab/deep-smoke-machine), [RISE paper](https://arxiv.org/html/2005.06111), [UrbanTracker](https://www.jpjodoin.com/urbantracker/dataset.html).']
    (ROOT/'docs/FIRE-SMOKE-CHALLENGE-20261028.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'phases':phases,'checks':checks},indent=2))


if __name__=='__main__':main()
