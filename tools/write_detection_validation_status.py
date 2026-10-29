"""Build a factual local detector status from preserved acceptance receipts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.runtime/capability-validation-20261027'


def read(path):
    return json.loads((WORK/path).read_text(encoding='utf-8'))


def main():
    fire=read('fire-followup/evaluation/fire_smoke-frozen-screen/report.json')
    extra=read('fire-followup/additional-negatives/fire_smoke-frozen-screen/report.json')
    violence=read('violence-followup/FINAL-SUMMARY-20261028.json')
    site=read('site-validation/screen-20261028/annotation-scores.json')
    rules=read('rule-coverage/RULE-COVERAGE.json')
    pose=read('site-validation/pose-heuristics-20261028/report.json')
    assert fire['complete_cases']==16 and extra['complete_cases']==2
    assert all(row['status']=='complete' for row in pose['reports'])
    status={'date':'2026-10-28 Africa/Algiers','all_detection_acceptance_passed':False,
        'software_distinct_tests_passed':186,'fire_smoke_original':fire['summary'],
        'fire_smoke_additional_negatives':extra['summary'],'fire_smoke_screening_passed':fire['screening_passed'] and extra['screening_passed'],
        'violence':violence['all_24'],'site_policy_screen':site,
        'pose_heuristic_cases':[{'video':r['video'],'status':r['status'],'events':[e['event']['type'] for e in r['events']]} for r in pose['reports']],
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'edge/detectors.py',ROOT/'tools/run_starter_review_checks.py',ROOT/'tools/eval_tests/test_detection_rule_coverage.py']},
        'production_activation_changed':False,'thresholds_or_models_changed':False,
        'camera_or_live_service_changed':False,
        'evidence_notes':['Historical person-down reference results reused; no new person-down benchmark claimed.',
            'Controlled logic tests are distinct from dataset recognition results.',
            'Traffic annotations label cars, not traffic violations; frame correspondence assumptions prevent certified benchmark claims.',
            'No broad reliability estimate, public activation, or dataset redistribution.']}
    (WORK/'DETECTION-STATUS-20261028.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    lines=['# Detector inventory and validation — 28 October 2026','',
        '**The work does not support an all-pass claim.** Software checks pass, fire/smoke passes its bounded reference screen, but violence and several site/heuristic checks fail or lack sufficient positive footage. No production profile, threshold, weight, camera, notification or public service was enabled or changed.','',
        '## Software corrections and checks','',
        '186 distinct test methods pass: 91 focused detection/evaluation/health (including37 new regressions),59 original application,8 isolated judge API,18 starter preservation/pipeline,10 triage. Three camera-integrity checks also appear in the focused combined94 receipt and are counted only once. Failed baseline attempts remain preserved.','',
        'Corrected vehicle and bag state expiry before matching after an expired observation gap; corrected loitering dwell across inactive schedule periods. The test runner now isolates the judge wrapper from original application tests because it owns process-global environment configuration. These corrections retain existing thresholds and cooldowns. See `.runtime/capability-validation-20261027/rule-coverage/RULE-COVERAGE.md` for per-capability test IDs, source backups and complete diffs.','',
        '## Per-detector disposition','',
        '| Capability in source | Evidence and result | Remaining acceptance |',
        '|---|---|---|',
        '| Person-down: legacy geometry, trained fall, confirmed pose and temporal observations | Controlled tests pass. Existing aligned reference4/4 positives and0/9 ordinary alerts, plus local fresh API fall8.7667s/walking quiet evidence retained. | Earlier dataset evidence reused, not new independent accuracy. Real camera/browser and representative field acceptance incomplete. |',
        '| Fire | Original4 positives detected/4 negatives quiet; additional negative quiet. | Only selected publisher clips, not field reliability. One original negative1.83s is shorter than2s persistence; retained and supplemented. |',
        '| Smoke | Original4 positives detected/4 negatives quiet; additional negative quiet. | Broader steam/dust/weather/viewpoint confusion tests remain necessary. |',
        '| Persistent violence model | 24 clips:0TP/12FN/1FP/11TN; failed. | Recognition needs a suitable model and new validation. Class mapping/loading/BGR/config checked; no threshold tuning to manufacture passes. |',
        '| Legacy violence adapter | Same real-score cache:0TP/12FN/3FP/9TN; failed. | Cache replay is3fps/default stride2, not another native-rate neural benchmark. |',
        '| Fight/altercation heuristic | Controlled positive/negative checks pass; real pedestrian and simulated-fight checks retained in pose/object reports below. | Not validated violence recall; ordinary pedestrian footage produced nuisance candidates. |',
        '| Crowd: legacy count/site region and dwell | Controlled checks pass. Annotation-policy test detects MOT17-04 crowd but misses MOT17-09; low-count CAVIAR controls quiet. | Counting/occlusion/resolution sensitivity; not a crowd-danger classifier. |',
        '| Intrusion | Controlled checks pass; all six pedestrian clips detect whole-view presence and empty-region controls stay quiet. | Artificial test zones, not real site authorization or certified location. |',
        '| Loitering | Timing regression fixed; controlled checks pass.5s demonstration policy agrees on3/6 pedestrian clips and misses3. | Tracking continuity and real site policy acceptance. Production default120s unchanged. |',
        '| Unattended bag/object | Controlled hold/owner/reset checks pass; CAVIAR LeftBag reference processed with compatible object model. No bag alert at default60s or optional5s demonstration. | Clip shorter than default60s; no default-duration positive acceptance. Bag recognition/ownership needs new footage, no claim all passes. |',
        '| Contact-and-flight/snatch indicator | Controlled positive/negative checks pass. Real pedestrian replay produces unsupported candidates. | Experimental movement indicator, no theft ground truth or validated theft recall. Remains opt-in. |',
        '| Running-group/panic indicator | Controlled runner-speed/count checks pass; selected footage processed. | No publisher-labelled3-runner positive in this subset; no claim of panic intent or positive recall. |',
        '| Stopped vehicles | Controlled movement/dwell/gap/region checks pass; two annotated traffic sequences processed. | Annotated segments shorter than default45s; no default-duration real positive acceptance. |',
        '| Congestion | Controlled stationary-count/region checks pass. Real traffic sequence outputs retained. | No labelled operational congestion positive or calibrated lane occupancy acceptance. |',
        '| Wrong-way | Controlled expected-direction positive/negative checks pass. | Publisher car tracks do not label legal lane direction; no real wrong-way positive validated. |',
        '| Motion fallback | Controlled blank/change/threshold checks pass. | Image motion only; no semantic incident recognition. |',
        '| Camera integrity/health, stale frames, model failures, recovery | Controlled checks pass. | No physical network/reconnect/multi-camera endurance test in this run. |',
        '| Night schedule/severity | Controlled schedule/escalation checks pass. | Site timezone and authorized operating schedule still required. |','',
        '## Actual data runs','',
        '- FIRESENSE:18 selected clips, frozen confidence.5/2s persistence;8 positives detected and10 negatives quiet. Original16 and added2 remain separate preserved reports. Negatives total169.689s across classes. No guaranteed false-alarm rate follows from zero observed false alerts.',
        '- AIRTLab:24 selected clips, pinned Git blobs, publisher action labels retained; known12 and predeclared followup12 reported separately. Some publisher positives include simulated weapons; none were dropped or relabelled after predictions. Training overlap unknown.',
        '- MOT17-04/09 and CAVIAR Walk1/LeftBag/Meet_Crowd/Fight_RunAway1:505 sampled frames.24 declared pedestrian policy checks:10TP/10TN/4FN. These are policy agreement checks, not24 independent cameras.',
        '- UrbanTracker Sherbrooke/Rouen:168 sampled frames in publisher annotated intervals. Car-box matches at IoU.5:151/202 and18/41 respectively. Approximate one-based video/annotation alignment, evaluation masks and sampling prevent treating this as certified tracking/detection benchmark accuracy. No precision estimate or traffic-violation truth inferred.','',
        '### Additional pose-model heuristic checks','']
    for row in status['pose_heuristic_cases']:
        lines.append('- '+Path(row['video']).name+': '+str(len(row['events']))+' events; '+(', '.join(sorted(set(row['events']))) or 'no heuristic alert')+'.')
    lines += ['', '## Reproducibility and provenance','',
        'All raw predictions, decoded timestamps, hashes, settings, source revisions, process exits, previous failed attempts and downloaded originals stay under `.runtime/capability-validation-20261027/` and `data/capability_validation_20261027/`. Scripts refuse to replace existing evaluation directories. Reports with process exit0 can still fail recognition acceptance.','',
        'Repeat site inference with a new report directory using `tools/run_site_validation.py collect`, then `replay`, then `score`. `tools/acquire_site_validation.py` preserves acquisition receipts and historical partials; `--traffic-only` handles the traffic references. Existing weights/environments were used; no packages or models installed. The separate full-source review archives and hosted replay sample were not updated by this work.','',
        'Primary dataset sources: [FIRESENSE](https://zenodo.org/records/836749), [AIRTLab](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos), [MOTChallenge MOT17](https://motchallenge.net/data/MOT17/), [CAVIAR](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/), [UrbanTracker](https://www.jpjodoin.com/urbantracker/dataset.html). Publisher terms/attribution and pinned/local hashes are retained. These local evaluation downloads are not a public media release.','',
        '## Next engineering gate','',
        'The remaining failures are recognition/tracking or missing representative positive-data gates. Use new labelled development/holdout material and a suitable model before enabling failed capabilities; do not reduce thresholds or narrow labels solely to pass the saved tests. Person-down and its evidence/review path remain isolated from experimental categories.']
    (ROOT/'docs/DETECTION-VALIDATION-20261028.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote detector status JSON and durable report; all_detection_acceptance_passed=false.')


if __name__=='__main__':main()
