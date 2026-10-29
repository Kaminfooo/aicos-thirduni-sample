"""Verify saved comparison evidence; never turn software passes into accuracy."""
import ast,difflib,hashlib,json,re,subprocess
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.runtime/detector-upgrade-20261028'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads((WORK/path).read_text())


def main():
    suites={key:read(key+'-final/summary.json') for key in ('application','focused','starter','completion')}
    assert all(not (s['failures'] or s['errors'] or s['skipped']) for s in suites.values())
    assert sum(s['passed'] for s in suites.values())==205
    node=read('dashboard-checks.stdout.txt');assert node['passed']==12 and node['failed']==0
    assert (WORK/'dashboard-checks.exit.txt').read_text().strip()=='0'
    yolo=read('yolo26-violence-evaluation/report.json');video=read('videomae-evaluation-compatible/report.json')
    assert all(r['complete'] and len(r['cases'])==20 and not r['gate_passed'] for r in (yolo,video))
    assert not video['load']['missing'] and not video['load']['unexpected']
    assert len(video['load']['compatibility']['layers'])==12
    bag={key:read('unattended-'+key+'-cfr/report.json') for key in ('nano','small')}
    assert all(r['complete'] and len(r['cases'])==3 for r in bag.values())
    pose=read('interaction-pose-evaluation/report.json');assert pose['complete'] and len(pose['cases'])==5
    clock=read('camera-protocol.json');assert clock['passed']
    changes=['edge/detectors.py','deploy/judge_capabilities.py','dashboard/judge-workspace.html',
        'tools/eval_yolo26_violence_upgrade.py','tools/eval_videomae_upgrade.py','tools/videomae_qkv_compat.py',
        'tools/eval_unattended_dataset_upgrade.py','tools/avi_reference_clock.py','tools/eval_interaction_pose_upgrade.py']
    for name in changes:
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(encoding='utf-8'))
    html=(ROOT/'dashboard/judge-workspace.html').read_text(encoding='utf-8')
    assert 'violence failed validation' not in html.lower() and 'Evaluation scope &amp;' not in html
    script='\n'.join(re.findall(r'<script>([\s\S]*?)</script>',html));(WORK/'dashboard-final.js').write_text(script,encoding='utf-8')
    syntax=subprocess.run(['node','--check',str(WORK/'dashboard-final.js')],capture_output=True,text=True,timeout=15)
    assert syntax.returncode==0,syntax.stderr
    before=(WORK/'detectors.before.py').read_text();after=(ROOT/'edge/detectors.py').read_text()
    (WORK/'unattended-fix.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before/edge/detectors.py',tofile='after/edge/detectors.py')))
    with urlopen('http://127.0.0.1:8084/health',timeout=5) as response:health=json.load(response)
    assert health['ok'] and len(health['capabilities'])==12
    assert sum(r['status']=='Research preview' for r in health['capabilities'])==5
    assert all('failed' not in r['status'].lower() for r in health['capabilities'])
    result=dict(software_suites=suites,python_tests=205,dashboard_checks=node,
        violence_candidates={'yolo26s':yolo['summary'],'videomae':video['summary']},
        recognition_acceptance_complete=False,new_violence_model_promoted=False,new_pose_model_promoted=False,
        new_object_model_promoted=False,holdout_untouched=True,
        unattended_bugs_fixed=['person-box proximity replaces torso-center proximity','one object track per detection per frame'],
        fresh_aboda_videos=3,matched_pose_comparison_videos=5,camera_protocol=clock,
        live_health=health,source_sha256={name:sha(ROOT/name) for name in changes},
        private_platform_published=False,public_demo_redeployed=False,physical_camera_tested=False,
        scope='Real recorded dataset replay and software checks; no claim of field reliability or all-detector recognition pass.')
    (WORK/'UPGRADE-RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(python_tests=205,dashboard_checks=12,local_service_healthy=True,
        unattended_bugs_fixed=2,violence_models_tested=2,new_violence_promoted=False)))


if __name__=='__main__':main()
