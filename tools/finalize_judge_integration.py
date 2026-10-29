"""Verify saved receipts/hashes and write a bounded local-delivery record."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.runtime/multidetector-integration-20261028'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    suites={}
    for name in ('application','focused','starter','completion'):
        folder=WORK/(name+'-final')
        value=json.loads((folder/'summary.json').read_text())
        if value['failures'] or value['errors'] or value['skipped'] or (folder/'exit-code.txt').read_text().strip()!='0':
            raise ValueError('Unsuccessful suite: '+name)
        suites[name]=value
    total=sum(v['passed'] for v in suites.values());assert total==197
    controls=json.loads((WORK/'dashboard-controls-final.stdout.txt').read_text())
    assert controls['passed']==12 and controls['failed']==0
    assert (WORK/'dashboard-controls-final.exit.txt').read_text().strip()=='0'
    inference=json.loads((WORK/'real-inference/report.json').read_text())
    assert inference['complete'] and len(inference['cases'])==7
    assert (WORK/'real-inference.process-exit.txt').read_text().strip()=='0'
    camera=json.loads((WORK/'camera-protocol.json').read_text());assert camera['passed']
    browser=json.loads((WORK/'BROWSER-ACCEPTANCE.json').read_text())
    assert browser['continuous_all_detector_run']['sampled_frames']==64
    assert browser['continuous_all_detector_run']['zone_incident_remained_detected']
    baseline=json.loads((ROOT/'.runtime/capability-validation-20261027/fire-followup/evaluation/fire_smoke-frozen-screen/report.json').read_text())
    unchanged={name:sha(ROOT/name)==digest for name,digest in baseline['source_sha256'].items()}
    assert all(unchanged.values()),unchanged
    changes=['deploy/judge_capabilities.py','deploy/judge_detector_pipeline.py','deploy/judge_workspace.py',
        'deploy/judge_inference_worker.py','tests/test_judge_workspace.py','tools/eval_tests/test_judge_detector_pipeline.py',
        'tools/check_judge_multidetector.py','tools/eval_violence_causal_windows.py','tools/finalize_judge_integration.py']
    for name in changes:ast.parse((ROOT/name).read_text(encoding='utf-8'))
    html=ROOT/'dashboard/judge-workspace.html';script='\n'.join(re.findall(r'<script>([\s\S]*?)</script>',html.read_text(encoding='utf-8')))
    (WORK/'dashboard-final.js').write_text(script,encoding='utf-8')
    syntax=subprocess.run(['node','--check',str(WORK/'dashboard-final.js')],capture_output=True,text=True,timeout=15)
    assert syntax.returncode==0,syntax.stderr
    with urlopen('http://127.0.0.1:8084/health',timeout=5) as response:health=json.load(response)
    assert health['ok'] and len(health['capabilities'])==12
    causal=json.loads((ROOT/'.runtime/violence-improvement-20261028/causal-window-evaluation/report.json').read_text())
    assert causal['complete'] and not causal['development_gate_passed']
    report=dict(local_server_ready=True,health=health,python_suites=suites,unique_python_tests=total,
        controller_checks=controls,real_inference_cases=7,camera_protocol=camera,browser_acceptance=browser,
        unchanged_reference_sources=unchanged,source_sha256={name:sha(ROOT/name) for name in changes+[str(html.relative_to(ROOT))]},
        violence_causal_experiment=dict(counts=causal['summary'],gate_passed=False,holdout_untouched=True),
        physical_camera_acceptance=False,selected_file_browser_acceptance=False,public_inference_deployed=False,
        submission_saved=False,limitations='Local integration passes do not establish recognition accuracy or complete public submission readiness.')
    (WORK/'DELIVERY-RESULT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(local_server_ready=True,python_tests=total,controller_checks=controls['passed'],
                         real_inference_cases=7,reference_sources_unchanged=True)))


if __name__=='__main__':main()
