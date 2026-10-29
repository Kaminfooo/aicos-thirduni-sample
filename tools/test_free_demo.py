"""Exercise the free judge wrapper twice in isolated project-local subprocesses."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECK = r'''
import io,json,os,sys
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path.cwd()))
from deploy import free_demo as demo
assert Path(os.environ['DB_PATH']).resolve().is_relative_to(Path.cwd()/'.runtime')
assert not demo.private.exists()
assert not os.environ.get('AICOS_BOOTSTRAP_ADMIN_PASSWORD')
assert not os.environ.get('AICOS_LLM_API_KEY')
assert not demo.core.ALLOW_SHARED_OPERATOR_KEYS
assert demo.core.user_auth.count_users()==1
with TestClient(demo.app) as client:
 assert client.get('/api/health').status_code==200
 info=client.get('/api/demo-info').json();assert not info['persistent'] and not info['live_inference']
 page=client.get('/dashboard/login.html');assert page.status_code==200
 assert 'Shared sandbox' in page.text and 'judge-demo' in page.text
 assert client.get('/api/events').status_code==401
 login=client.post('/api/auth/login',json={'username':demo.JUDGE_USER,'password':demo.JUDGE_PASSWORD})
 assert login.status_code==200,login.text
 headers={'Authorization':'Bearer '+login.json()['token']}
 events=client.get('/api/events',headers=headers).json()['events'];assert len(events)==1
 assert events[0]['source_id']=='REPLAY-LICENSED-01' and events[0]['status']=='detected'
 for kind in ('snapshot','clip'):
  response=client.get('/api/events/1/'+kind,headers=headers);assert response.status_code==200
  Image.open(io.BytesIO(response.content)).verify()
 assert client.get('/api/events/1/evidence/verify',headers=headers).json()['ok']
 assert client.post('/api/events',headers=headers,json={}).status_code==403
 assert client.post('/api/admin/users',headers=headers,json={}).status_code==403
 assert client.post('/api/cameras/heartbeat',headers=headers,json={}).status_code==403
 assert client.get('/demo-assets/credits.html').status_code==200
 with client.websocket_connect('/ws/events') as ws:
  ws.send_json({'kind':'auth','token':login.json()['token']})
  snapshot=ws.receive_json();assert snapshot['kind']=='snapshot' and len(snapshot['events'])==1
 draft=client.post('/api/events/1/triage',headers=headers,json={});assert draft.status_code==201,draft.text
 did=draft.json()['id']
 review=client.post(f'/api/triage/{did}/review',headers=headers,json={'decision':'accept','note':'Sandbox verification'})
 assert review.status_code==200,review.text
 assert review.json()['review_status']=='accepted_for_demo'
 response=client.post('/api/events/1/status',headers=headers,json={'status':'approved'});assert response.status_code==200,response.text
 assert client.get('/api/evidence/chain/verify',headers=headers).json()['ok']
 rows=demo.core._conn.execute('SELECT action,note FROM audit WHERE event_id=1').fetchall()
 assert any(r['action']=='approved' and 'judge-demo' in r['note'] for r in rows)
 assert client.post('/api/events/1/triage',headers=headers,content=b'x'*32769).status_code==413
 demo.app.state.rate_windows={};demo.app.state.mutation_count=0
 def chunks():
  yield b'x'*20000
  yield b'x'*20000
 assert client.post('/api/events/1/triage',headers=headers,content=chunks()).status_code==413
 demo.app.state.rate_windows={};demo.app.state.mutation_count=1000
 assert client.post('/api/events/1/triage',headers=headers,json={}).status_code==429
print(json.dumps({'flow_assertions_passed':True,'fresh_incidents':len(events),'signed_chain_verified':True,
 'live_inference':False,'bootstrap_credentials_retained':False,'boot_folder':demo.BOOT.name}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / '.runtime'):
        parser.error('Choose a new project .runtime output directory')
    out.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('AICOS_')}
    env.update(AICOS_DEMO_RUNTIME_DIR=str(out / 'boot-data'), TEMP=str(out), TMP=str(out), PYTHONDONTWRITEBYTECODE='1')
    results = []
    for attempt in range(2):
        result = subprocess.run([sys.executable, '-I', '-B', '-c', CHECK], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=60)
        (out / f'boot-{attempt + 1}.stdout.txt').write_text(result.stdout, encoding='utf-8')
        (out / f'boot-{attempt + 1}.stderr.txt').write_text(result.stderr, encoding='utf-8')
        (out / f'boot-{attempt + 1}.exit.txt').write_text(str(result.returncode))
        if result.returncode:
            print(result.stderr)
            return result.returncode
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert results[0]['boot_folder'] != results[1]['boot_folder']
    summary = {'boots_verified': 2, 'separate_fresh_databases': True, 'both_seeded_with_one_licensed_incident': True,
               'flow_assertions_passed_both_boots': all(r['flow_assertions_passed'] for r in results), 'private_database_used': False,
               'live_models_downloaded': False, 'cloud_deployment_tested': False, 'published': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
