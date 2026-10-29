import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from crypto_utils import load_private_key, sign_envelope, write_keypair
from licensing import LICENSE_SCHEMA
from certification import CERT_SCHEMA


class GovernmentBackendE2E(unittest.TestCase):
    def test_signed_government_mode(self):
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s); certs = td / "certs"; certs.mkdir(); snaps = td / "snaps"; snaps.mkdir()
            vp, vu = td / "vendor-private.pem", td / "vendor-public.pem"
            ep, eu = td / "evidence-private.pem", td / "evidence-public.pem"
            write_keypair(vp, vu); write_keypair(ep, eu)
            now = time.time()
            lic_payload = {
                "license_id":"E2E-LIC", "customer":"Test Government", "deployment_id":"E2E",
                "not_before":now-10, "expires_at":now+3600, "grace_days":0,
                "max_cameras":2, "modules":["violence"],
                "features":["forensic_search","signed_evidence","camera_certification"]
            }
            lic = td / "license.json"
            lic.write_text(json.dumps(sign_envelope(LICENSE_SCHEMA, lic_payload, load_private_key(vp))))
            cert_payload = {
                "certificate_id":"E2E-CERT", "camera_id":"CAM-1", "not_before":now-10,
                "expires_at":now+3600, "assurance_score":90,
                "detectors":{"violence":{"status":"certified","score":90}}
            }
            (certs / "CAM-1.json").write_text(json.dumps(sign_envelope(CERT_SCHEMA, cert_payload, load_private_key(vp))))
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(ROOT / "backend"),
                "AICOS_API_KEY":"device-key-abcdefghijklmnopqrstuvwxyz-123456",
                "AICOS_OPERATOR_KEY":"operator-key-abcdefghijklmnopqrstuvwxyz-123456",
                "AICOS_VIEWER_KEY":"viewer-key-abcdefghijklmnopqrstuvwxyz-12345678",
                "AICOS_LICENSE_REQUIRED":"true", "AICOS_LICENSE_FILE":str(lic), "AICOS_LICENSE_PUBLIC_KEY":str(vu),
                "AICOS_REQUIRE_CAMERA_CERT":"true", "AICOS_CAMERA_CERT_DIR":str(certs), "AICOS_CERT_PUBLIC_KEY":str(vu),
                "AICOS_EVIDENCE_SIGNING_REQUIRED":"true", "AICOS_EVIDENCE_PRIVATE_KEY":str(ep), "AICOS_EVIDENCE_PUBLIC_KEY":str(eu),
                "DB_PATH":str(td / "aicos.db"), "SNAPSHOT_DIR":str(snaps),
            })
            code = r'''
import base64, os, time
from fastapi.testclient import TestClient
import main
c=TestClient(main.app); dev={'X-AICOS-KEY':os.environ['AICOS_API_KEY']}; view={'X-AICOS-OPERATOR-KEY':os.environ['AICOS_VIEWER_KEY']}
ev={'event_uuid':'22222222-2222-4222-8222-222222222222','source_id':'CAM-1','agent':'security','type':'Possible fight / altercation','severity':4,'confidence':.91,'snapshot':base64.b64encode(b'\xff\xd8e2e').decode()}
r=c.post('/api/events',headers=dev,json=ev); assert r.status_code==201, r.text
id=r.json()['id']; assert c.get(f'/api/events/{id}/evidence/verify',headers=view).status_code==200
assert len(c.get('/api/forensics/search',headers=view,params={'q':'fight'}).json()['events'])==1
obs={'source_id':'CAM-1','observations':[{'ts':time.time(),'kind':'person','confidence':.9,'color':'red','direction':'right','bbox':[.1,.2,.2,.4]}]}
assert c.post('/api/forensics/observations',headers=dev,json=obs).status_code==201
assert len(c.get('/api/forensics/objects',headers=view,params={'kind':'person','color':'red'}).json()['observations'])==1
print('PASS')
'''
            proc = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
            self.assertIn("PASS", proc.stdout)


if __name__ == "__main__":
    unittest.main()
