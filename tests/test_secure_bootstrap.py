import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SecureBootstrapProcessTest(unittest.TestCase):
    def test_secure_rbac_mode_starts_without_shared_operator_keys(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(root / "backend"),
                "AICOS_API_KEY": "device-bootstrap-key-abcdefghijklmnopqrstuvwxyz-123456",
                "AICOS_ALLOW_SHARED_OPERATOR_KEYS": "false",
                "AICOS_OPERATOR_KEY": "",
                "AICOS_VIEWER_KEY": "",
                "AICOS_BOOTSTRAP_ADMIN_USER": "aicosadmin",
                "AICOS_BOOTSTRAP_ADMIN_PASSWORD": "Government Pilot Admin Password 123!",
                "DB_PATH": str(Path(td) / "aicos.db"),
                "SNAPSHOT_DIR": str(Path(td) / "snapshots"),
            })
            code = r'''
from fastapi.testclient import TestClient
import main
c=TestClient(main.app)
assert c.get('/api/health').status_code == 200
login=c.post('/api/auth/login',json={'username':'aicosadmin','password':'Government Pilot Admin Password 123!'})
assert login.status_code == 200, login.text
t=login.json()['token']
assert c.get('/api/events',headers={'Authorization':'Bearer '+t}).status_code == 200
assert c.get('/api/events?token='+t).status_code == 401
print('SECURE_BOOTSTRAP_OK')
'''
            cp = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(root), capture_output=True, text=True, timeout=30)
            if cp.returncode:
                self.fail((cp.stdout + "\n" + cp.stderr).strip())
            self.assertIn("SECURE_BOOTSTRAP_OK", cp.stdout)


if __name__ == "__main__":
    unittest.main()
