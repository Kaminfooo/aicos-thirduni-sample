"""Isolated, resettable licensed-replay demo. Never import into a live backend process.

No model/camera/external AI service runs here. Existing authentication, evidence,
triage and audit code executes against a fresh demo-only database on each boot.
"""
from contextlib import asynccontextmanager
import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name('free-demo-assets')
if 'main' in sys.modules:
    raise RuntimeError('Free demo must run in a separate process from the private backend')
runtime = Path(os.environ.get('AICOS_DEMO_RUNTIME_DIR', ROOT / '.runtime/free-judge-demo')).resolve()
if not runtime.is_relative_to(ROOT):
    raise RuntimeError('Demo runtime must stay inside its project checkout')
runtime.mkdir(parents=True, exist_ok=True)
BOOT = Path(tempfile.mkdtemp(prefix='boot-', dir=runtime))
for name in list(os.environ):
    if name.startswith('AICOS_') or name in ('DB_PATH', 'SNAPSHOT_DIR', 'BACKEND_URL'):
        del os.environ[name]
device_key = secrets.token_urlsafe(40)
hostname = os.environ.get('RENDER_EXTERNAL_HOSTNAME', '').strip()
os.environ.update(DB_PATH=str(BOOT / 'demo.db'), SNAPSHOT_DIR=str(BOOT / 'evidence'),
    AICOS_API_KEY=device_key, AICOS_ALLOW_SHARED_OPERATOR_KEYS='false',
    AICOS_WS_QUERY_TOKEN_ALLOWED='false', AICOS_NOTIFY_ON='off',
    AICOS_INTEGRATION_WEBHOOK_URLS='', AICOS_CORS_ORIGINS='',
    AICOS_EVIDENCE_SIGNING_REQUIRED='true', AICOS_SESSION_SECONDS='3600',
    AICOS_MAX_REQUEST_BYTES='32768', AICOS_AI_REPORTS='false',
    AICOS_TRUSTED_HOSTS=','.join(filter(None, ('127.0.0.1', 'localhost', 'testserver', hostname))),
    AICOS_BEHIND_HTTPS='true' if hostname else 'false')
sys.path.insert(0, str(ROOT / 'backend'))
from crypto_utils import write_keypair
private, public = BOOT / 'private.pem', BOOT / 'public.pem'
write_keypair(private, public)
os.environ.update(AICOS_EVIDENCE_PRIVATE_KEY=str(private), AICOS_EVIDENCE_PUBLIC_KEY=str(public))
try:
    core = importlib.import_module('main')
finally:
    private.unlink(missing_ok=True)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Intentionally public sandbox credentials, never an owner/device/admin password.
JUDGE_USER, JUDGE_PASSWORD = 'judge-demo', 'ThirdUniDemo2026!'
core.user_auth.create_user(JUDGE_USER, JUDGE_PASSWORD, 'operator')
BOOT_TIME = time.time()
CREDIT = ('ASSIST-IoT dataset — Anastasiya Danilenka, Piotr Sowiński, Monika Kobus, '
          'Anna Dąbrowska, Kajetan Rachwał, Karolina Bogacka, Krzysztof Baszczyński. '
          'CC BY 4.0; EU grant 957258; CIOP-PIB facilities. No endorsement implied.')


def verified_asset(name, expected):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise RuntimeError('Invalid demo asset name')
    path = ASSETS / name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('Missing or linked demo asset')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError('Demo asset hash mismatch: ' + name)
    return data


async def seed():
    manifest = json.loads((ASSETS / 'seed.json').read_text(encoding='utf-8'))
    for name, sha in manifest['files'].items():
        verified_asset(name, sha)
    event = manifest['event']
    event['snapshot'] = base64.b64encode(verified_asset('trigger.jpg', manifest['files']['trigger.jpg'])).decode()
    event['clip'] = base64.b64encode(verified_asset('context.gif', manifest['files']['context.gif'])).decode()
    result = await core.ingest_event(core.EventIn(**event), x_aicos_key=device_key)
    eid = result['id']
    core.triage_assistant.create(eid, 'system:licensed-demo-seed')
    if not core.evidence_manager.verify(eid).get('ok'):
        raise RuntimeError('Seed evidence verification failed')


@asynccontextmanager
async def lifespan(application):
    async with core.app.router.lifespan_context(core.app):
        await seed()
        yield
    core._conn.close()


app = FastAPI(title='AICOS resettable ThirdUni demonstration', lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)
app.state.mutation_count = 0
app.state.rate_windows = {}


def permitted_write(path):
    return (path in ('/api/auth/login', '/api/auth/logout') or
            re.fullmatch(r'/api/events/1/(triage|status)', path) or
            re.fullmatch(r'/api/triage/[1-9][0-9]*/review', path))


@app.middleware('http')
async def sandbox(request: Request, call_next):
    path = request.url.path
    if path.startswith('/api/admin') or path.startswith('/api/integrations'):
        return JSONResponse({'detail': 'Unavailable in the public replay sandbox'}, status_code=403)
    if request.method not in ('GET', 'HEAD'):
        if request.method != 'POST' or not permitted_write(path):
            return JSONResponse({'detail': 'Only demo login and review decisions are enabled'}, status_code=403)
        try:
            size = int(request.headers.get('content-length', '0'))
        except ValueError:
            return JSONResponse({'detail': 'Invalid content length'}, status_code=400)
        if size > 32768:
            return JSONResponse({'detail': 'Demo request too large'}, status_code=413)
        # Bound actual streamed bodies too; Content-Length is not trusted alone.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 32768:
                return JSONResponse({'detail': 'Demo request too large'}, status_code=413)
        request._body = bytes(body)
        now = time.monotonic()
        bucket = int(now // 60)
        app.state.rate_windows = {k: v for k, v in app.state.rate_windows.items() if k[1] == bucket}
        peer = request.client.host if request.client else 'unknown'
        key = (peer, bucket)
        count = app.state.rate_windows.get(key, 0)
        if count >= 30 or app.state.mutation_count >= 1000:
            return JSONResponse({'detail': 'Demo review limit reached; try later'}, status_code=429)
        app.state.rate_windows[key] = count + 1
        app.state.mutation_count += 1
    response = await call_next(request)
    if path.startswith('/dashboard/') and response.status_code == 200 and 'text/html' in response.headers.get('content-type', ''):
        data = b''.join([part async for part in response.body_iterator]).decode('utf-8')
        notice = ('<aside style="padding:14px 18px;background:#fff5d6;color:#624900;border:1px solid #e5ca7b;border-radius:10px;margin-bottom:18px;font:14px system-ui">'
                  '<strong>Recorded demonstration — no live camera or cloud inference.</strong> '
                  'Shared sandbox: reviews, sessions and signing identity reset when this free service restarts. '
                  'Demonstration procedures only; no dispatch. '
                  '<a href="/demo-assets/credits.html">Source, model provenance and limitations</a>.</aside>')
        data = re.sub(r'(<body\b[^>]*>)', lambda match: match[1] + notice, data, count=1)
        if path.endswith('/login.html'):
            instructions = ('<p style="color:#43566d">Public sandbox login: <strong>' + JUDGE_USER +
                            '</strong><br>Password: <strong>' + JUDGE_PASSWORD +
                            '</strong><br>This account controls demo data only. Never enter your personal password.</p>')
            data = data.replace('<button id="login">', instructions + '<button id="login">', 1)
        headers = {k: v for k, v in response.headers.items() if k.lower() not in ('content-length', 'content-type', 'content-encoding')}
        headers['Cache-Control'] = 'no-store'
        return HTMLResponse(data, headers=headers, status_code=response.status_code)
    return response


@app.get('/')
def home():
    return RedirectResponse('/dashboard/login.html')


@app.get('/api/demo-info')
def info():
    return {'mode': 'recorded_replay', 'live_inference': False, 'persistent': False,
            'shared_test_account': True, 'started_at': BOOT_TIME,
            'procedures': 'demonstration', 'owner_private_data_included': False,
            'credits': '/demo-assets/credits.html'}


app.mount('/demo-assets', StaticFiles(directory=str(ASSETS)), name='licensed-demo-assets')
app.mount('/', core.app)
