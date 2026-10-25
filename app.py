"""Standalone replay/review sample. Contains no private AICOS application imports."""
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import tempfile
import threading
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parent
COOKIE = 'aicos_sample_session'
PROCEDURE = {
    'id': 'DEMO-PERSON-DOWN', 'version': '1.0', 'status': 'demonstration',
    'steps': [
        'Review the image and surrounding recording; record uncertainty.',
        'Check the source and visible context. A lying posture does not establish fainting.',
        'For a real incident, use an approved site procedure and authorized human responders.'
    ]
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


class Review(BaseModel):
    decision: str = Field(pattern='^(accept|reject)$')
    note: str = Field(min_length=1, max_length=500)


class Decision(BaseModel):
    status: str = Field(pattern='^(approved|dismissed)$')


def create_app(runtime_dir=None):
    runtime = Path(runtime_dir or ROOT / '.runtime').resolve()
    if not runtime.is_relative_to(ROOT):
        raise ValueError('Runtime must remain inside this sample folder')
    assets = ROOT / 'assets'
    record = json.loads((assets / 'record.json').read_text(encoding='utf-8'))
    lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        runtime.mkdir(parents=True, exist_ok=True)
        boot = Path(tempfile.mkdtemp(prefix='boot-', dir=runtime))
        db = sqlite3.connect(boot / 'sample.db', check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.executescript('''
            CREATE TABLE sessions(id TEXT PRIMARY KEY, review TEXT, status TEXT);
            CREATE TABLE audit(id INTEGER PRIMARY KEY, session TEXT, payload TEXT, previous TEXT, hash TEXT);
        ''')
        app.state.db = db
        app.state.rate = {}
        app.state.boot = boot.name
        app.state.started = time.time()
        yield
        db.close()

    app = FastAPI(title='AICOS standalone replay sample', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    host = os.environ.get('RENDER_EXTERNAL_HOSTNAME', '').strip()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', 'testserver', *([host] if host else [])])

    def media_ok():
        for name, expected in record['media_sha256'].items():
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
                return False
            path = assets / name
            if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return False
        return True

    def audit(sid, action, details):
        row = app.state.db.execute('SELECT hash FROM audit WHERE session=? ORDER BY id DESC LIMIT 1', (sid,)).fetchone()
        previous = row['hash'] if row else ''
        payload = canonical({'time': time.time(), 'actor': 'public-demo-session', 'action': action, 'details': details})
        digest = hashlib.sha256((previous + payload).encode()).hexdigest()
        app.state.db.execute('INSERT INTO audit(session,payload,previous,hash) VALUES(?,?,?,?)', (sid, payload, previous, digest))

    def session(request, response, new=False):
        sid = request.cookies.get(COOKIE, '')
        with lock:
            row = None if new else app.state.db.execute('SELECT * FROM sessions WHERE id=?', (sid,)).fetchone()
            if row is None:
                if app.state.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0] >= 500:
                    raise HTTPException(503, 'Sample capacity reached; restart creates a fresh demo')
                sid = secrets.token_hex(24)
                app.state.db.execute('INSERT INTO sessions VALUES(?,?,?)', (sid, 'pending', 'detected'))
                audit(sid, 'replay_loaded', {'record_id': record['id'], 'media_sha256': record['media_sha256']})
                app.state.db.commit()
                response.set_cookie(COOKIE, sid, httponly=True, secure=bool(host), samesite='strict', max_age=3600)
            return sid

    def state(sid):
        row = app.state.db.execute('SELECT review,status FROM sessions WHERE id=?', (sid,)).fetchone()
        return dict(row)

    def require_media():
        if not media_ok():
            raise HTTPException(409, 'Sample media integrity check failed')

    @app.middleware('http')
    async def bounds(request: Request, call_next):
        if request.method == 'POST':
            origin = request.headers.get('origin')
            if origin and urlsplit(origin).netloc != request.headers.get('host'):
                return JSONResponse({'detail': 'Cross-origin writes are disabled'}, status_code=403)
            size = 0
            chunks = []
            async for part in request.stream():
                size += len(part)
                if size > 4096:
                    return JSONResponse({'detail': 'Sample request too large'}, status_code=413)
                chunks.append(part)
            request._body = b''.join(chunks)
            bucket = int(time.monotonic() // 60)
            peer = request.client.host if request.client else 'unknown'
            with lock:
                app.state.rate = {key: value for key, value in app.state.rate.items() if key[1] == bucket}
                key = (peer, bucket)
                if app.state.rate.get(key, 0) >= 30:
                    return JSONResponse({'detail': 'Demo rate limit; try again later'}, status_code=429)
                app.state.rate[key] = app.state.rate.get(key, 0) + 1
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Frame-Options'] = 'DENY'
        return response

    @app.get('/health')
    def health():
        ok = media_ok()
        return JSONResponse({'ok': ok, 'mode': 'standalone-recorded-sample', 'live_inference': False}, status_code=200 if ok else 503)

    @app.get('/')
    def home():
        return FileResponse(ROOT / 'web/index.html')

    @app.get('/api/demo')
    def demo(request: Request, response: Response):
        sid = session(request, response)
        with lock:
            return {'record': record, 'procedure': PROCEDURE, 'state': state(sid), 'media_match': media_ok(),
                    'limitations': 'Independent replay sample. No private detector, live inference, login identity or dispatch. Resets on restart.'}

    @app.post('/api/review')
    def review(body: Review, request: Request, response: Response):
        if not body.note.strip():
            raise HTTPException(422, 'Add a review note')
        sid = session(request, response)
        require_media()
        with lock:
            if state(sid)['review'] != 'pending':
                raise HTTPException(409, 'Already reviewed; start a new sample session to repeat')
            status = 'accepted_for_demo' if body.decision == 'accept' else 'rejected'
            app.state.db.execute('UPDATE sessions SET review=? WHERE id=?', (status, sid))
            audit(sid, 'review_' + status, {'note': body.note, 'procedure': PROCEDURE, 'media_sha256': record['media_sha256']})
            app.state.db.commit()
            return state(sid)

    @app.post('/api/decision')
    def decide(body: Decision, request: Request, response: Response):
        sid = session(request, response)
        require_media()
        with lock:
            current = state(sid)
            if body.status == 'approved' and current['review'] != 'accepted_for_demo':
                raise HTTPException(409, 'Review and accept this demonstration draft before approving the sample incident')
            if current['status'] != 'detected':
                raise HTTPException(409, 'Decision already recorded')
            app.state.db.execute('UPDATE sessions SET status=? WHERE id=?', (body.status, sid))
            audit(sid, 'incident_' + body.status, {'dispatch': False})
            app.state.db.commit()
            return state(sid)

    @app.get('/api/audit')
    def history(request: Request, response: Response):
        sid = session(request, response)
        with lock:
            rows = app.state.db.execute('SELECT payload,previous,hash FROM audit WHERE session=? ORDER BY id', (sid,)).fetchall()
            previous, consistent, entries = '', True, []
            for row in rows:
                consistent &= row['previous'] == previous and hashlib.sha256((previous + row['payload']).encode()).hexdigest() == row['hash']
                previous = row['hash']
                entries.append({**json.loads(row['payload']), 'hash': row['hash']})
            return {'consistent': bool(consistent), 'media_match': media_ok(), 'entries': entries,
                    'scope': 'Hash-linked sample log, not a digital signature or the private AICOS evidence system.'}

    @app.post('/api/reset')
    def reset(request: Request, response: Response):
        sid = session(request, response, new=True)
        return state(sid)

    app.mount('/assets', StaticFiles(directory=str(assets)), name='licensed-assets')
    return app
