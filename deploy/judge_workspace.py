"""Local three-input judge workspace using unchanged private detection/evidence code.

Separate process and fresh project-local DB. Not the deployed public replay sample.
Never mount the production API here: anonymous access is scoped to one demo run.
"""
import asyncio
import base64
from collections import deque
from contextlib import asynccontextmanager
from io import BytesIO
import json
import hashlib
import math
import os
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
os.environ['AICOS_DEMO_RUNTIME_DIR'] = str(ROOT / '.runtime/judge-workspace-20261026')
from deploy import free_demo as isolated
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from pydantic import model_validator
from deploy.judge_capabilities import IDS, catalog
from starlette.middleware.trustedhost import TrustedHostMiddleware

core = isolated.core
COOKIE = 'aicos_judge_run'
MAX_BODY = 700_000
MAX_SECONDS = 90
MAX_FRAMES = 270


def inference_interpreter(root=ROOT, platform=None, executable=None):
    platform = os.name if platform is None else platform
    relative = '.venv-inference/Scripts/python.exe' if platform == 'nt' else '.venv-inference/bin/python'
    candidate = Path(root) / relative
    if candidate.is_file():
        return candidate
    if platform != 'nt':
        # Hosted Linux uses one environment containing API and CPU inference deps.
        return Path(executable or sys.executable)
    raise RuntimeError('Existing project-local .venv-inference is required')


def workspace_hosts(hostname=None):
    hostname = isolated.hostname if hostname is None else hostname
    hosts = ['127.0.0.1', 'localhost', 'testserver']
    if hostname:
        import re
        if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?', hostname):
            raise RuntimeError('Invalid external workspace hostname')
        hosts.append(hostname)
    return hosts


class Worker:
    def __init__(self):
        self.lock = threading.Lock()
        self.lines = queue.Queue()
        self.process = None

    def start(self):
        python = inference_interpreter()
        self.log = (isolated.BOOT / 'inference-stderr.txt').open('w', encoding='utf-8')
        env = dict(os.environ, TEMP=str(isolated.BOOT), TMP=str(isolated.BOOT),
                   YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(isolated.BOOT / 'yolo'),
                   MPLCONFIGDIR=str(isolated.BOOT / 'matplotlib'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        if isolated.hostname:
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for name in ('yolo', 'matplotlib'):
            (isolated.BOOT / name).mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen([str(python), '-I', '-B', str(ROOT / 'deploy/judge_inference_worker.py')],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None if isolated.hostname else self.log, text=True, encoding='utf-8', bufsize=1,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        def read():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        threading.Thread(target=read, daemon=True).start()
        try:
            ready = self._read(180 if isolated.hostname else 45)
            if not ready.get('ready'):
                raise RuntimeError('Inference worker did not become ready')
        except Exception:
            self.close()
            raise

    def _read(self, timeout):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError('Inference deadline exceeded; restart this workspace after inspecting its log')
        if line is None:
            raise RuntimeError('Inference worker exited; inspect project-local inference-stderr.txt')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def infer(self, session, image, seconds, settings=None):
        with self.lock:
            if not self.process or self.process.poll() is not None:
                raise RuntimeError('Inference worker is unavailable')
            self.process.stdin.write(json.dumps(dict(session=session, image=image, seconds=seconds, settings=settings)) + '\n')
            self.process.stdin.flush()
            return self._read(45)

    def drop(self, session):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.stdin.write(json.dumps(dict(session=session, op='drop')) + '\n')
                self.process.stdin.flush()
                self._read(5)

    def close(self):
        if self.process and self.process.poll() is None:
            # Windows venv python.exe is a launcher for the base interpreter.
            # EOF lets BOTH exit normally; killing only the launcher can orphan ML.
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    subprocess.run(['taskkill.exe', '/PID', str(self.process.pid), '/T', '/F'],
                                   capture_output=True, timeout=10)
                else:
                    self.process.kill()
                self.process.wait(timeout=5)
        if hasattr(self, 'log'):
            self.log.close()


class RunIn(BaseModel):
    mode: str = Field(pattern='^(scenario|file|camera)$')
    detectors: list[str] = Field(default_factory=lambda:['person_down'],min_length=1,max_length=len(IDS))
    region: list[float] | None = None
    crowd_warn: int = Field(default=8,ge=2,le=50)
    loiter_seconds: float = Field(default=120,ge=5,le=600,allow_inf_nan=False)
    stopped_seconds: float = Field(default=45,ge=5,le=600,allow_inf_nan=False)
    direction: str = Field(default='',pattern='^(|left|right|up|down)$')
    camera_name: str = Field(default='Current input', min_length=1, max_length=80)
    location_label: str = Field(default='', max_length=120)
    latitude: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    longitude: float | None = Field(default=None, ge=-180, le=180, allow_inf_nan=False)

    @model_validator(mode='after')
    def configured(self):
        if not self.camera_name.strip():
            raise ValueError('Camera name cannot be blank')
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError('Both latitude and longitude are required together')
        if len(set(self.detectors))!=len(self.detectors) or any(k not in IDS for k in self.detectors):
            raise ValueError('Unknown or duplicate detector')
        if self.region is not None:
            if len(self.region)!=4 or not all(math.isfinite(x) and 0<=x<=1 for x in self.region):
                raise ValueError('Region must be four finite normalized coordinates')
            if self.region[0]>=self.region[2] or self.region[1]>=self.region[3]:
                raise ValueError('Region must have positive width and height')
        if set(self.detectors)&{'intrusion','loitering','vehicles'} and self.region is None:
            raise ValueError('Zone and traffic detectors require an explicit test region')
        if self.direction and 'vehicles' not in self.detectors:
            raise ValueError('Direction requires traffic selection')
        return self


class FrameIn(BaseModel):
    image: str = Field(min_length=8, max_length=680_000)
    seconds: float = Field(ge=0, le=MAX_SECONDS, allow_inf_nan=False)


class ReviewIn(BaseModel):
    decision: str = Field(pattern='^(accept|reject)$')
    note: str = Field(min_length=1, max_length=500)


class DecisionIn(BaseModel):
    status: str = Field(pattern='^(approved|dismissed)$')


def create_app(worker=None):
    engine = worker or Worker()
    sessions = {}
    lock = threading.RLock()
    gates = threading.Semaphore(1)
    rate = {}

    @asynccontextmanager
    async def lifespan(app):
        async with core.app.router.lifespan_context(core.app):
            await asyncio.to_thread(engine.start)
            try:
                yield
            finally:
                await asyncio.to_thread(engine.close)

    app = FastAPI(title='AICOS Safety Operations', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=workspace_hosts())

    @app.middleware('http')
    async def boundary(request, call_next):
        if request.method not in ('GET', 'HEAD', 'POST'):
            return JSONResponse({'detail': 'Method unavailable'}, status_code=405)
        if request.method == 'POST':
            origin = request.headers.get('origin')
            if origin and (urlsplit(origin).scheme not in ('http', 'https') or
                           urlsplit(origin).netloc != request.headers.get('host')):
                return JSONResponse({'detail': 'Cross-origin writes disabled'}, status_code=403)
            parts, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BODY:
                    return JSONResponse({'detail': 'Frame request too large'}, status_code=413)
                parts.append(chunk)
            request._body = b''.join(parts)
            now = int(time.monotonic() // 60)
            peer = request.client.host if request.client else 'unknown'
            with lock:
                for key in list(rate):
                    if key[1] != now:
                        del rate[key]
                key = (peer, now)
                if rate.get(key, 0) >= 240:
                    return JSONResponse({'detail': 'Request limit reached; wait one minute'}, status_code=429)
                rate[key] = rate.get(key, 0) + 1
        response = await call_next(request)
        response.headers.update({'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Referrer-Policy': 'strict-origin-when-cross-origin', 'X-Frame-Options': 'DENY',
            'Permissions-Policy': 'camera=(self), microphone=(), geolocation=(self)'})
        return response

    def current(request):
        sid = request.cookies.get(COOKIE, '')
        with lock:
            run = sessions.get(sid)
        if not run or time.monotonic() - run['created'] > 600:
            raise HTTPException(401, 'Start a new analysis run; this session is missing or expired')
        return sid, run

    def event(request):
        sid, run = current(request)
        requested = request.query_params.get('event_id')
        selected = next((r for r in run['incidents'] if str(r['event_id'])==requested),None) if requested else (run['incidents'][0] if run['incidents'] else None)
        if not selected:
            raise HTTPException(409, 'No detector alert in this run')
        return sid, dict(selected, mutex=run['mutex'])

    @app.get('/')
    def home():
        return FileResponse(ROOT / 'dashboard/judge-workspace.html')

    @app.get('/workspace-assets/{name}')
    def workspace_asset(name: str):
        if name not in ('leaflet.js', 'leaflet.css', 'LEAFLET-LICENSE.txt', 'location-map.js', 'location-map.css', 'camera-wall.js', 'camera-wall.css'):
            raise HTTPException(404)
        return FileResponse(ROOT / 'dashboard/map-assets' / name)

    @app.get('/health')
    def health():
        ok = not isinstance(engine, Worker) or bool(engine.process and engine.process.poll() is None)
        return JSONResponse({'ok': ok, 'mode': 'local-fresh-inference', 'inputs': ['scenario', 'file', 'camera'],
                'capability': 'multiple development detectors; human review required', 'capabilities':catalog(), 'dispatch': False},
                status_code=200 if ok else 503)

    @app.get('/judge/api/capabilities')
    def capabilities():
        return {'capabilities':catalog(),'public_inference':False,'max_seconds':MAX_SECONDS}

    @app.get('/judge/assets/{name}')
    def asset(name: str):
        if name not in ('controlled-fall.mp4', 'walking.mp4', 'LICENSE.txt', 'credits.html'):
            raise HTTPException(404)
        if name != 'credits.html':
            manifest = json.loads((isolated.ASSETS / 'seed.json').read_text(encoding='utf-8'))
            isolated.verified_asset(name, manifest['files'][name])
        return FileResponse(isolated.ASSETS / name)

    @app.post('/judge/api/run')
    def start(body: RunIn, request: Request, response: Response):
        with lock:
            expired = [sid for sid, run in sessions.items() if time.monotonic() - run['created'] > 600]
            prior = request.cookies.get(COOKIE, '')
            if prior in sessions:
                expired.append(prior)
            for sid in set(expired):
                old = sessions[sid]
                if not old['mutex'].acquire(blocking=False):
                    raise HTTPException(409, 'Previous inference still running')
                try:
                    engine.drop(sid)
                    del sessions[sid]
                finally:
                    old['mutex'].release()
            if len(sessions) >= 8:
                raise HTTPException(503, 'Eight active analysis runs; retry after one expires')
            sid = secrets.token_hex(24)
            sessions[sid] = dict(mode=body.mode, created=time.monotonic(), first_wall=None,
                last=-1, frames=0, event_id=None, draft_id=None, decision='detected',
                ring=deque(maxlen=18), mutex=threading.Lock(), finished=False,
                settings=body.model_dump(), incidents=[],alerted=set())
            response.set_cookie(COOKIE, sid, httponly=True, samesite='strict', max_age=600,
                                secure=bool(isolated.hostname))
            return {'mode': body.mode, 'max_seconds': MAX_SECONDS, 'max_frames': MAX_FRAMES,
                    'fresh_detection': True, 'settings':body.model_dump(), 'note': 'No stored incident has been seeded'}

    @app.post('/judge/api/frame')
    async def frame(body: FrameIn, request: Request):
        sid, run = current(request)
        if not run['mutex'].acquire(blocking=False):
            raise HTTPException(409, 'One frame is already in progress; do not retry it')
        acquired = gates.acquire(blocking=False)
        try:
            if not acquired:
                raise HTTPException(429, 'Detector is busy with another analysis run')
            if run['finished'] or run['frames'] >= MAX_FRAMES:
                raise HTTPException(409, 'Run finished; start a new run')
            seconds = body.seconds
            if run['mode'] == 'camera':
                if run['first_wall'] is None:
                    run['first_wall'] = time.monotonic()
                seconds = time.monotonic() - run['first_wall']
            if seconds > MAX_SECONDS or not math.isfinite(seconds) or seconds <= run['last']:
                raise HTTPException(422, 'Non-increasing or out-of-range frame time')
            if run['last'] >= 0 and seconds - run['last'] < .25:
                raise HTTPException(422, 'Maximum analysis rate is four frames per media second')
            try:
                raw = base64.b64decode(body.image, validate=True)
                with Image.open(BytesIO(raw)) as image:
                    if image.format not in ('JPEG', 'PNG') or max(image.size) > 640 or min(image.size) < 32:
                        raise ValueError('Only image frames from 32 to 640 pixels per dimension are accepted')
                    image.verify()
            except Exception as exc:
                raise HTTPException(422, 'Invalid bounded PNG/JPEG frame') from exc
            # Advance before inference: a lost POST reply cannot safely be retried.
            run['last'] = seconds
            run['frames'] += 1
            run['ring'].append((seconds, raw))
            try:
                found = await asyncio.to_thread(engine.infer, sid, body.image, seconds, run['settings'])
            except RuntimeError as exc:
                run['finished'] = True
                raise HTTPException(503, str(exc)) from exc
            for detected in found['events']:
                key = (detected.get('extra',{}).get('judge_capability','person_down'),detected['type'])
                if key in run['alerted'] or len(run['incidents'])>=24:
                    continue
                frames, durations = [], []
                for index, (stamp, data) in enumerate(run['ring']):
                    with Image.open(BytesIO(data)) as image:
                        frames.append(image.convert('RGB'))
                    following = list(run['ring'])[index + 1][0] if index + 1 < len(run['ring']) else stamp + .333
                    durations.append(max(20, min(2000, int((following - stamp) * 1000))))
                gif = BytesIO()
                frames[0].save(gif, format='GIF', save_all=True, append_images=frames[1:], duration=durations, loop=0)
                extra = dict(detected['extra'], live_inference=True, input_mode=run['mode'],
                    capture_seconds=round(seconds, 4), model_sha256=detected.get('extra',{}).get('model_sha256',found['model_sha256']),
                    model_hashes=found.get('model_hashes',{}),judge_settings=run['settings'],
                    time_source='server elapsed camera time' if run['mode'] == 'camera' else 'browser-reported media time; not authenticated',
                    location_source='operator-configured camera position' if run['settings']['latitude'] is not None else 'unconfigured',
                    camera_name=run['settings']['camera_name'], location_label=run['settings']['location_label'],
                    location_configured=run['settings']['latitude'] is not None,
                    location_precision='camera position, not exact incident position', demonstration=True, dispatch=False)
                # Preserve lossless pixels for inference. The existing evidence store
                # requires JPEG snapshots, so convert only AFTER the model decision.
                snapshot = BytesIO()
                with Image.open(BytesIO(raw)) as image:
                    image.convert('RGB').save(snapshot, format='JPEG', quality=95)
                extra['inference_frame_sha256'] = hashlib.sha256(raw).hexdigest()
                extra['snapshot_derivation'] = 'JPEG quality 95 from the lossless inference frame'
                result = await core.ingest_event(core.EventIn(source_id='AICOS-' + sid[:12],
                    lat=run['settings']['latitude'] if run['settings']['latitude'] is not None else 0.0,
                    lon=run['settings']['longitude'] if run['settings']['longitude'] is not None else 0.0,
                    district=run['settings']['location_label'][:80],
                    event_uuid=secrets.token_hex(16), agent=detected['agent'], type=detected['type'],
                    severity=detected['severity'], confidence=detected['confidence'], extra=extra,
                    snapshot=base64.b64encode(snapshot.getvalue()).decode(),
                    clip=base64.b64encode(gif.getvalue()).decode()), x_aicos_key=isolated.device_key)
                run['event_id'] = result['id']
                draft = core.triage_assistant.create(result['id'], 'anonymous-demo:' + sid[:12])
                run['draft_id'] = draft['id']
                run['incidents'].append(dict(event_id=result['id'],draft_id=draft['id'],decision='detected',
                    type=detected['type'],capability=key[0], location={
                        'latitude':run['settings']['latitude'],'longitude':run['settings']['longitude'],
                        'camera_name':run['settings']['camera_name'],'label':run['settings']['location_label']}))
                run['alerted'].add(key)
            return {'frames': run['frames'], 'seconds': round(seconds, 4),
                    'observation': found['observation'], 'alert': bool(run['event_id']),
                    'detectors':found.get('detectors',[]),'incidents':[{k:r[k] for k in ('event_id','type','capability','location')} for r in run['incidents']],
                    'processing': 'fresh model inference', 'event_id': run['event_id']}
        finally:
            if acquired:
                gates.release()
            run['mutex'].release()

    @app.post('/judge/api/finish')
    def finish(request: Request):
        sid, run = current(request)
        if not run['mutex'].acquire(blocking=False):
            raise HTTPException(409, 'Wait for the current frame')
        try:
            engine.drop(sid)
            run['finished'] = True
            run['ring'].clear()
            return {'finished': True, 'frames': run['frames'], 'alert': bool(run['event_id']),
                    'coverage': 'Sampled frames only; no alert does not establish safety'}
        finally:
            run['mutex'].release()

    @app.get('/judge/api/incident')
    def incident(request: Request):
        _, run = event(request)
        draft = core.triage_assistant.get(run['draft_id'])
        with core._db_lock:
            row = core._conn.execute('SELECT * FROM events WHERE id=?', (run['event_id'],)).fetchone()
            audit = [dict(r) for r in core._conn.execute('SELECT ts,action,note FROM audit WHERE event_id=? ORDER BY id', (run['event_id'],))]
        return {'event': core._row_to_dict(row), 'draft': draft,
                'verification': core.evidence_manager.verify(run['event_id']), 'audit': audit,
                'actor_scope': 'anonymous demo session, not a verified person', 'dispatch': False}

    @app.get('/judge/api/incidents')
    def incidents(request:Request):
        _,run=current(request)
        return {'incidents':[{k:r[k] for k in ('event_id','type','capability','location')} for r in run['incidents']]}

    @app.get('/judge/api/media/{kind}')
    def media(kind: str, request: Request):
        _, run = event(request)
        if kind not in ('snapshot', 'clip'):
            raise HTTPException(404)
        return (core.event_snapshot if kind == 'snapshot' else core.event_clip)(run['event_id'], _role='demo-viewer')

    @app.post('/judge/api/review')
    def review(body: ReviewIn, request: Request):
        sid, run = event(request)
        if not body.note.strip():
            raise HTTPException(422, 'Record a review note')
        if not run['mutex'].acquire(blocking=False):
            raise HTTPException(409, 'Another action is in progress')
        try:
            return core._triage_call(lambda: core.triage_assistant.review(run['draft_id'], body.decision,
                'anonymous-demo:' + sid[:12], body.note.strip()))
        finally:
            run['mutex'].release()

    @app.post('/judge/api/decision')
    async def decision(body: DecisionIn, request: Request):
        sid, run = event(request)
        if not run['mutex'].acquire(blocking=False):
            raise HTTPException(409, 'Another action is in progress')
        try:
            if run['decision'] != 'detected':
                raise HTTPException(409, 'Decision already recorded')
            draft = core.triage_assistant.get(run['draft_id'])
            if body.status == 'approved' and draft['review_status'] not in ('accepted', 'accepted_for_demo'):
                raise HTTPException(409, 'Accept the review draft before incident approval')
            result = await core.update_status(run['event_id'], core.StatusIn(status=body.status),
                _role='anonymous-demo:' + sid[:12])
            # Update the stored row, not the scoped copy returned by event().
            _, active = current(request)
            next(r for r in active['incidents'] if r['event_id']==run['event_id'])['decision']=body.status
            return result
        finally:
            run['mutex'].release()

    return app
