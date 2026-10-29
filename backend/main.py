"""
AICOS Event Backend
-------------------
FastAPI service that receives detection events from edge workers or
camera-native analytics, stores them in SQLite, and streams them live
to connected dashboards over WebSocket.

Endpoints
    POST /api/events               ingest one event (X-AICOS-KEY header)
    GET  /api/events               query recent events
    POST /api/events/{id}/status   approve / dismiss / resolve an event
    GET  /api/stats                counters for dashboards
    GET  /api/health               liveness probe
    WS   /ws/events                live event stream (JSON messages)
    GET  /dashboard/live-ops.html  bundled live operations dashboard

Dev run:
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload
Production:
    docker compose up -d --build   (see docker-compose.yml at repo root)
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import sqlite3
import hmac
import io
import threading
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

import notify
import report
from licensing import LicenseManager, LicenseError
from certification import CameraCertificateRegistry, CameraCertificateError
from evidence import EvidenceManager, EvidenceError
from forensics import ForensicIndex
from camera_registry import CameraRegistry
from integrations import IntegrationBus
from auth_users import UserAuth, ROLES
from triage import TriageAssistant, TriageError
from versioning import AICOS_VERSION

DEVICE_API_KEY = os.environ.get("AICOS_API_KEY", "").strip()
OPERATOR_KEY = os.environ.get("AICOS_OPERATOR_KEY", "").strip()
VIEWER_KEY = os.environ.get("AICOS_VIEWER_KEY", "").strip() or OPERATOR_KEY
DB_PATH = os.environ.get("DB_PATH", "aicos.db")
WS_QUERY_TOKEN_ALLOWED = os.environ.get("AICOS_WS_QUERY_TOKEN_ALLOWED", "false").lower() in {"1", "true", "yes"}
MAX_REQUEST_BYTES = int(os.environ.get("AICOS_MAX_REQUEST_BYTES", "20000000"))
BEHIND_HTTPS = os.environ.get("AICOS_BEHIND_HTTPS", "false").lower() in {"1", "true", "yes"}
ALLOW_SHARED_OPERATOR_KEYS = os.environ.get("AICOS_ALLOW_SHARED_OPERATOR_KEYS", "true").lower() in {"1", "true", "yes"}

# Snapshots: the frame that triggered an alert, so an operator can verify
# it before acting. Stored as files next to the database, never in the DB.
SNAPSHOT_DIR = os.environ.get(
    "SNAPSHOT_DIR", os.path.join(os.path.dirname(os.path.abspath(DB_PATH)), "snapshots"))
SNAPSHOT_MAX_BYTES = int(os.environ.get("SNAPSHOT_MAX_BYTES", 3_000_000))
CLIP_MAX_BYTES = int(os.environ.get("CLIP_MAX_BYTES", 8_000_000))
SNAPSHOT_RETENTION_DAYS = float(os.environ.get("SNAPSHOT_RETENTION_DAYS", 14))
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

VALID_AGENTS = {"security", "emergency", "traffic", "infrastructure", "environmental"}
VALID_STATUS_UPDATES = {"approved", "dismissed", "resolved", "detected"}

# ---------------------------------------------------------------------------
# Storage (SQLite, single connection guarded by a lock — fine for pilot scale)
# ---------------------------------------------------------------------------

_db_lock = threading.RLock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
# Government deployments favor durability over a small write-speed gain.
# WAL improves concurrent read/write behavior; FULL synchronous mode keeps
# committed incidents resilient to process/host power loss.
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute("PRAGMA synchronous=FULL")
_conn.execute("PRAGMA foreign_keys=ON")
_conn.execute("PRAGMA busy_timeout=5000")


def _init_db() -> None:
    with _db_lock:
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL    NOT NULL,
                source_id  TEXT    NOT NULL,
                agent      TEXT    NOT NULL,
                type       TEXT    NOT NULL,
                severity   INTEGER NOT NULL,
                confidence REAL    NOT NULL,
                district   TEXT    DEFAULT '',
                lat        REAL    DEFAULT 0.0,
                lon        REAL    DEFAULT 0.0,
                status     TEXT    DEFAULT 'detected',
                extra      TEXT    DEFAULT '{}'
            )
            """
        )
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                ts       REAL    NOT NULL,
                action   TEXT    NOT NULL,
                note     TEXT    DEFAULT ''
            )
            """
        )
        for column in ("has_snapshot", "has_clip"):
            try:
                _conn.execute(
                    f"ALTER TABLE events ADD COLUMN {column} INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass   # column already present — normal on every restart
        try:
            _conn.execute("ALTER TABLE events ADD COLUMN event_uuid TEXT")
        except sqlite3.OperationalError:
            pass
        _conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_uuid ON events(event_uuid) "
            "WHERE event_uuid IS NOT NULL")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts)")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON events (source_id)")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_event ON audit (event_id)")
        _conn.commit()


_init_db()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["extra"] = json.loads(d.get("extra") or "{}")
    except (TypeError, ValueError):
        d["extra"] = {}
    return d


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EventIn(BaseModel):
    event_uuid: Optional[str] = Field(default=None, min_length=16, max_length=64, description="Idempotency key generated at the edge")
    source_id: str = Field(min_length=1, max_length=64, description="Camera or sensor id, e.g. CAM-001")
    agent: str = Field(description="One of: security, emergency, traffic, infrastructure, environmental")
    type: str = Field(min_length=1, max_length=120, description="Human-readable event type")
    severity: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    district: str = Field(default="", max_length=80)
    lat: float = 0.0
    lon: float = 0.0
    extra: dict = Field(default_factory=dict)
    snapshot: Optional[str] = Field(
        default=None,
        description="Base64 JPEG of the frame that triggered the event (optional)")
    clip: Optional[str] = Field(
        default=None,
        description="Base64 animated GIF of the seconds leading up to it (optional)")


class StatusIn(BaseModel):
    status: str


class TriageReviewIn(BaseModel):
    decision: str = Field(pattern='^(accept|reject)$')
    note: str = Field(default='', max_length=500)


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class UserCreateIn(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=12, max_length=256)
    role: str = Field(default="viewer", max_length=20)


class UserEnabledIn(BaseModel):
    enabled: bool


class CameraLinkIn(BaseModel):
    source_id: str = Field(min_length=1, max_length=64)
    next_source_id: str = Field(min_length=1, max_length=64)
    min_seconds: float = Field(default=0.0, ge=0.0, le=86400.0)
    max_seconds: float = Field(default=900.0, ge=1.0, le=86400.0)
    notes: str = Field(default="", max_length=240)
    bidirectional: bool = False


class ObservationIn(BaseModel):
    ts: float
    kind: str = Field(min_length=1, max_length=32)
    confidence: float = Field(ge=0.0, le=1.0)
    color: str = Field(default="", max_length=32)
    direction: str = Field(default="", max_length=32)
    track_id: str = Field(default="", max_length=96)
    appearance_key: str = Field(default="", max_length=128, description="Optional non-biometric coarse appearance correlation key")
    bbox: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0], min_length=4, max_length=4)
    extra: dict = Field(default_factory=dict)


class ObservationBatch(BaseModel):
    source_id: str = Field(min_length=1, max_length=64)
    observations: list[ObservationIn] = Field(min_length=1, max_length=1000)


class CameraHeartbeatIn(BaseModel):
    source_id: str = Field(min_length=1, max_length=64)
    name: str = Field(default="", max_length=120)
    district: str = Field(default="", max_length=120)
    worker_version: str = Field(default="", max_length=32)
    source_kind: str = Field(default="", max_length=32)
    capture_fps: float = Field(default=0.0, ge=0.0, le=240.0)
    analysis_fps: float = Field(default=0.0, ge=0.0, le=240.0)
    dropped_frames: int = Field(default=0, ge=0)
    reconnects: int = Field(default=0, ge=0)
    uptime_seconds: float = Field(default=0.0, ge=0.0)
    integrity_status: str = Field(default="unknown", max_length=40)
    integrity_score: float = Field(default=0.0, ge=0.0, le=100.0)
    integrity_reason: str = Field(default="", max_length=240)
    integrity_metrics: dict = Field(default_factory=dict)
    extra: dict = Field(default_factory=dict)


def _snapshot_path(event_id: int) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{int(event_id)}.jpg")


def _save_snapshot(event_id: int, b64: str) -> bool:
    """Decode and store the frame. Never raises — a bad image must not lose
    the event itself."""
    try:
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        raw = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError):
        return False
    if not raw or len(raw) > SNAPSHOT_MAX_BYTES:
        return False
    if raw[:2] != b"\xff\xd8":      # JPEG magic bytes
        return False
    try:
        with open(_snapshot_path(event_id), "wb") as fh:
            fh.write(raw)
    except OSError:
        return False
    return True


def _clip_path(event_id: int) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{int(event_id)}.gif")


def _save_clip(event_id: int, b64: str) -> bool:
    """Store the lead-up clip. Never raises — a bad clip must not lose the
    event or the snapshot."""
    try:
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        raw = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError):
        return False
    if not raw or len(raw) > CLIP_MAX_BYTES:
        return False
    if raw[:6] not in (b"GIF87a", b"GIF89a"):
        return False
    try:
        with open(_clip_path(event_id), "wb") as fh:
            fh.write(raw)
    except OSError:
        return False
    return True


_prune_counter = {"n": 0}


def _prune_snapshots() -> None:
    """Delete snapshots past the retention window. Runs occasionally, not
    on every event."""
    _prune_counter["n"] += 1
    if _prune_counter["n"] % 200 != 1:
        return
    cutoff = time.time() - SNAPSHOT_RETENTION_DAYS * 86400
    try:
        for name in os.listdir(SNAPSHOT_DIR):
            path = os.path.join(SNAPSHOT_DIR, name)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                continue
    except OSError:
        pass


# ---------------------------------------------------------------------------
# WebSocket hub
# ---------------------------------------------------------------------------


class Hub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def join(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def leave(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        message = json.dumps(payload)
        async with self._lock:
            targets = list(self._clients)
        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                await self.leave(ws)


hub = Hub()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AICOS Government Event Backend", version=AICOS_VERSION)


def _weak_secret(value: str) -> bool:
    return (not value) or value in {"change-me", "changeme", "password", "secret"} or len(value) < 24


def _assert_secure_config() -> None:
    if _weak_secret(DEVICE_API_KEY):
        raise RuntimeError("AICOS_API_KEY must be set to a strong 24+ character device key")
    # Government deployments should normally disable shared control-room keys
    # and use local RBAC accounts. Shared secrets remain available only for
    # pilots/legacy integrations and are validated when explicitly enabled.
    if ALLOW_SHARED_OPERATOR_KEYS and _weak_secret(OPERATOR_KEY):
        raise RuntimeError("AICOS_OPERATOR_KEY must be a strong 24+ character key when shared keys are enabled")
    if ALLOW_SHARED_OPERATOR_KEYS and VIEWER_KEY and _weak_secret(VIEWER_KEY):
        raise RuntimeError("AICOS_VIEWER_KEY must be 24+ characters when set")


_assert_secure_config()

# Same-origin dashboard access needs no CORS. Cross-origin operator consoles must
# be explicitly listed: AICOS_CORS_ORIGINS=https://ops.example,https://backup.example
_cors_origins = [x.strip() for x in os.environ.get("AICOS_CORS_ORIGINS", "").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-AICOS-KEY", "X-AICOS-OPERATOR-KEY"],
    allow_credentials=False,
)

_trusted_hosts = [x.strip() for x in os.environ.get("AICOS_TRUSTED_HOSTS", "").split(",") if x.strip()]
if _trusted_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_trusted_hosts)

@app.middleware("http")
async def _security_middleware(request: Request, call_next):
    try:
        content_length = int(request.headers.get("content-length", "0") or 0)
    except ValueError:
        content_length = 0
    if content_length > MAX_REQUEST_BYTES:
        return JSONResponse({"detail": "request body too large"}, status_code=413)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if request.url.path.startswith("/api/") or request.url.path.startswith("/ws/"):
        response.headers.setdefault("Cache-Control", "no-store")
    if BEHIND_HTTPS:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response

# Government-edition control planes. These are deliberately constructed at
# startup so an invalid required license/certificate/signing configuration
# fails closed instead of becoming a silent runtime weakness.
license_manager = LicenseManager()
certificate_registry = CameraCertificateRegistry()
evidence_manager = EvidenceManager(_conn, _db_lock, SNAPSHOT_DIR)
forensic_index = ForensicIndex(_conn, _db_lock, SNAPSHOT_DIR)
camera_registry = CameraRegistry(_conn, _db_lock)
integration_bus = IntegrationBus(_conn, _db_lock)
user_auth = UserAuth(_conn, _db_lock)
user_auth.purge_expired_sessions()
triage_assistant = TriageAssistant(_conn, _db_lock, SNAPSHOT_DIR, evidence_manager,
                                   os.environ.get('AICOS_SOP_FILE') or None)


def _same_secret(a: str, b: str) -> bool:
    return bool(a and b) and hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _bearer(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _request_token(request: Request) -> str:
    # HTTP credentials belong in headers, not URLs. Query-string credentials
    # are intentionally unsupported because they leak into logs/history.
    return request.headers.get("x-aicos-operator-key", "").strip() or _bearer(request)


def _session_user(request: Request) -> dict | None:
    # Resolve local RBAC sessions from the standard Bearer header. For
    # backwards-compatible dashboard clients we also accept the same random
    # session token in X-AICOS-OPERATOR-KEY, but never from a URL query
    # parameter. This keeps credentials out of reverse-proxy access logs.
    token = _bearer(request) or request.headers.get("x-aicos-operator-key", "").strip()
    return user_auth.resolve(token)


def require_viewer(request: Request) -> str:
    user = _session_user(request)
    if user and user.get("role") in {"viewer", "operator", "auditor", "admin"}:
        return f"user:{user['username']}:{user['role']}"
    if ALLOW_SHARED_OPERATOR_KEYS:
        token = _request_token(request)
        if _same_secret(token, OPERATOR_KEY):
            return "shared:operator"
        if _same_secret(token, VIEWER_KEY):
            return "shared:viewer"
    raise HTTPException(status_code=401, detail="authenticated viewer/operator session required")


def require_operator(request: Request) -> str:
    user = _session_user(request)
    if user and user.get("role") in {"operator", "admin"}:
        return f"user:{user['username']}:{user['role']}"
    if ALLOW_SHARED_OPERATOR_KEYS:
        token = _request_token(request)
        if _same_secret(token, OPERATOR_KEY):
            return "shared:operator"
    raise HTTPException(status_code=403 if (_bearer(request) or _request_token(request)) else 401, detail="operator role required")


def require_admin(request: Request) -> str:
    user = _session_user(request)
    if user and user.get("role") == "admin":
        return f"user:{user['username']}:admin"
    raise HTTPException(status_code=403 if _bearer(request) else 401, detail="admin role required")


def _require_device_key(key: str) -> None:
    if not _same_secret(key, DEVICE_API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-AICOS-KEY header")


def _known_sources() -> list[str]:
    with _db_lock:
        return [str(r["source_id"]) for r in _conn.execute(
            "SELECT DISTINCT source_id FROM events WHERE source_id IS NOT NULL"
        ).fetchall()]


def _authorize_event_policy(event: EventIn) -> None:
    try:
        license_manager.assert_event_allowed(
            event.source_id, event.type, event.agent, _known_sources()
        )
        certificate_registry.assert_event_allowed(event.source_id, event.type, event.agent)
    except (LicenseError, CameraCertificateError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _postprocess_event(event_id: int) -> dict:
    forensic_index.index_event(event_id)
    signed = None
    try:
        signed = evidence_manager.sign_event(event_id)
    except EvidenceError as exc:
        if evidence_manager.required:
            raise HTTPException(status_code=503, detail=f"evidence signing failed: {exc}") from exc
    return {"evidence": signed}


@app.get("/")
def root() -> dict:
    return {
        "service": "AICOS Government Event Backend",
        "version": AICOS_VERSION,
        "dashboard": "/dashboard/live-ops.html",
        "forensics": "/dashboard/forensics.html",
        "fleet": "/dashboard/fleet.html",
        "health": "/api/health",
        "license": "/api/license",
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "time": time.time(), "version": AICOS_VERSION, "edition": "Government"}


_login_attempts: dict[str, list[float]] = {}
_login_lock = threading.Lock()

def _login_rate_limit(ip: str) -> None:
    now = time.time(); cutoff = now - 60.0
    with _login_lock:
        attempts = [t for t in _login_attempts.get(ip, []) if t >= cutoff]
        if len(attempts) >= 8:
            raise HTTPException(status_code=429, detail="too many login attempts")
        attempts.append(now); _login_attempts[ip] = attempts

@app.post("/api/auth/login")
def auth_login(body: LoginIn, request: Request) -> dict:
    _login_rate_limit(request.client.host if request.client else "unknown")
    result = user_auth.login(body.username, body.password)
    if result is None:
        raise HTTPException(status_code=401, detail="invalid credentials")
    token, user = result
    return {"token": token, "token_type": "Bearer", "user": user}

@app.post("/api/auth/logout")
def auth_logout(request: Request, _role: str = Depends(require_viewer)) -> dict:
    token = _bearer(request)
    return {"revoked": user_auth.logout(token) if token else False}

@app.get("/api/auth/me")
def auth_me(request: Request, _role: str = Depends(require_viewer)) -> dict:
    user = _session_user(request)
    return {"actor": _role, "user": user}

@app.get("/api/admin/users")
def admin_users(_role: str = Depends(require_admin)) -> dict:
    return {"roles": sorted(ROLES), "users": user_auth.list_users()}

@app.post("/api/admin/users", status_code=201)
def admin_create_user(body: UserCreateIn, _role: str = Depends(require_admin)) -> dict:
    try:
        return user_auth.create_user(body.username, body.password, body.role)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="username already exists") from exc

@app.post("/api/admin/users/{username}/enabled")
def admin_enable_user(username: str, body: UserEnabledIn, _role: str = Depends(require_admin)) -> dict:
    if not user_auth.set_enabled(username, body.enabled):
        raise HTTPException(status_code=404, detail="user not found")
    return {"username": username, "enabled": body.enabled}


@app.get("/api/ready")
def ready(_role: str = Depends(require_viewer)) -> dict:
    checks = {}
    try:
        with _db_lock:
            _conn.execute("SELECT 1").fetchone()
        checks["database"] = True
    except Exception as exc:
        checks["database"] = False
        checks["database_error"] = str(exc)[:160]
    checks["snapshot_dir_writable"] = os.access(SNAPSHOT_DIR, os.W_OK)
    checks["license"] = license_manager.status()
    checks["camera_certificates"] = certificate_registry.summary()
    checks["evidence_signing_configured"] = evidence_manager.private_key is not None
    checks["local_operator_accounts"] = user_auth.count_users()
    checks["shared_operator_keys_enabled"] = ALLOW_SHARED_OPERATOR_KEYS
    checks["bootstrap_password_present"] = bool(os.environ.get("AICOS_BOOTSTRAP_ADMIN_PASSWORD", ""))
    checks["integration_bus"] = integration_bus.status()
    cameras = camera_registry.list()
    checks["camera_runtime"] = {k: cameras[k] for k in ("count", "online", "integrity_attention")}
    ok = bool(checks.get("database") and checks.get("snapshot_dir_writable"))
    return {"ok": ok, "time": time.time(), "version": AICOS_VERSION, "checks": checks}


@app.get("/api/metrics")
def metrics(_role: str = Depends(require_viewer)):
    cameras = camera_registry.list()
    with _db_lock:
        total = int(_conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"])
        last_hour = int(_conn.execute("SELECT COUNT(*) AS n FROM events WHERE ts >= ?", (time.time()-3600,)).fetchone()["n"])
        pending = int(_conn.execute("SELECT COUNT(*) AS n FROM events WHERE status = 'detected'").fetchone()["n"])
        manifests = int(_conn.execute("SELECT COUNT(*) AS n FROM evidence_manifests").fetchone()["n"])
    integration = integration_bus.status()
    lines = [
        "# TYPE aicos_events_total gauge", f"aicos_events_total {total}",
        "# TYPE aicos_events_last_hour gauge", f"aicos_events_last_hour {last_hour}",
        "# TYPE aicos_events_pending gauge", f"aicos_events_pending {pending}",
        "# TYPE aicos_cameras_total gauge", f"aicos_cameras_total {cameras['count']}",
        "# TYPE aicos_cameras_online gauge", f"aicos_cameras_online {cameras['online']}",
        "# TYPE aicos_camera_integrity_attention gauge", f"aicos_camera_integrity_attention {cameras['integrity_attention']}",
        "# TYPE aicos_evidence_manifests_total gauge", f"aicos_evidence_manifests_total {manifests}",
        "# TYPE aicos_integration_pending gauge", f"aicos_integration_pending {integration['pending']}",
        "# TYPE aicos_integration_retrying gauge", f"aicos_integration_retrying {integration['retrying']}",
        "# TYPE aicos_integration_abandoned gauge", f"aicos_integration_abandoned {integration['abandoned']}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/api/cameras/heartbeat", status_code=202)
def camera_heartbeat(body: CameraHeartbeatIn, x_aicos_key: str = Header(default="")) -> dict:
    _require_device_key(x_aicos_key)
    try:
        license_manager.assert_camera_allowed(body.source_id, _known_sources())
    except AttributeError:
        # v5-compatible LicenseManager implementations gate on events only.
        pass
    except LicenseError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    item = camera_registry.heartbeat(body.source_id, body.model_dump(exclude={"source_id"}))
    return {"accepted": True, "camera": item}


@app.get("/api/cameras")
def cameras(_role: str = Depends(require_viewer)) -> dict:
    data = camera_registry.list()
    certs = {c.get("camera_id"): c for c in certificate_registry.summary().get("certificates", [])}
    for cam in data["cameras"]:
        cam["certificate"] = certs.get(cam.get("source_id"))
    return data


@app.get("/api/cameras/{source_id}")
def camera_detail(source_id: str, _role: str = Depends(require_viewer)) -> dict:
    item = camera_registry.get(source_id)
    if item is None:
        raise HTTPException(status_code=404, detail="camera has not reported runtime health")
    certs = {c.get("camera_id"): c for c in certificate_registry.summary().get("certificates", [])}
    item["certificate"] = certs.get(source_id)
    return item


@app.post("/api/events", status_code=201)
async def ingest_event(event: EventIn, x_aicos_key: str = Header(default="")) -> dict:
    _require_device_key(x_aicos_key)
    if event.agent not in VALID_AGENTS:
        raise HTTPException(status_code=422, detail=f"agent must be one of {sorted(VALID_AGENTS)}")
    _authorize_event_policy(event)

    ts = time.time()
    with _db_lock:
        if event.event_uuid:
            existing = _conn.execute(
                "SELECT * FROM events WHERE event_uuid = ? LIMIT 1", (event.event_uuid,)).fetchone()
            if existing is not None:
                # A retry may arrive after the event row was committed but before
                # evidence media finished writing (for example, power loss). Heal
                # any missing evidence instead of treating the duplicate as done.
                has_snap = bool(existing["has_snapshot"])
                has_clip = bool(existing["has_clip"])
                if event.snapshot and not has_snap:
                    has_snap = _save_snapshot(existing["id"], event.snapshot)
                if event.clip and not has_clip:
                    has_clip = _save_clip(existing["id"], event.clip)
                _conn.execute(
                    "UPDATE events SET has_snapshot = ?, has_clip = ? WHERE id = ?",
                    (1 if has_snap else 0, 1 if has_clip else 0, existing["id"]))
                _conn.commit()
                post = _postprocess_event(int(existing["id"]))
                return {"id": existing["id"], "status": "stored", "duplicate": True, **post}
        cur = _conn.execute(
            """
            INSERT INTO events
                (event_uuid, ts, source_id, agent, type, severity, confidence,
                 district, lat, lon, status, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'detected', ?)
            """,
            (
                event.event_uuid, ts, event.source_id, event.agent, event.type, event.severity,
                event.confidence, event.district, event.lat, event.lon,
                json.dumps(event.extra),
            ),
        )
        event_id = cur.lastrowid
        _conn.execute(
            "INSERT INTO audit (event_id, ts, action, note) VALUES (?, ?, ?, ?)",
            (event_id, ts, "detected", "auto-ingested from " + event.source_id),
        )
        _conn.commit()

    has_snap = False
    has_clip = False
    if event.snapshot:
        has_snap = _save_snapshot(event_id, event.snapshot)
    if event.clip:
        has_clip = _save_clip(event_id, event.clip)
    if has_snap or has_clip:
        with _db_lock:
            _conn.execute(
                "UPDATE events SET has_snapshot = ?, has_clip = ? WHERE id = ?",
                (1 if has_snap else 0, 1 if has_clip else 0, event_id))
            _conn.commit()
    if event.snapshot or event.clip:
        _prune_snapshots()

    post = _postprocess_event(int(event_id))

    stored = {
        "id": event_id,
        "event_uuid": event.event_uuid,
        "ts": ts,
        "source_id": event.source_id,
        "agent": event.agent,
        "type": event.type,
        "severity": event.severity,
        "confidence": event.confidence,
        "district": event.district,
        "lat": event.lat,
        "lon": event.lon,
        "status": "detected",
        "extra": event.extra,
        "has_snapshot": 1 if has_snap else 0,
        "has_clip": 1 if has_clip else 0,
    }
    await hub.broadcast({"kind": "event", "event": stored})
    notify.notify("detected", stored)
    integration_bus.queue("detected", stored)
    evidence = post.get("evidence")
    evidence_summary = None if not evidence else {
        "revision": evidence.get("revision"), "manifest_hash": evidence.get("manifest_hash")
    }
    return {"id": event_id, "status": "stored", "evidence": evidence_summary}


@app.get("/api/events")
def list_events(
    _role: str = Depends(require_viewer),
    limit: int = Query(default=100, ge=1, le=500),
    agent: Optional[str] = Query(default=None),
    min_severity: int = Query(default=1, ge=1, le=5),
    since: Optional[float] = Query(default=None, description="Unix timestamp lower bound"),
) -> dict:
    sql = "SELECT * FROM events WHERE severity >= ?"
    params: list = [min_severity]
    if agent:
        sql += " AND agent = ?"
        params.append(agent)
    if since is not None:
        sql += " AND ts >= ?"
        params.append(since)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _db_lock:
        rows = _conn.execute(sql, params).fetchall()
    return {"events": [_row_to_dict(r) for r in rows]}


@app.post("/api/events/{event_id}/status")
async def update_status(
    event_id: int, body: StatusIn, _role: str = Depends(require_operator)
) -> dict:
    if body.status not in VALID_STATUS_UPDATES:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(VALID_STATUS_UPDATES)}")

    with _db_lock:
        cur = _conn.execute("UPDATE events SET status = ? WHERE id = ?", (body.status, event_id))
        if cur.rowcount == 0:
            _conn.commit()
            raise HTTPException(status_code=404, detail="event not found")
        _conn.execute(
            "INSERT INTO audit (event_id, ts, action, note) VALUES (?, ?, ?, ?)",
            (event_id, time.time(), body.status, f"operator decision by {_role}"),
        )
        _conn.commit()

    forensic_index.index_event(event_id)
    # Operator decisions become a new signed evidence revision so the
    # chain-of-custody covers human review, not only the original media.
    try:
        evidence_manager.sign_event(event_id)
    except EvidenceError as exc:
        if evidence_manager.required:
            raise HTTPException(status_code=503, detail=f"evidence signing failed: {exc}") from exc
    await hub.broadcast({"kind": "status", "id": event_id, "status": body.status})

    # An operator confirming an alert is the dispatch signal — this is where
    # responders are told, never automatically on detection.
    if body.status == "approved":
        with _db_lock:
            row = _conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is not None:
            approved_event = _row_to_dict(row)
            notify.notify("approved", approved_event)
            integration_bus.queue("approved", approved_event)
    return {"id": event_id, "status": body.status}


@app.get("/api/stats")
def stats(_role: str = Depends(require_viewer)) -> dict:
    with _db_lock:
        total = _conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        by_agent = {
            r["agent"]: r["n"]
            for r in _conn.execute("SELECT agent, COUNT(*) AS n FROM events GROUP BY agent")
        }
        by_status = {
            r["status"]: r["n"]
            for r in _conn.execute("SELECT status, COUNT(*) AS n FROM events GROUP BY status")
        }
        last_hour = _conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE ts >= ?", (time.time() - 3600,)
        ).fetchone()["n"]
    return {"total": total, "last_hour": last_hour, "by_agent": by_agent, "by_status": by_status}


@app.get("/api/events/{event_id}/snapshot")
def event_snapshot(event_id: int, _role: str = Depends(require_viewer)):
    """The frame that triggered the alert. Viewer/operator authentication
    is required because evidence media is sensitive even on a private LAN."""
    path = _snapshot_path(event_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="no snapshot for this event")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/events/{event_id}/clip")
def event_clip(event_id: int, _role: str = Depends(require_viewer)):
    """The seconds leading up to the alert, as an animated GIF."""
    path = _clip_path(event_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="no clip for this event")
    return FileResponse(path, media_type="image/gif")


@app.get("/api/events/{event_id}/report")
def incident_report(
    event_id: int,
    format: str = Query(default="json"),
    _role: str = Depends(require_viewer),
):
    data = report.gather(_conn, _db_lock, event_id)
    if data is None:
        raise HTTPException(status_code=404, detail="event not found")
    text = report.render_text(data)
    text = report.maybe_llm_polish(text, data)
    if format == "text":
        return PlainTextResponse(text)
    return {"id": event_id, "generated_at": time.time(), "report": text}


def _triage_call(action):
    try:
        return action()
    except TriageError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    except EvidenceError as exc:
        raise HTTPException(status_code=503, detail='Evidence signing failed; inspect the incident audit') from exc


@app.get('/api/triage/procedures')
def triage_procedures(_role: str = Depends(require_viewer)):
    return _triage_call(lambda: {'procedures': triage_assistant.procedures()})


@app.get('/api/events/{event_id}/triage')
def list_triage(event_id: int, _role: str = Depends(require_viewer)):
    return _triage_call(lambda: {'drafts': triage_assistant.list(event_id)})


@app.post('/api/events/{event_id}/triage', status_code=201)
def create_triage(event_id: int, _role: str = Depends(require_operator)):
    return _triage_call(lambda: triage_assistant.create(event_id, _role))


@app.get('/api/triage/{draft_id}')
def get_triage(draft_id: int, _role: str = Depends(require_viewer)):
    return _triage_call(lambda: triage_assistant.get(draft_id))


@app.post('/api/triage/{draft_id}/review')
def review_triage(draft_id: int, body: TriageReviewIn, _role: str = Depends(require_operator)):
    return _triage_call(lambda: triage_assistant.review(draft_id, body.decision, _role, body.note))


@app.get("/api/license")
def license_status(_role: str = Depends(require_viewer)) -> dict:
    return license_manager.status()


@app.get("/api/integrations/status")
def integrations_status(_role: str = Depends(require_viewer)) -> dict:
    return integration_bus.status()


@app.post("/api/integrations/retry")
def integrations_retry(_role: str = Depends(require_operator)) -> dict:
    delivered = integration_bus.flush_once(limit=100)
    return {"delivered_now": delivered, **integration_bus.status()}


@app.get("/api/certificates")
def certificate_status(_role: str = Depends(require_viewer)) -> dict:
    return certificate_registry.summary()


@app.get("/api/events/{event_id}/evidence")
def event_evidence(event_id: int, _role: str = Depends(require_viewer)) -> dict:
    item = evidence_manager.latest(event_id)
    if item is None:
        raise HTTPException(status_code=404, detail="no signed evidence manifest")
    return {"event_id": event_id, "manifest": item, "verification": evidence_manager.verify(event_id)}


@app.get("/api/events/{event_id}/evidence/verify")
def verify_event_evidence(event_id: int, _role: str = Depends(require_viewer)) -> dict:
    result = evidence_manager.verify(event_id)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result)
    return result


@app.get("/api/evidence/chain/verify")
def verify_evidence_chain(_role: str = Depends(require_viewer)) -> dict:
    result = evidence_manager.verify_chain()
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result)
    return result


@app.get("/api/events/{event_id}/evidence/export")
def export_event_evidence(event_id: int, _role: str = Depends(require_viewer)):
    try:
        blob = evidence_manager.export_package(event_id)
    except EvidenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    headers = {"Content-Disposition": f'attachment; filename="AICOS-EVIDENCE-{event_id:06d}.zip"'}
    return StreamingResponse(io.BytesIO(blob), media_type="application/zip", headers=headers)


@app.post("/api/forensics/observations", status_code=201)
def ingest_forensic_observations(batch: ObservationBatch, x_aicos_key: str = Header(default="")) -> dict:
    _require_device_key(x_aicos_key)
    try:
        license_manager.assert_feature("forensic_search")
        certificate_registry.assert_camera_current(batch.source_id)
    except (LicenseError, CameraCertificateError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    inserted = forensic_index.add_observations(
        batch.source_id, [o.model_dump() for o in batch.observations]
    )
    return {"source_id": batch.source_id, "accepted": len(batch.observations), "inserted": inserted}


@app.get("/api/forensics/objects")
def forensic_object_search(
    _role: str = Depends(require_viewer),
    since: Optional[float] = Query(default=None),
    until: Optional[float] = Query(default=None),
    source_id: str = Query(default="", max_length=64),
    kind: str = Query(default="", max_length=32),
    color: str = Query(default="", max_length=32),
    direction: str = Query(default="", max_length=32),
    track_id: str = Query(default="", max_length=96),
    appearance_key: str = Query(default="", max_length=128),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0, le=100000),
) -> dict:
    try:
        license_manager.assert_feature("forensic_search")
    except LicenseError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return forensic_index.search_observations(
        since=since, until=until, source_id=source_id, kind=kind, color=color,
        direction=direction, track_id=track_id, appearance_key=appearance_key,
        min_confidence=min_confidence, limit=limit, offset=offset
    )


@app.get("/api/forensics/search")
def forensic_search(
    _role: str = Depends(require_viewer),
    q: str = Query(default="", max_length=200),
    since: Optional[float] = Query(default=None),
    until: Optional[float] = Query(default=None),
    source_id: str = Query(default="", max_length=64),
    district: str = Query(default="", max_length=80),
    agent: str = Query(default="", max_length=32),
    event_type: str = Query(default="", max_length=120),
    status: str = Query(default="", max_length=32),
    min_severity: int = Query(default=1, ge=1, le=5),
    max_severity: int = Query(default=5, ge=1, le=5),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    has_snapshot: Optional[bool] = Query(default=None),
    has_clip: Optional[bool] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0, le=100000),
) -> dict:
    if min_severity > max_severity:
        raise HTTPException(status_code=422, detail="min_severity cannot exceed max_severity")
    try:
        license_manager.assert_feature("forensic_search")
    except LicenseError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return forensic_index.search(
        q=q, since=since, until=until, source_id=source_id, district=district,
        agent=agent, event_type=event_type, status=status, min_severity=min_severity,
        max_severity=max_severity, min_confidence=min_confidence,
        has_snapshot=has_snapshot, has_clip=has_clip, limit=limit, offset=offset,
    )


@app.post("/api/forensics/camera-links")
def forensic_camera_link(body: CameraLinkIn, _role: str = Depends(require_operator)) -> dict:
    try:
        license_manager.assert_feature("forensic_search")
        return forensic_index.set_camera_link(
            body.source_id, body.next_source_id, min_seconds=body.min_seconds,
            max_seconds=body.max_seconds, notes=body.notes, bidirectional=body.bidirectional,
        )
    except LicenseError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/forensics/camera-links")
def forensic_camera_links(_role: str = Depends(require_viewer)) -> dict:
    try:
        license_manager.assert_feature("forensic_search")
    except LicenseError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return forensic_index.list_camera_links()


@app.get("/api/forensics/journey/{observation_id}")
def forensic_journey(observation_id: int, _role: str = Depends(require_viewer),
                     limit: int = Query(default=50, ge=1, le=200)) -> dict:
    try:
        license_manager.assert_feature("forensic_search")
    except LicenseError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return forensic_index.journey_candidates(observation_id, limit=limit)


@app.get("/api/forensics/similar/{event_id}")
def forensic_similar(
    event_id: int, _role: str = Depends(require_viewer),
    limit: int = Query(default=20, ge=1, le=100),
    max_distance: int = Query(default=20, ge=0, le=64),
) -> dict:
    try:
        license_manager.assert_feature("forensic_search")
    except LicenseError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return forensic_index.similar(event_id, limit=limit, max_distance=max_distance)


def _ws_token_authorized(token: str) -> bool:
    if not token:
        return False
    user = user_auth.resolve(token)
    if user and user.get("role") in {"viewer", "operator", "auditor", "admin"}:
        return True
    if ALLOW_SHARED_OPERATOR_KEYS:
        return _same_secret(token, OPERATOR_KEY) or _same_secret(token, VIEWER_KEY)
    return False


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    # Government default: credentials do not appear in a URL where reverse
    # proxies/browser history may log them. The first WebSocket frame accepts
    # either a local RBAC session token or (only when explicitly allowed) a
    # legacy shared operator/viewer key.
    token = ws.query_params.get("token", "") if WS_QUERY_TOKEN_ALLOWED else ""
    await ws.accept()
    if not _ws_token_authorized(token):
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
            try:
                msg = json.loads(raw)
                token = str(msg.get("token", "")) if isinstance(msg, dict) else ""
            except Exception:
                token = raw.strip()
        except Exception:
            await ws.close(code=4401)
            return
    if not _ws_token_authorized(token):
        await ws.close(code=4401)
        return
    await hub.join(ws)
    try:
        with _db_lock:
            rows = _conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT 50").fetchall()
        await ws.send_text(json.dumps({"kind": "snapshot", "events": [_row_to_dict(r) for r in rows]}))
        while True:
            # Clients may send pings / keepalives; content is ignored.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await hub.leave(ws)


def find_dashboard_dir(base: str = None):
    """Locate the dashboard folder.

    Under Docker the Dockerfile copies it next to this file; running from
    a source checkout it sits one level up beside backend/. Checking both
    means `uvicorn main:app` works without Docker, which is how most people
    try it first.
    """
    here = base or os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "dashboard"),
                      os.path.join(os.path.dirname(here), "dashboard")):
        if os.path.isdir(candidate):
            return candidate
    return None


_dash_dir = find_dashboard_dir()
if _dash_dir:
    app.mount("/dashboard", StaticFiles(directory=_dash_dir, html=True), name="dashboard")
else:
    import logging
    logging.getLogger("uvicorn.error").warning(
        "dashboard folder not found — /dashboard will return 404")
