"""
AICOS Edge Worker
-----------------
Connects to one or more RTSP camera streams, runs a detector on sampled
frames, and posts events to the AICOS backend.

Ships with a motion-based demo detector so the full pipeline (camera ->
edge -> backend -> dashboard) runs end-to-end with any RTSP camera or
webcam BEFORE production models are installed. Production detection
models (violence / weapon / crash / crowd analytics from your model
vendor or your own trained models) plug in through the same Detector
interface below.

Usage:
    export BACKEND_URL=http://192.168.1.50:8080
    export AICOS_API_KEY=<strong-32-plus-character-device-key>
    python rtsp_worker.py --config ../config/cameras.yaml

Tip: a laptop webcam works for a desk test — set rtsp_url to 0 in the
config (the worker treats integer strings as local device indexes).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sqlite3
import uuid
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
import requests
import yaml

from versioning import AICOS_VERSION

log = logging.getLogger("aicos.edge")

DEFAULT_SETTINGS = {
    "process_every_n_frames": 12,
    "event_cooldown_seconds": 30,
    "resize_width": 640,
    "snapshot_interval_seconds": 2.0,   # for JPEG-snapshot cameras only
    "event_snapshots": True,            # attach the triggering frame to events
    "event_snapshot_quality": 70,       # JPEG quality 1-100
    "event_clips": True,                # attach a short clip of the lead-up
    "clip_seconds": 6.0,                # how much history to keep and send
    "clip_fps": 3,                      # frames per second in the saved clip
    "clip_width": 480,                  # clip is downscaled to this width
    "camera_heartbeat_seconds": 30,       # low-bandwidth operational health to backend
    "integrity_enabled": True,            # compare view against a site-approved baseline
    "integrity_check_seconds": 300,       # drift/obscuration check cadence
    "integrity_auto_baseline": False,     # government default: baseline must be deliberately approved
}

# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------


class Detector:
    """Base class. detect() returns a list of event dicts:
    {"agent": ..., "type": ..., "severity": 1-5, "confidence": 0-1}
    """

    name = "base"

    def detect(self, frame, prev_frame):  # pragma: no cover - interface
        return []


class MotionDetector(Detector):
    """Demo detector: flags significant motion in the frame.

    Purpose: prove the pipeline end-to-end. Replace or extend with
    production models for real deployments.
    """

    name = "motion-demo"

    def __init__(self, min_ratio: float = 0.02):
        self.min_ratio = min_ratio

    def detect(self, frame, prev_frame):
        if prev_frame is None:
            return []
        g1 = cv2.GaussianBlur(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (9, 9), 0)
        g2 = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0)
        diff = cv2.absdiff(g1, g2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        ratio = cv2.countNonZero(thresh) / float(thresh.shape[0] * thresh.shape[1])
        if ratio < self.min_ratio:
            return []
        confidence = min(0.95, 0.5 + ratio * 4)
        severity = 2 if ratio > 0.15 else 1
        return [{
            "agent": "security",
            "type": "Unusual activity (motion)",
            "severity": severity,
            "confidence": round(confidence, 2),
        }]


# ---------------------------------------------------------------------------
# PRODUCTION MODEL SLOT (example pattern, commented out)
# ---------------------------------------------------------------------------
# class YoloObjectDetector(Detector):
#     """Example: object detection with Ultralytics YOLO.
#     Install on the edge box:  pip install ultralytics
#     Then map model classes to AICOS event types below.
#     """
#     name = "yolo"
#
#     def __init__(self, weights="yolov8n.pt"):
#         from ultralytics import YOLO
#         self.model = YOLO(weights)
#
#     def detect(self, frame, prev_frame):
#         results = self.model(frame, verbose=False)[0]
#         persons = sum(1 for b in results.boxes if int(b.cls) == 0)
#         events = []
#         if persons >= 12:   # crowd build-up example rule
#             events.append({
#                 "agent": "security",
#                 "type": "Crowd build-up",
#                 "severity": 3,
#                 "confidence": 0.8,
#             })
#         return events
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Backend client
# ---------------------------------------------------------------------------


class BackendClient:
    """Reliable edge-to-backend delivery with a durable SQLite outbox.

    A network outage must not turn a detected incident into a lost incident.
    Every event gets a UUID. Failed sends are persisted locally and retried
    with exponential backoff; the backend deduplicates by event_uuid.

    v6.0.2 adds a quarantine/dead-letter path for permanent backend rejects.
    A permanent 4xx stops automatic retry but NEVER silently destroys the
    event.  Severity 4/5 events are protected from automatic row eviction.
    """

    PERMANENT_REJECTS = (400, 401, 403, 413, 422)

    def __init__(self, base_url: str, api_key: str):
        if not api_key or api_key in {"change-me", "changeme"} or len(api_key) < 24:
            raise RuntimeError("AICOS_API_KEY must be a strong 24+ character device key")
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/api/events"
        self.heartbeat_url = self.base_url + "/api/cameras/heartbeat"
        self.headers = {"X-AICOS-KEY": api_key, "Content-Type": "application/json"}
        self.session = requests.Session()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.outbox_path = os.environ.get("AICOS_OUTBOX_PATH", "aicos-outbox.db")
        self.outbox_max_mb = max(1.0, float(os.environ.get("AICOS_OUTBOX_MAX_MB", "256")))
        self.outbox_max_rows = max(10, int(os.environ.get("AICOS_OUTBOX_MAX_ROWS", "5000")))
        self.outbox_min_free_mb = max(0.0, float(os.environ.get("AICOS_OUTBOX_MIN_FREE_MB", "512")))
        self.dlq_max_mb = max(1.0, float(os.environ.get("AICOS_DLQ_MAX_MB", "256")))
        self.dlq_max_rows = max(10, int(os.environ.get("AICOS_DLQ_MAX_ROWS", "10000")))
        self.dropped_events = 0
        self.dropped_dead_letters = 0
        self.quarantined_events = 0
        self.compacted_events = 0
        parent = os.path.dirname(os.path.abspath(self.outbox_path)) or "."
        os.makedirs(parent, exist_ok=True)
        self._db = sqlite3.connect(self.outbox_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_uuid TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                payload TEXT NOT NULL,
                severity INTEGER NOT NULL DEFAULT 1,
                protected INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL DEFAULT 0,
                last_error TEXT DEFAULT ''
            )
            """
        )
        # Upgrade v6.0/v6.0.1 queues in place.
        for ddl in (
            "ALTER TABLE outbox ADD COLUMN severity INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE outbox ADD COLUMN protected INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                self._db.execute(ddl)
            except sqlite3.OperationalError:
                pass
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS dead_letter (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_uuid TEXT,
                created_at REAL NOT NULL,
                quarantined_at REAL NOT NULL,
                payload TEXT NOT NULL,
                severity INTEGER NOT NULL DEFAULT 1,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT DEFAULT '',
                reason TEXT NOT NULL DEFAULT 'permanent_reject'
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_outbox_due ON outbox(next_attempt_at, id)")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_outbox_protected ON outbox(protected,severity,id)")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_dead_letter_time ON dead_letter(quarantined_at,id)")
        self._backfill_priority_columns()
        self._db.commit()
        self._thread = threading.Thread(target=self._flush_loop, name="aicos-outbox", daemon=True)
        self._thread.start()

    @staticmethod
    def _severity(payload: dict) -> int:
        try:
            return max(1, min(5, int(payload.get("severity", 1))))
        except (TypeError, ValueError):
            return 1

    def _backfill_priority_columns(self) -> None:
        rows = self._db.execute(
            "SELECT id,payload,severity,protected FROM outbox"
        ).fetchall()
        for row_id, blob, severity, protected in rows:
            try:
                payload = json.loads(blob)
                sev = self._severity(payload)
            except Exception:
                sev = int(severity or 1)
            prot = 1 if sev >= 4 else int(protected or 0)
            self._db.execute(
                "UPDATE outbox SET severity=?,protected=? WHERE id=?",
                (sev, prot, row_id),
            )

    def _send(self, payload: dict) -> tuple[bool, str, bool]:
        """Return ``(delivered, error, permanent)``."""
        try:
            r = self.session.post(self.url, json=payload, headers=self.headers, timeout=5)
            if r.status_code in (200, 201):
                return True, "", False
            permanent = r.status_code in self.PERMANENT_REJECTS
            return False, "HTTP %s: %s" % (r.status_code, r.text[:200]), permanent
        except requests.RequestException as exc:
            return False, str(exc)[:300], False

    @staticmethod
    def _normalise_send(result):
        """Accept legacy tuple senders, but never guess that an odd value succeeded."""
        if isinstance(result, tuple) and len(result) == 3:
            return bool(result[0]), str(result[1] or ""), bool(result[2])
        if isinstance(result, tuple) and len(result) == 2:
            return bool(result[0]), str(result[1] or ""), False
        return False, f"unexpected _send result: {type(result).__name__}", False

    def _disk_free_mb(self) -> float:
        try:
            base = os.path.dirname(os.path.abspath(self.outbox_path)) or "."
            return shutil.disk_usage(base).free / (1024 * 1024)
        except OSError:
            return float("inf")

    def _queue_totals_locked(self) -> tuple[int, int]:
        row = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM outbox"
        ).fetchone()
        return int(row[0]), int(row[1])

    def _compact_payload_locked(self, row_id: int, blob: str, severity: int) -> int:
        """Remove bulky media while preserving incident metadata.

        Clip is removed first.  For severity 1-3, snapshot may also be removed
        under storage pressure.  Severity 4/5 keeps its snapshot so an operator
        still has visual evidence even during a long network outage.
        Returns bytes removed from the JSON payload.
        """
        try:
            payload = json.loads(blob)
        except Exception:
            return 0
        before = len(blob)
        changed = False
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        if payload.get("clip"):
            payload.pop("clip", None)
            extra["delivery_clip_omitted"] = True
            changed = True
        if severity <= 3 and payload.get("snapshot"):
            payload.pop("snapshot", None)
            extra["delivery_snapshot_omitted"] = True
            changed = True
        if not changed:
            return 0
        extra["delivery_degraded_at"] = time.time()
        extra["delivery_degraded_reason"] = "edge_outbox_storage_pressure"
        payload["extra"] = extra
        new_blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        self._db.execute("UPDATE outbox SET payload=? WHERE id=?", (new_blob, row_id))
        self.compacted_events += 1
        return max(0, before - len(new_blob))

    def _trim_outbox(self) -> None:
        """Bound edge storage without silently destroying critical incidents.

        Strategy: compact media first, then evict oldest severity 1-3 rows.
        Severity 4/5 rows are never automatically deleted; if they alone exceed
        the configured row budget AICOS logs a CRITICAL condition and keeps them.
        """
        rows, total = self._queue_totals_locked()
        max_bytes = int(self.outbox_max_mb * 1024 * 1024)
        low_disk = self._disk_free_mb() < self.outbox_min_free_mb
        if rows <= self.outbox_max_rows and total <= max_bytes and not low_disk:
            return

        # First shed media, preferring low severity rows. This keeps incident
        # metadata even when an outage is much longer than planned.
        candidates = self._db.execute(
            "SELECT id,payload,severity FROM outbox "
            "ORDER BY protected ASC, severity ASC, id ASC"
        ).fetchall()
        for row_id, blob, severity in candidates:
            if total <= max_bytes and not low_disk:
                break
            removed = self._compact_payload_locked(int(row_id), str(blob), int(severity or 1))
            total -= removed
            low_disk = self._disk_free_mb() < self.outbox_min_free_mb

        # If row count or bytes are still high, evict only non-critical events.
        dropped = 0
        while rows > self.outbox_max_rows or total > max_bytes or low_disk:
            victim = self._db.execute(
                "SELECT id,LENGTH(payload),event_uuid,severity FROM outbox "
                "WHERE protected=0 ORDER BY severity ASC,id ASC LIMIT 1"
            ).fetchone()
            if victim is None:
                log.critical(
                    "edge outbox pressure remains but only protected severity 4/5 events are left "
                    "(rows=%d payload=%.1fMB free=%.1fMB); preserving critical metadata",
                    rows, total / (1024 * 1024), self._disk_free_mb(),
                )
                break
            self._db.execute("DELETE FROM outbox WHERE id=?", (victim[0],))
            rows -= 1
            total -= int(victim[1] or 0)
            dropped += 1
            low_disk = self._disk_free_mb() < self.outbox_min_free_mb
        if dropped:
            self.dropped_events += dropped
            log.error(
                "edge outbox storage pressure — dropped %d low-severity event(s); "
                "%d low-severity events dropped since start",
                dropped, self.dropped_events,
            )

    def _prepare_dead_letter_blob(self, payload_blob: str, severity: int) -> str:
        """Compact quarantined payloads so a bad key cannot fill the edge disk.

        Dead-letter rows retain all event metadata. Clips are removed for every
        severity because the primary purpose of the DLQ is recoverable delivery;
        severity 4/5 keeps its snapshot, while severity 1-3 keeps metadata only.
        """
        try:
            payload = json.loads(payload_blob)
        except Exception:
            return payload_blob
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        if payload.pop("clip", None) is not None:
            extra["dlq_clip_omitted"] = True
        if severity <= 3 and payload.pop("snapshot", None) is not None:
            extra["dlq_snapshot_omitted"] = True
        extra["dead_letter_compacted"] = True
        payload["extra"] = extra
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    def _trim_dead_letter_locked(self) -> None:
        row = self._db.execute(
            "SELECT COUNT(*),COALESCE(SUM(LENGTH(payload)),0) FROM dead_letter"
        ).fetchone()
        rows, total = int(row[0]), int(row[1])
        max_bytes = int(self.dlq_max_mb * 1024 * 1024)
        dropped = 0
        while rows > self.dlq_max_rows or total > max_bytes:
            victim = self._db.execute(
                "SELECT id,LENGTH(payload) FROM dead_letter "
                "WHERE severity <= 3 ORDER BY severity ASC,id ASC LIMIT 1"
            ).fetchone()
            if victim is None:
                log.critical(
                    "dead-letter queue exceeds budget but only severity 4/5 items remain "
                    "(rows=%d payload=%.1fMB); preserving critical evidence",
                    rows, total / (1024 * 1024),
                )
                break
            self._db.execute("DELETE FROM dead_letter WHERE id=?", (victim[0],))
            rows -= 1; total -= int(victim[1] or 0); dropped += 1
        if dropped:
            self.dropped_dead_letters += dropped
            log.error(
                "dead-letter storage pressure — dropped %d low-severity quarantined event(s); "
                "%d dropped since start", dropped, self.dropped_dead_letters,
            )

    def _quarantine_locked(
        self, payload_blob: str, event_uuid: str | None, severity: int,
        attempts: int, error: str, reason: str, outbox_row_id: int | None = None,
    ) -> None:
        payload_blob = self._prepare_dead_letter_blob(payload_blob, int(severity))
        self._db.execute(
            "INSERT INTO dead_letter(event_uuid,created_at,quarantined_at,payload,severity,attempts,last_error,reason) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (event_uuid, time.time(), time.time(), payload_blob, int(severity), int(attempts),
             str(error)[:500], str(reason)[:80]),
        )
        if outbox_row_id is not None:
            self._db.execute("DELETE FROM outbox WHERE id=?", (int(outbox_row_id),))
        self.quarantined_events += 1
        self._trim_dead_letter_locked()

    def _quarantine_payload(self, payload: dict, error: str, reason: str, attempts: int = 0) -> None:
        blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._quarantine_locked(
                blob, str(payload.get("event_uuid") or "") or None,
                self._severity(payload), attempts, error, reason,
            )
            self._db.commit()

    def _enqueue(self, payload: dict, error: str) -> None:
        event_uuid = str(payload["event_uuid"])
        blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        severity = self._severity(payload)
        protected = 1 if severity >= 4 else 0
        with self._lock:
            self._db.execute(
                """
                INSERT INTO outbox(event_uuid,created_at,payload,severity,protected,attempts,next_attempt_at,last_error)
                VALUES (?,?,?,?,?,0,?,?)
                ON CONFLICT(event_uuid) DO UPDATE SET
                    payload=excluded.payload,
                    severity=excluded.severity,
                    protected=excluded.protected,
                    last_error=excluded.last_error
                """,
                (event_uuid, time.time(), blob, severity, protected, time.time() + 2.0, error[:300]),
            )
            self._trim_outbox()
            self._db.commit()

    def post_event(self, payload: dict) -> bool:
        payload = dict(payload)
        payload.setdefault("event_uuid", str(uuid.uuid4()))
        try:
            result = self._send(payload)
        except Exception as exc:
            result = (False, f"_send exception: {exc}", False)
        ok, error, permanent = self._normalise_send(result)
        if ok:
            return True
        if permanent:
            self._quarantine_payload(payload, error, "permanent_backend_reject")
            log.error(
                "backend permanently rejected event %s; quarantined in dead-letter queue: %s",
                payload["event_uuid"], error,
            )
            return False
        self._enqueue(payload, error)
        log.warning(
            "backend unavailable/rejected; event %s persisted to outbox: %s",
            payload["event_uuid"], error,
        )
        return True

    def _flush_loop(self) -> None:
        while not self._stop.wait(2.0):
            now = time.time()
            with self._lock:
                rows = self._db.execute(
                    "SELECT id,event_uuid,payload,severity,attempts FROM outbox "
                    "WHERE next_attempt_at <= ? ORDER BY protected DESC,severity DESC,id LIMIT 20",
                    (now,),
                ).fetchall()
            for row_id, event_uuid, blob, severity, attempts in rows:
                if self._stop.is_set():
                    break
                try:
                    payload = json.loads(blob)
                except (TypeError, ValueError) as exc:
                    with self._lock:
                        self._quarantine_locked(
                            str(blob), str(event_uuid) or None, int(severity or 1), int(attempts or 0),
                            f"corrupt JSON: {exc}", "corrupt_outbox_payload", int(row_id),
                        )
                        self._db.commit()
                    log.error("quarantined corrupt outbox row %s", row_id)
                    continue
                try:
                    result = self._send(payload)
                except Exception as exc:
                    result = (False, f"_send exception: {exc}", False)
                ok, error, permanent = self._normalise_send(result)
                with self._lock:
                    if ok:
                        self._db.execute("DELETE FROM outbox WHERE id=?", (row_id,))
                    elif permanent:
                        self._quarantine_locked(
                            str(blob), str(event_uuid) or None, int(severity or 1), int(attempts or 0) + 1,
                            error, "permanent_backend_reject", int(row_id),
                        )
                        log.error(
                            "queued event %s permanently rejected; moved to dead-letter queue: %s",
                            event_uuid, error,
                        )
                    else:
                        next_attempt = time.time() + min(
                            300.0, 2.0 ** min(int(attempts) + 1, 8)
                        )
                        self._db.execute(
                            "UPDATE outbox SET attempts=attempts+1,next_attempt_at=?,last_error=? WHERE id=?",
                            (next_attempt, error[:300], row_id),
                        )
                    self._db.commit()
                if ok:
                    log.info("delivered queued event %s", event_uuid)

    def status(self) -> dict:
        with self._lock:
            rows, payload_bytes = self._queue_totals_locked()
            dlq_row = self._db.execute(
                "SELECT COUNT(*),COALESCE(SUM(LENGTH(payload)),0),"
                "COALESCE(SUM(CASE WHEN severity>=4 THEN 1 ELSE 0 END),0) FROM dead_letter"
            ).fetchone()
            dlq, dlq_bytes, dlq_protected = int(dlq_row[0]), int(dlq_row[1]), int(dlq_row[2])
            protected = int(self._db.execute(
                "SELECT COUNT(*) FROM outbox WHERE protected=1"
            ).fetchone()[0])
        try:
            db_bytes = os.path.getsize(self.outbox_path)
        except OSError:
            db_bytes = 0
        return {
            "queued": rows,
            "queued_payload_bytes": payload_bytes,
            "protected_queued": protected,
            "dead_letter": dlq,
            "dead_letter_payload_bytes": dlq_bytes,
            "protected_dead_letters": dlq_protected,
            "db_bytes": db_bytes,
            "disk_free_mb": round(self._disk_free_mb(), 1),
            "dropped_low_severity": self.dropped_events,
            "dropped_low_severity_dead_letters": self.dropped_dead_letters,
            "quarantined": self.quarantined_events,
            "media_compactions": self.compacted_events,
        }

    def list_dead_letters(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id,event_uuid,created_at,quarantined_at,severity,attempts,last_error,reason "
                "FROM dead_letter ORDER BY id DESC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        keys = ("id", "event_uuid", "created_at", "quarantined_at", "severity", "attempts", "last_error", "reason")
        return [dict(zip(keys, row)) for row in rows]

    def requeue_dead_letter(self, dlq_id: int) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT id,event_uuid,payload,severity,last_error FROM dead_letter WHERE id=?",
                (int(dlq_id),),
            ).fetchone()
            if row is None:
                return False
            try:
                payload = json.loads(row[2])
            except Exception:
                return False
            event_uuid = str(payload.get("event_uuid") or row[1] or "")
            if not event_uuid:
                return False
            severity = self._severity(payload)
            self._db.execute(
                "INSERT INTO outbox(event_uuid,created_at,payload,severity,protected,attempts,next_attempt_at,last_error) "
                "VALUES(?,?,?,?,?,0,?,?) ON CONFLICT(event_uuid) DO UPDATE SET "
                "payload=excluded.payload,severity=excluded.severity,protected=excluded.protected," 
                "attempts=0,next_attempt_at=excluded.next_attempt_at,last_error=excluded.last_error",
                (event_uuid, time.time(), row[2], severity, 1 if severity >= 4 else 0,
                 time.time(), "manual replay from dead-letter queue"),
            )
            self._db.execute("DELETE FROM dead_letter WHERE id=?", (int(dlq_id),))
            self._db.commit()
            return True

    def post_heartbeat(self, payload: dict) -> bool:
        """Best-effort operational telemetry. Incident delivery is durable."""
        try:
            r = self.session.post(
                self.heartbeat_url, json=payload, headers=self.headers, timeout=4
            )
            return r.status_code in (200, 202)
        except requests.RequestException:
            return False

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        with self._lock:
            self._db.commit()
            self._db.close()
        self.session.close()


# ---------------------------------------------------------------------------
# Camera worker thread
# ---------------------------------------------------------------------------


class FrameBuffer:
    """A few seconds of recent frames, kept in memory.

    Alerts fire AFTER a behavior has persisted — a fall is only a fall once
    the person stays down. By then the interesting moment has passed. This
    keeps the lead-up so the evidence shows the incident itself, and is
    discarded continuously unless an event claims it.
    """

    def __init__(self, seconds: float):
        self.seconds = max(1.0, float(seconds))
        self._frames = []          # [(timestamp, frame), ...] oldest first

    def add(self, ts: float, frame) -> None:
        self._frames.append((ts, frame))
        cutoff = ts - self.seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.pop(0)

    def at(self, ts: float):
        """The buffered frame closest to a moment in time."""
        if not self._frames:
            return None
        return min(self._frames, key=lambda item: abs(item[0] - ts))[1]

    def since(self, ts: float, max_frames: int = 24):
        """Frames from ts onwards, evenly thinned to at most max_frames."""
        picked = [f for t, f in self._frames if t >= ts]
        if not picked:
            picked = [f for _, f in self._frames]
        if len(picked) <= max_frames:
            return picked
        step = len(picked) / float(max_frames)
        return [picked[int(i * step)] for i in range(max_frames)]

    def __len__(self):
        return len(self._frames)


class LatestFrame:
    """Holds only the newest frame from a live camera.

    A live stream does not wait for us. Reading frames in order means
    falling further behind with every inference until the analysis is
    minutes stale — the incident has happened and been missed. This
    discards everything except the current frame, so the detectors always
    see NOW, whatever the machine's speed.

    Video FILES are the opposite case and are read sequentially: every
    frame matters and pacing keeps them at real speed.
    """

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.stop = threading.Event()
        self.dropped = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.stop.set()
                break
            with self._lock:
                if self.frame is not None:
                    self.dropped += 1
                self.frame = frame

    def get(self):
        with self._lock:
            frame, self.frame = self.frame, None
            return frame

    def close(self):
        self.stop.set()
        self._thread.join(timeout=2)


class CameraWorker(threading.Thread):
    def __init__(self, cam: dict, settings: dict, backend: BackendClient,
                 detectors: list, stop_event: threading.Event):
        super().__init__(daemon=True, name=f"cam-{cam.get('id', '?')}")
        self.cam = cam
        # Optional ground calibration: turns an event's image position into
        # real coordinates, so incidents are reported where the person is
        # rather than at the camera. Absent or invalid -> camera coordinates.
        try:
            import geo
            self.calibration = geo.build(cam.get("calibration"),
                                         str(cam.get("id", "?")))
        except Exception as exc:
            log.error("[%s] calibration unavailable: %s", cam.get("id"), exc)
            self.calibration = None
        self.settings = settings
        self.backend = backend
        self.detectors = detectors
        self.stop_event = stop_event
        self._last_sent: dict[str, float] = {}
        self.buffer = FrameBuffer(settings.get("clip_seconds", 6.0))
        self._started_at = time.time()
        self._analysis_count = 0
        self._analysis_window_started = time.time()
        self._last_heartbeat = 0.0
        self._last_integrity_check = 0.0
        self._integrity = {"status": "unknown", "score": 0, "reason": "not assessed"}
        self._capture_fps_hint = 0.0
        self._dropped_frames = 0
        self._reconnects = 0
        self._source_kind = "unknown"
        self._integrity_monitor = None
        if self.settings.get("integrity_enabled", True):
            try:
                from camera_integrity import IntegrityMonitor
                base_dir = os.environ.get("AICOS_INTEGRITY_BASELINE_DIR", "aicos-integrity-baselines")
                self._integrity_monitor = IntegrityMonitor(
                    str(cam.get("id", "CAM-?")), base_dir,
                    auto_create=bool(self.settings.get("integrity_auto_baseline", False)),
                )
            except Exception as exc:
                log.warning("[%s] integrity monitor unavailable: %s", cam.get("id"), exc)

    def _analysis_fps(self) -> float:
        elapsed = max(0.001, time.time() - self._analysis_window_started)
        return self._analysis_count / elapsed

    def _maybe_integrity(self, frame) -> None:
        if self._integrity_monitor is None or frame is None:
            return
        interval = max(15.0, float(self.settings.get("integrity_check_seconds", 300)))
        now = time.time()
        if now - self._last_integrity_check < interval:
            return
        self._last_integrity_check = now
        try:
            self._integrity = self._integrity_monitor.assess(frame)
            if self._integrity.get("status") in {"warning", "drift_suspected", "obscured", "degraded"}:
                log.warning("[%s] camera integrity %s: %s", self.cam.get("id"),
                            self._integrity.get("status"), self._integrity.get("reason"))
        except Exception as exc:
            self._integrity = {"status":"error", "score":0, "reason":str(exc)[:160]}

    def _maybe_heartbeat(self) -> None:
        interval = max(10.0, float(self.settings.get("camera_heartbeat_seconds", 30)))
        now = time.time()
        if now - self._last_heartbeat < interval:
            return
        self._last_heartbeat = now
        integ = dict(self._integrity or {})
        metrics = {k:v for k,v in integ.items() if k not in {"status","score","reason"}}
        payload = {
            "source_id": str(self.cam.get("id", "CAM-?")),
            "name": str(self.cam.get("name", "")),
            "district": str(self.cam.get("district", "")),
            "worker_version": AICOS_VERSION,
            "source_kind": self._source_kind,
            "capture_fps": round(float(self._capture_fps_hint or 0), 3),
            "analysis_fps": round(float(self._analysis_fps()), 3),
            "dropped_frames": int(self._dropped_frames),
            "reconnects": int(self._reconnects),
            "uptime_seconds": round(now - self._started_at, 1),
            "integrity_status": str(integ.get("status", "unknown")),
            "integrity_score": float(integ.get("score", 0) or 0),
            "integrity_reason": str(integ.get("reason", "")),
            "integrity_metrics": metrics,
            "extra": {"rotate": self.cam.get("rotate", 0), "delivery": self.backend.status()},
        }
        if not self.backend.post_heartbeat(payload):
            log.debug("[%s] heartbeat delivery failed", self.cam.get("id"))

    @staticmethod
    def _is_youtube(src) -> bool:
        return isinstance(src, str) and (
            "youtube.com/" in src or "youtu.be/" in src)

    def _resolve_source(self, src):
        """Resolve a YouTube live URL to a direct stream URL via yt-dlp.

        Resolved URLs expire after a few hours. _open() runs again on every
        reconnect, so the worker re-resolves automatically and keeps running
        unattended.
        """
        if not self._is_youtube(src):
            return src
        if shutil.which("yt-dlp") is None:
            log.error("[%s] yt-dlp not installed - run: pip install yt-dlp",
                      self.cam.get("id"))
            return None
        try:
            out = subprocess.run(
                ["yt-dlp", "-g", "-f", "best[height<=720]/best", src],
                capture_output=True, text=True, timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.error("[%s] yt-dlp failed: %s", self.cam.get("id"), exc)
            return None
        if out.returncode != 0:
            log.error("[%s] yt-dlp error: %s", self.cam.get("id"),
                      (out.stderr or "").strip()[:200])
            return None
        # -g may print video and audio URLs; the first line is the video track.
        for line in (out.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("http"):
                log.info("[%s] resolved live stream URL", self.cam.get("id"))
                return line
        log.error("[%s] yt-dlp returned no stream URL", self.cam.get("id"))
        return None

    def _open(self):
        src = self.cam.get("rtsp_url", "")
        # Integer strings ("0") select a local webcam for desk testing.
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        else:
            src = self._resolve_source(src)
            if src is None:
                return None
        cap = cv2.VideoCapture(src)
        return cap if cap.isOpened() else None

    VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")

    def _is_file_source(self) -> bool:
        """True for video FILES — local paths and remote video URLs alike.
        Files are decoded as fast as the CPU allows, which would break
        wall-clock behavior thresholds, so run() paces them to their real
        frame rate. Live streams (rtsp/rtmp/HLS) are already real-time and
        must NOT be paced.
        """
        src = self.cam.get("rtsp_url", "")
        if not isinstance(src, str) or src.isdigit():
            return False
        low = src.lower()
        if low.startswith(("http://", "https://")):
            return low.split("?")[0].endswith(self.VIDEO_EXTS)
        if low.startswith(("rtsp://", "rtsps://", "rtmp://")):
            return False
        return os.path.isfile(src)

    def _cooldown_ok(self, event_type: str) -> bool:
        cooldown = float(self.settings["event_cooldown_seconds"])
        now = time.time()
        if now - self._last_sent.get(event_type, 0.0) < cooldown:
            return False
        self._last_sent[event_type] = now
        return True

    def _encode_snapshot(self, frame):
        """JPEG-encode the frame that triggered an alert so an operator can
        verify it. Failures are non-fatal — the event still goes out."""
        if frame is None or not self.settings.get("event_snapshots", True):
            return None
        try:
            quality = int(self.settings.get("event_snapshot_quality", 70))
            ok, buf = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return None
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception as exc:
            log.warning("[%s] snapshot encode failed: %s", self.cam.get("id"), exc)
            return None

    def _encode_clip(self, start_ts: float):
        """Animated GIF of the lead-up to an event.

        GIF because it plays in every browser with no codec, no plugin and
        no transcoding — the evidence has to open on whatever machine the
        operator is sitting at.
        """
        if not self.settings.get("event_clips", True):
            return None
        try:
            import io
            from PIL import Image
        except ImportError:
            return None
        try:
            fps = max(1, int(self.settings.get("clip_fps", 3)))
            secs = float(self.settings.get("clip_seconds", 6.0))
            width = int(self.settings.get("clip_width", 480))
            frames = self.buffer.since(start_ts, max_frames=int(fps * secs))
            if len(frames) < 2:
                return None
            images = []
            for raw in frames:
                h, w = raw.shape[:2]
                small = raw
                if w > width:
                    small = cv2.resize(raw, (width, int(h * width / w)))
                images.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            out = io.BytesIO()
            images[0].save(out, format="GIF", save_all=True,
                           append_images=images[1:],
                           duration=int(1000 / fps), loop=0, optimize=True)
            data = out.getvalue()
            if len(data) > 6_000_000:
                return None
            return base64.b64encode(data).decode("ascii")
        except Exception as exc:
            log.warning("[%s] clip encode failed: %s", self.cam.get("id"), exc)
            return None

    def _emit(self, detection: dict, frame=None) -> None:
        if not self._cooldown_ok(detection["type"]):
            return
        det = dict(detection)
        extra = {"camera_name": self.cam.get("name", "")}
        extra.update(det.pop("extra", {}) or {})
        if self._integrity:
            extra["camera_integrity_status"] = self._integrity.get("status", "unknown")
            extra["camera_integrity_score"] = self._integrity.get("score", 0)

        lat = float(self.cam.get("lat", 0.0))
        lon = float(self.cam.get("lon", 0.0))
        located = "camera"
        px = extra.get("px")
        if self.calibration is not None and isinstance(px, (list, tuple)) \
                and len(px) == 2:
            fixed = self.calibration.to_world(px[0], px[1])
            if fixed is not None:
                lat, lon = fixed
                located = "calibrated"
        extra["located"] = located

        payload = {
            "source_id": str(self.cam.get("id", "CAM-?")),
            "district": str(self.cam.get("district", "")),
            "lat": lat,
            "lon": lon,
            "extra": extra,
            **det,
        }
        # An alert fires after the behavior has persisted, so the current
        # frame shows the aftermath. Wind back to when it actually started.
        now = time.time()
        try:
            lead = float(extra.get("lead_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            lead = 0.0
        best = frame
        if lead > 0.5 and len(self.buffer):
            recovered = self.buffer.at(now - lead)
            if recovered is not None:
                best = recovered
                extra["snapshot_lead_seconds"] = round(lead, 1)

        snap = self._encode_snapshot(best)
        if snap:
            payload["snapshot"] = snap
        clip = self._encode_clip(now - max(lead, 2.0))
        if clip:
            payload["clip"] = clip
        if self.backend.post_event(payload):
            log.info("[%s] event accepted for delivery: %s (sev %s, conf %s)",
                     self.cam.get("id"), detection["type"],
                     detection["severity"], detection["confidence"])

    def _is_snapshot_source(self) -> bool:
        """True for cameras that publish a single JPEG that refreshes, rather
        than a continuous stream. Very common on public / municipal traffic
        cameras. Set `snapshot: true` on the camera to force this mode.
        """
        src = self.cam.get("rtsp_url", "")
        if not isinstance(src, str):
            return False
        if not src.lower().startswith(("http://", "https://")):
            return False
        if self.cam.get("snapshot") is True:
            return True
        base = src.split("?")[0].lower()
        return base.endswith((".jpg", ".jpeg", ".png"))

    def _run_snapshot(self) -> None:
        """Poll a still-image camera on an interval and analyze each frame.

        Note: at multi-second intervals, count- and persistence-based
        behaviors (crowd, person down, intrusion, loitering) work; motion
        heuristics that need continuity (fight, panic) will not fire.
        """
        url = self.cam.get("rtsp_url", "")
        try:
            interval = float(self.settings.get("snapshot_interval_seconds", 2.0))
        except (TypeError, ValueError):
            interval = 2.0
        interval = max(0.5, interval)
        resize_w = int(self.settings["resize_width"])
        session = requests.Session()
        prev_frame = None
        fails = 0
        log.info("[%s] snapshot source: polling every %.1fs", self.cam.get("id"), interval)
        while not self.stop_event.is_set():
            started = time.time()
            frame = None
            try:
                resp = session.get(url, timeout=10)
                if resp.status_code == 200 and resp.content:
                    buf = np.frombuffer(resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                else:
                    log.warning("[%s] snapshot HTTP %s", self.cam.get("id"), resp.status_code)
            except requests.RequestException as exc:
                log.warning("[%s] snapshot fetch failed: %s", self.cam.get("id"), exc)

            if frame is None:
                fails += 1
                if fails in (1, 5, 20) or fails % 50 == 0:
                    log.warning("[%s] no usable snapshot (%d consecutive)",
                                self.cam.get("id"), fails)
            else:
                if fails:
                    log.info("[%s] snapshot recovered", self.cam.get("id"))
                fails = 0
                frame = self._orient(frame)
                h, w = frame.shape[:2]
                if w > resize_w:
                    frame = cv2.resize(frame, (resize_w, int(h * resize_w / w)))
                self.buffer.add(time.time(), frame)
                self._analysis_count += 1
                self._maybe_integrity(frame)
                self._maybe_heartbeat()
                for det in self.detectors:
                    try:
                        for detection in det.detect(frame, prev_frame):
                            self._emit(detection, frame)
                    except Exception as exc:
                        log.error("[%s] detector %s failed: %s",
                                  self.cam.get("id"), det.name, exc)
                prev_frame = frame

            # back off when the camera is unreachable, but never stop trying
            wait = interval * (4 if fails > 5 else 1) - (time.time() - started)
            if wait > 0:
                self.stop_event.wait(wait)

    def _orient(self, frame):
        """Rotate the frame so 'up' in the image is up in the world.

        Every detector reasons about vertical: a standing person is tall, a
        fallen one is not. A camera mounted sideways — a phone on its side,
        a wall mount turned to fit — makes standing people look horizontal
        and breaks all of it. Correcting here means everything downstream,
        including zones and calibration, works in the right orientation.
        """
        rot = self.cam.get("rotate", 0)
        try:
            rot = int(rot) % 360
        except (TypeError, ValueError):
            return frame
        if rot == 0:
            return frame
        code = {90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rot)
        if code is None:
            if not getattr(self, "_rot_warned", False):
                log.warning("[%s] rotate must be 0, 90, 180 or 270 — got %s, "
                            "ignoring", self.cam.get("id"), rot)
                self._rot_warned = True
            return frame
        return cv2.rotate(frame, code)

    def _analyze(self, frame, prev_frame, resize_w: int):
        """Resize, buffer and run every detector over one frame."""
        frame = self._orient(frame)
        h, w = frame.shape[:2]
        if w > resize_w:
            frame = cv2.resize(frame, (resize_w, int(h * resize_w / w)))
        self.buffer.add(time.time(), frame)
        self._analysis_count += 1
        self._maybe_integrity(frame)
        self._maybe_heartbeat()
        for det in self.detectors:
            try:
                for detection in det.detect(frame, prev_frame):
                    self._emit(detection, frame)
            except Exception as exc:
                log.error("[%s] detector %s failed: %s",
                          self.cam.get("id"), det.name, exc)
        return frame

    def _run_live(self, cap, resize_w: int) -> None:
        """Live camera: always analyze the newest frame, never a stale one."""
        grabber = LatestFrame(cap)
        prev_frame = None
        reported = False
        try:
            while not self.stop_event.is_set() and not grabber.stop.is_set():
                frame = grabber.get()
                if frame is None:
                    self.stop_event.wait(0.02)
                    continue
                self._dropped_frames = grabber.dropped
                prev_frame = self._analyze(frame, prev_frame, resize_w)
                if not reported and grabber.dropped > 200:
                    log.info("[%s] keeping up with the live feed by skipping "
                             "stale frames (%d so far)",
                             self.cam.get("id"), grabber.dropped)
                    reported = True
        finally:
            grabber.close()

    def run(self) -> None:
        if self._is_snapshot_source():
            self._run_snapshot()
            return
        every_n = int(self.settings["process_every_n_frames"])
        resize_w = int(self.settings["resize_width"])
        backoff = 5
        while not self.stop_event.is_set():
            cap = self._open()
            if cap is None:
                log.warning("[%s] cannot open stream, retrying in %ss",
                            self.cam.get("id"), backoff)
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 60)
                continue
            log.info("[%s] connected: %s", self.cam.get("id"), self.cam.get("name", ""))
            self._reconnects += 1 if self._capture_fps_hint else 0
            try:
                hint = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                self._capture_fps_hint = hint if 0.5 <= hint <= 240 else 0.0
            except Exception:
                self._capture_fps_hint = 0.0
            src = self.cam.get("rtsp_url", "")
            if self._is_file_source(): self._source_kind = "file"
            elif isinstance(src, str) and src.isdigit(): self._source_kind = "local"
            elif isinstance(src, str) and src.lower().startswith(("rtsp://","rtsps://")): self._source_kind = "rtsp"
            elif isinstance(src, str) and src.lower().startswith(("http://","https://")): self._source_kind = "http"
            else: self._source_kind = "video"
            backoff = 5
            prev_frame = None
            frame_no = 0
            is_file = self._is_file_source()
            if not is_file:
                # live camera — newest frame only, so inference speed can
                # never push the analysis behind real time
                self._run_live(cap, int(self.settings["resize_width"]))
                cap.release()
                if not self.stop_event.is_set():
                    log.warning("[%s] stream lost, reconnecting", self.cam.get("id"))
                    self.stop_event.wait(3.0)
                continue
            fps = 0.0
            if is_file:
                try:
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                except Exception:
                    fps = 0.0
                if not (1.0 < fps < 120.0):   # also rejects NaN
                    fps = 25.0
                log.info("[%s] file source: pacing playback to %.1f fps",
                         self.cam.get("id"), fps)
            t_start = time.time()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_file:
                        log.info("[%s] file ended, looping", self.cam.get("id"))
                    else:
                        log.warning("[%s] stream lost, reconnecting", self.cam.get("id"))
                    break
                frame = self._orient(frame)
                frame_no += 1
                if is_file:
                    # keep video-file playback at real speed
                    delay = (t_start + frame_no / fps) - time.time()
                    if delay > 0:
                        self.stop_event.wait(min(delay, 1.0))
                if frame_no % every_n != 0:
                    continue
                h, w = frame.shape[:2]
                if w > resize_w:
                    frame = cv2.resize(frame, (resize_w, int(h * resize_w / w)))
                self.buffer.add(time.time(), frame)
                for det in self.detectors:
                    try:
                        for detection in det.detect(frame, prev_frame):
                            self._emit(detection, frame)
                    except Exception as exc:
                        log.error("[%s] detector %s failed: %s",
                                  self.cam.get("id"), det.name, exc)
                prev_frame = frame
            cap.release()
            if not self.stop_event.is_set():
                # Brief pause before reconnecting. Without it, a camera that
                # opens but delivers no frames spins the CPU and floods logs.
                self.stop_event.wait(0.5 if is_file else 3.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base without mutating either."""
    out = dict(base or {})
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _model_detector_config(settings: dict, cam: dict, name: str) -> dict:
    """Read original nested settings and canonical settings without silent loss.

    Camera settings override global settings. Within either scope the explicit
    canonical key overrides its legacy yolo-nested equivalent.
    """
    result = {}
    for scope in (settings.get('detectors') or {}, cam.get('detectors') or {}):
        result = deep_merge(result, (scope.get('yolo') or {}).get(name) or {})
        result = deep_merge(result, scope.get(name) or {})
    return result


def build_detectors(settings: dict, cam: dict) -> list:
    """Build the detector stack for ONE camera.

    Detector settings come from settings.detectors, and any camera may
    override them with its own `detectors:` block — a busy square and a
    quiet alley need different crowd thresholds. The suite keeps
    per-camera state, so instances are never shared between cameras.
    """
    dcfg = settings.get("detectors") or {}
    cam_over = cam.get("detectors") or {}
    ycfg = deep_merge(dcfg.get("yolo") or {}, cam_over.get("yolo") or {})
    mcfg = deep_merge(dcfg.get("motion") or {}, cam_over.get("motion") or {})
    vcfg = _model_detector_config(settings, cam, 'violence')
    dets = []
    if ycfg.get("enabled"):
        from detectors import YoloBehaviorSuite   # requires requirements-yolo.txt
        dets.append(YoloBehaviorSuite(camera=cam, config=ycfg))
    # Preserve the original no-config demo, but respect an explicit disable.
    if mcfg.get("enabled", not ycfg.get("enabled")):
        dets.append(MotionDetector())
    if vcfg.get("enabled"):
        from violence import ViolenceDetector     # trained model, see docs/
        dets.append(ViolenceDetector(config=vcfg))
    fcfg = _model_detector_config(settings, cam, 'fall_model')
    if fcfg.get("enabled"):
        from falldet import FallModelDetector     # trained on fallen people
        dets.append(FallModelDetector(config=fcfg))
    return dets


def load_config(path: str) -> tuple[dict, list]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    settings = {**DEFAULT_SETTINGS, **(cfg.get("settings") or {})}
    cameras = cfg.get("cameras") or []
    if not cameras:
        raise SystemExit("No cameras defined in config: " + path)
    return settings, cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="AICOS edge worker")
    parser.add_argument("--config", default="../config/cameras.yaml",
                        help="Path to cameras.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:8080")
    api_key = os.environ.get("AICOS_API_KEY", "").strip()

    settings, cameras = load_config(args.config)
    backend = BackendClient(backend_url, api_key)
    stop_event = threading.Event()

    workers = [CameraWorker(cam, settings, backend,
                            build_detectors(settings, cam), stop_event)
               for cam in cameras]
    for wkr in workers:
        wkr.start()
    log.info("edge worker running: %d camera(s) -> %s", len(workers), backend_url)

    try:
        while any(wkr.is_alive() for wkr in workers):
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("stopping...")
    finally:
        stop_event.set()
        for wkr in workers:
            wkr.join(timeout=5)
        backend.close()


if __name__ == "__main__":
    main()
