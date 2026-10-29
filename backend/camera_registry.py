"""Operational camera registry for AICOS Government Edition.

The registry receives low-bandwidth heartbeats from edge workers and keeps a
single authoritative view of camera liveness, inference throughput and camera
integrity/drift state. No video frames are stored here.
"""
from __future__ import annotations

import json
import math
import os
import time


class CameraRegistry:
    def __init__(self, conn, lock) -> None:
        self.conn = conn
        self.lock = lock
        try:
            self.stale_seconds = max(15.0, float(os.environ.get("AICOS_CAMERA_STALE_SECONDS", "90")))
        except ValueError:
            self.stale_seconds = 90.0
        self._init_db()

    def _init_db(self) -> None:
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS camera_runtime (
                    source_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    district TEXT DEFAULT '',
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    worker_version TEXT DEFAULT '',
                    source_kind TEXT DEFAULT '',
                    capture_fps REAL DEFAULT 0,
                    analysis_fps REAL DEFAULT 0,
                    dropped_frames INTEGER DEFAULT 0,
                    reconnects INTEGER DEFAULT 0,
                    uptime_seconds REAL DEFAULT 0,
                    integrity_status TEXT DEFAULT 'unknown',
                    integrity_score REAL DEFAULT 0,
                    integrity_reason TEXT DEFAULT '',
                    integrity_metrics TEXT DEFAULT '{}',
                    extra TEXT DEFAULT '{}'
                )
                """
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_camera_runtime_seen ON camera_runtime(last_seen DESC)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_camera_runtime_integrity ON camera_runtime(integrity_status, last_seen DESC)")
            self.conn.commit()

    def heartbeat(self, source_id: str, data: dict) -> dict:
        now = time.time()
        metrics = data.get("integrity_metrics") or {}
        extra = data.get("extra") or {}
        with self.lock:
            existing = self.conn.execute(
                "SELECT first_seen FROM camera_runtime WHERE source_id = ?", (source_id,)
            ).fetchone()
            first_seen = float(existing["first_seen"]) if existing is not None else now
            self.conn.execute(
                """
                INSERT INTO camera_runtime (
                    source_id,name,district,first_seen,last_seen,worker_version,source_kind,
                    capture_fps,analysis_fps,dropped_frames,reconnects,uptime_seconds,
                    integrity_status,integrity_score,integrity_reason,integrity_metrics,extra
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source_id) DO UPDATE SET
                    name=excluded.name,district=excluded.district,last_seen=excluded.last_seen,
                    worker_version=excluded.worker_version,source_kind=excluded.source_kind,
                    capture_fps=excluded.capture_fps,analysis_fps=excluded.analysis_fps,
                    dropped_frames=excluded.dropped_frames,reconnects=excluded.reconnects,
                    uptime_seconds=excluded.uptime_seconds,integrity_status=excluded.integrity_status,
                    integrity_score=excluded.integrity_score,integrity_reason=excluded.integrity_reason,
                    integrity_metrics=excluded.integrity_metrics,extra=excluded.extra
                """,
                (
                    source_id, str(data.get("name", ""))[:120], str(data.get("district", ""))[:120],
                    first_seen, now, str(data.get("worker_version", ""))[:32], str(data.get("source_kind", ""))[:32],
                    float(data.get("capture_fps", 0) or 0), float(data.get("analysis_fps", 0) or 0),
                    int(data.get("dropped_frames", 0) or 0), int(data.get("reconnects", 0) or 0),
                    float(data.get("uptime_seconds", 0) or 0), str(data.get("integrity_status", "unknown"))[:40],
                    float(data.get("integrity_score", 0) or 0), str(data.get("integrity_reason", ""))[:240],
                    json.dumps(metrics, separators=(",", ":"), ensure_ascii=False),
                    json.dumps(extra, separators=(",", ":"), ensure_ascii=False),
                ),
            )
            self.conn.commit()
        return self.get(source_id) or {"source_id": source_id}

    @staticmethod
    def _decode(row, stale_seconds: float) -> dict:
        d = dict(row)
        for field in ("integrity_metrics", "extra"):
            try:
                d[field] = json.loads(d.get(field) or "{}")
            except Exception:
                d[field] = {}
        age = max(0.0, time.time() - float(d.get("last_seen") or 0))
        d["heartbeat_age_seconds"] = round(age, 1)
        d["online"] = age <= stale_seconds
        d["runtime_status"] = "online" if d["online"] else "offline"
        extra = d['extra'] if isinstance(d['extra'], dict) else {}
        try:
            frame_age = float(extra['frame_age_seconds_at_heartbeat']) + age
            if not math.isfinite(frame_age) or frame_age < 0:
                frame_age = None
        except (KeyError, TypeError, ValueError, OverflowError):
            frame_age = None
        d['frame_age_seconds'] = round(frame_age, 1) if frame_age is not None else None
        d['operational_status'] = ('offline' if not d['online'] else
                                   'detector_failed' if extra.get('detector_status') == 'failed' else
                                   'frames_stale' if frame_age is not None and frame_age > 15 else
                                   'online')
        return d

    def get(self, source_id: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM camera_runtime WHERE source_id = ?", (source_id,)).fetchone()
        return self._decode(row, self.stale_seconds) if row is not None else None

    def list(self) -> dict:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM camera_runtime ORDER BY source_id").fetchall()
        items = [self._decode(r, self.stale_seconds) for r in rows]
        return {
            "stale_after_seconds": self.stale_seconds,
            "count": len(items),
            "online": sum(1 for x in items if x["online"]),
            "integrity_attention": sum(1 for x in items if x.get("integrity_status") not in {"ok", "baseline", "unknown"}),
            "cameras": items,
        }
