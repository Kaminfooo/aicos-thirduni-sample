"""Privacy-first forensic search over AICOS incident evidence.

This module intentionally does NOT perform facial recognition or biometric
identity search. It provides fast structured/keyword search, timeline search,
and scene-level visual similarity using a non-biometric perceptual hash.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def _event_text(row: dict) -> str:
    extra = row.get("extra", "{}")
    if not isinstance(extra, str):
        extra = json.dumps(extra, ensure_ascii=False)
    return " ".join(str(row.get(k, "")) for k in ("source_id", "district", "agent", "type", "status")) + " " + extra


def image_dhash(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        from PIL import Image
        with Image.open(p) as im:
            im = im.convert("L").resize((9, 8))
            px = list(im.getdata())
        bits = 0
        for y in range(8):
            for x in range(8):
                bits = (bits << 1) | (1 if px[y * 9 + x] > px[y * 9 + x + 1] else 0)
        return f"{bits:016x}"
    except Exception:
        return None


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


class ForensicIndex:
    def __init__(self, conn, lock, snapshot_dir: str) -> None:
        self.conn = conn
        self.lock = lock
        self.snapshot_dir = Path(snapshot_dir)
        self.fts = False
        self._init_db()

    def _init_db(self) -> None:
        with self.lock:
            try:
                self.conn.execute("ALTER TABLE events ADD COLUMN visual_hash TEXT")
            except Exception:
                pass
            try:
                self.conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(event_id UNINDEXED, text)"
                )
                self.fts = True
            except Exception:
                self.fts = False
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_forensics_time ON events(ts DESC)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_forensics_source ON events(source_id, ts DESC)")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forensic_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    source_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    color TEXT DEFAULT '',
                    direction TEXT DEFAULT '',
                    track_id TEXT DEFAULT '',
                    appearance_key TEXT DEFAULT '',
                    x REAL DEFAULT 0, y REAL DEFAULT 0, w REAL DEFAULT 0, h REAL DEFAULT 0,
                    extra TEXT DEFAULT '{}',
                    UNIQUE(source_id, ts, kind, x, y, w, h)
                )
                """
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_time ON forensic_observations(ts DESC)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_source_time ON forensic_observations(source_id, ts DESC)")
            for coldef in ("track_id TEXT DEFAULT ''", "appearance_key TEXT DEFAULT ''"):
                try:
                    self.conn.execute(f"ALTER TABLE forensic_observations ADD COLUMN {coldef}")
                except Exception:
                    pass
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_kind_color ON forensic_observations(kind, color, ts DESC)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_track ON forensic_observations(source_id, track_id, ts DESC)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_appearance ON forensic_observations(appearance_key, ts DESC)")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forensic_camera_links (
                    source_id TEXT NOT NULL,
                    next_source_id TEXT NOT NULL,
                    min_seconds REAL NOT NULL DEFAULT 0,
                    max_seconds REAL NOT NULL DEFAULT 900,
                    notes TEXT DEFAULT '',
                    PRIMARY KEY(source_id, next_source_id)
                )
                """
            )
            self.conn.commit()
        self.rebuild_missing()

    def index_event(self, event_id: int) -> None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
            if row is None:
                return
            data = dict(row)
        vh = image_dhash(self.snapshot_dir / f"{event_id}.jpg")
        with self.lock:
            self.conn.execute("UPDATE events SET visual_hash = ? WHERE id = ?", (vh, event_id))
            if self.fts:
                self.conn.execute("DELETE FROM events_fts WHERE event_id = ?", (event_id,))
                self.conn.execute("INSERT INTO events_fts(event_id, text) VALUES (?, ?)", (event_id, _event_text(data)))
            self.conn.commit()

    def rebuild_missing(self) -> None:
        with self.lock:
            rows = self.conn.execute("SELECT id FROM events WHERE visual_hash IS NULL AND has_snapshot = 1 LIMIT 5000").fetchall()
        for row in rows:
            self.index_event(int(row["id"]))

    @staticmethod
    def _row_dict(row) -> dict:
        d = dict(row)
        try:
            d["extra"] = json.loads(d.get("extra") or "{}")
        except Exception:
            d["extra"] = {}
        return d

    def search(self, *, q: str = "", since: Optional[float] = None, until: Optional[float] = None,
               source_id: str = "", district: str = "", agent: str = "", event_type: str = "",
               status: str = "", min_severity: int = 1, max_severity: int = 5,
               min_confidence: float = 0.0, has_snapshot: Optional[bool] = None,
               has_clip: Optional[bool] = None, limit: int = 100, offset: int = 0) -> dict:
        joins = ""
        where = ["e.severity BETWEEN ? AND ?", "e.confidence >= ?"]
        params: list = [min_severity, max_severity, min_confidence]
        q = (q or "").strip()
        if q and self.fts:
            joins = " JOIN events_fts f ON CAST(f.event_id AS INTEGER) = e.id "
            where.append("events_fts MATCH ?")
            # Treat words as AND terms; quote each to avoid FTS operators from user input.
            terms = [t.replace('"', '') for t in q.split() if t.strip()][:12]
            params.append(" AND ".join(f'"{t}"' for t in terms))
        elif q:
            where.append("(LOWER(e.type) LIKE ? OR LOWER(e.district) LIKE ? OR LOWER(e.source_id) LIKE ? OR LOWER(e.extra) LIKE ?)")
            like = f"%{q.lower()}%"
            params.extend([like, like, like, like])
        for col, val in (("source_id", source_id), ("district", district), ("agent", agent), ("status", status)):
            if val:
                where.append(f"e.{col} = ?")
                params.append(val)
        if event_type:
            where.append("LOWER(e.type) LIKE ?")
            params.append(f"%{event_type.lower()}%")
        if since is not None:
            where.append("e.ts >= ?")
            params.append(float(since))
        if until is not None:
            where.append("e.ts <= ?")
            params.append(float(until))
        if has_snapshot is not None:
            where.append("e.has_snapshot = ?")
            params.append(1 if has_snapshot else 0)
        if has_clip is not None:
            where.append("e.has_clip = ?")
            params.append(1 if has_clip else 0)
        sql = f"SELECT e.* FROM events e {joins} WHERE {' AND '.join(where)} ORDER BY e.ts DESC, e.id DESC LIMIT ? OFFSET ?"
        params.extend([min(500, max(1, int(limit))), max(0, int(offset))])
        with self.lock:
            rows = self.conn.execute(sql, params).fetchall()
        return {"events": [self._row_dict(r) for r in rows], "limit": limit, "offset": offset, "fts": self.fts}

    def add_observations(self, source_id: str, observations: list[dict]) -> int:
        rows = []
        for o in observations[:1000]:
            try:
                ts = float(o.get("ts"))
                kind = str(o.get("kind", "")).strip().lower()
                conf = float(o.get("confidence", 0.0))
                if not kind or not (0.0 <= conf <= 1.0):
                    continue
                bbox = o.get("bbox") or [0, 0, 0, 0]
                if len(bbox) != 4:
                    continue
                x, y, w, h = [float(v) for v in bbox]
                if not all(math.isfinite(v) for v in (ts, conf, x, y, w, h)):
                    continue
                rows.append((ts, source_id, kind, conf, str(o.get("color", ""))[:32].lower(),
                             str(o.get("direction", ""))[:32].lower(),
                             str(o.get("track_id", ""))[:96], str(o.get("appearance_key", ""))[:128],
                             x, y, w, h,
                             json.dumps(o.get("extra") or {}, separators=(",", ":"), ensure_ascii=False)))
            except Exception:
                continue
        if not rows:
            return 0
        with self.lock:
            before = self.conn.total_changes
            self.conn.executemany(
                "INSERT OR IGNORE INTO forensic_observations(ts,source_id,kind,confidence,color,direction,track_id,appearance_key,x,y,w,h,extra) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self.conn.commit()
            return self.conn.total_changes - before

    def search_observations(self, *, since: Optional[float] = None, until: Optional[float] = None,
                            source_id: str = "", kind: str = "", color: str = "", direction: str = "",
                            track_id: str = "", appearance_key: str = "",
                            min_confidence: float = 0.0, limit: int = 200, offset: int = 0) -> dict:
        where = ["confidence >= ?"]
        params: list = [float(min_confidence)]
        for col, val in (("source_id", source_id), ("kind", kind.lower()), ("color", color.lower()),
                         ("direction", direction.lower()), ("track_id", track_id), ("appearance_key", appearance_key)):
            if val:
                where.append(f"{col} = ?")
                params.append(val)
        if since is not None:
            where.append("ts >= ?"); params.append(float(since))
        if until is not None:
            where.append("ts <= ?"); params.append(float(until))
        sql = f"SELECT * FROM forensic_observations WHERE {' AND '.join(where)} ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([min(1000, max(1, int(limit))), max(0, int(offset))])
        with self.lock:
            rows = self.conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try: d["extra"] = json.loads(d.get("extra") or "{}")
            except Exception: d["extra"] = {}
            d["bbox"] = [d.pop("x"), d.pop("y"), d.pop("w"), d.pop("h")]
            out.append(d)
        return {"observations": out, "limit": limit, "offset": offset}

    def set_camera_link(self, source_id: str, next_source_id: str, *, min_seconds: float = 0,
                        max_seconds: float = 900, notes: str = "", bidirectional: bool = False) -> dict:
        source_id, next_source_id = str(source_id).strip(), str(next_source_id).strip()
        if not source_id or not next_source_id or source_id == next_source_id:
            raise ValueError("source_id and next_source_id must be distinct non-empty camera ids")
        lo, hi = max(0.0, float(min_seconds)), max(1.0, float(max_seconds))
        if lo > hi:
            raise ValueError("min_seconds cannot exceed max_seconds")
        rows = [(source_id, next_source_id, lo, hi, str(notes)[:240])]
        if bidirectional:
            rows.append((next_source_id, source_id, lo, hi, str(notes)[:240]))
        with self.lock:
            self.conn.executemany(
                "INSERT INTO forensic_camera_links(source_id,next_source_id,min_seconds,max_seconds,notes) VALUES(?,?,?,?,?) "
                "ON CONFLICT(source_id,next_source_id) DO UPDATE SET min_seconds=excluded.min_seconds,max_seconds=excluded.max_seconds,notes=excluded.notes",
                rows,
            )
            self.conn.commit()
        return {"links": [{"source_id":a,"next_source_id":b,"min_seconds":lo,"max_seconds":hi,"notes":str(notes)[:240]} for a,b,*_ in rows]}

    def list_camera_links(self) -> dict:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM forensic_camera_links ORDER BY source_id,next_source_id").fetchall()
        return {"links": [dict(r) for r in rows]}

    def journey_candidates(self, observation_id: int, *, limit: int = 50) -> dict:
        """Find plausible next-camera observations without claiming identity.

        Correlation is deliberately based on configured camera adjacency, time,
        object kind, coarse color and optional non-biometric appearance_key. It
        returns candidates for an operator to review; it never asserts that two
        observations are the same person/vehicle.
        """
        with self.lock:
            src = self.conn.execute("SELECT * FROM forensic_observations WHERE id = ?", (int(observation_id),)).fetchone()
        if src is None:
            return {"observation_id": observation_id, "candidates": [], "reason": "observation not found"}
        s = dict(src)
        with self.lock:
            links = self.conn.execute("SELECT * FROM forensic_camera_links WHERE source_id = ?", (s["source_id"],)).fetchall()
        candidates = []
        for link in links:
            start = float(s["ts"]) + float(link["min_seconds"])
            end = float(s["ts"]) + float(link["max_seconds"])
            params = [link["next_source_id"], s["kind"], start, end]
            where = ["source_id = ?", "kind = ?", "ts BETWEEN ? AND ?"]
            with self.lock:
                rows = self.conn.execute(
                    f"SELECT * FROM forensic_observations WHERE {' AND '.join(where)} ORDER BY ts ASC LIMIT 500", params
                ).fetchall()
            for row in rows:
                d = dict(row)
                score = 0.45
                reasons = ["adjacent-camera time window", "same object class"]
                if s.get("color") and d.get("color") and s["color"] == d["color"]:
                    score += 0.20; reasons.append("same coarse color")
                if s.get("appearance_key") and d.get("appearance_key") and s["appearance_key"] == d["appearance_key"]:
                    score += 0.25; reasons.append("same non-biometric appearance key")
                if s.get("direction") and d.get("direction") and s["direction"] == d["direction"]:
                    score += 0.05; reasons.append("same direction label")
                score += min(0.05, max(0.0, float(d.get("confidence") or 0)) * 0.05)
                try: d["extra"] = json.loads(d.get("extra") or "{}")
                except Exception: d["extra"] = {}
                d["bbox"] = [d.pop("x"), d.pop("y"), d.pop("w"), d.pop("h")]
                candidates.append({"score": round(min(0.99, score), 3), "reasons": reasons, "observation": d})
        candidates.sort(key=lambda x: (-x["score"], x["observation"]["ts"]))
        return {
            "observation_id": observation_id,
            "method": "operator-reviewed non-biometric adjacency correlation",
            "identity_claim": False,
            "candidates": candidates[:max(1, min(int(limit), 200))],
        }

    def similar(self, event_id: int, limit: int = 20, max_distance: int = 20) -> dict:
        with self.lock:
            src = self.conn.execute("SELECT visual_hash FROM events WHERE id = ?", (event_id,)).fetchone()
            rows = self.conn.execute("SELECT * FROM events WHERE id != ? AND visual_hash IS NOT NULL", (event_id,)).fetchall()
        if src is None or not src["visual_hash"]:
            return {"event_id": event_id, "matches": [], "reason": "source event has no snapshot visual hash"}
        needle = str(src["visual_hash"])
        scored = []
        for row in rows:
            d = hamming_hex(needle, str(row["visual_hash"]))
            if d <= max_distance:
                similarity = round(1.0 - d / 64.0, 4)
                scored.append((d, similarity, self._row_dict(row)))
        scored.sort(key=lambda x: (x[0], -float(x[2].get("ts", 0))))
        return {"event_id": event_id, "method": "64-bit scene dHash (non-biometric)",
                "matches": [{"distance": d, "similarity": sim, "event": ev} for d, sim, ev in scored[:limit]]}
