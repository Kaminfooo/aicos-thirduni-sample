"""Durable outbound integration bus for VMS/SIEM/dispatch systems.

Government deployments often integrate AICOS with an existing command center.
This module emits a vendor-neutral JSON envelope to configured HTTPS webhooks,
HMAC-signs each payload, persists failed deliveries in SQLite and retries with
exponential backoff. It is intentionally independent of any one VMS SDK.

Delivery semantics are deliberately fail-safe:
* real HTTP status codes are classified from ``HTTPError.code`` (never by
  string matching in the production sender),
* unknown custom sender return shapes are failures, never successes,
* permanent failures are surfaced as ``abandoned`` rather than retried forever,
* transient failures remain in the durable queue until delivery or the
  configured retry ceiling is reached.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request

from versioning import AICOS_VERSION


log = logging.getLogger("aicos.integrations")


class IntegrationBus:
    # Statuses where repeating the identical request is not useful without a
    # human/configuration change.  408/429 and 5xx are intentionally transient.
    PERMANENT_HTTP_STATUSES = frozenset({400, 401, 403, 404, 405, 410, 413, 422})
    _HTTP_DETAIL_RE = re.compile(r"^HTTP(?: Error)?\s+(\d{3})\b", re.IGNORECASE)

    def __init__(self, conn, lock) -> None:
        self.conn = conn
        self.max_attempts = max(1, int(os.environ.get("AICOS_INTEGRATION_MAX_ATTEMPTS", "12")))
        self.history_seconds = max(0.0, float(os.environ.get("AICOS_INTEGRATION_HISTORY_DAYS", "7")) * 86400.0)
        self.max_rows = max(100, int(os.environ.get("AICOS_INTEGRATION_MAX_ROWS", "50000")))
        self.abandoned = 0
        self._last_prune = 0.0
        self.lock = lock
        self.urls = [u.strip() for u in os.environ.get("AICOS_INTEGRATION_WEBHOOK_URLS", "").split(",") if u.strip()]
        self.secret = os.environ.get("AICOS_INTEGRATION_HMAC_SECRET", "").strip()
        self.stages = {s.strip().lower() for s in os.environ.get("AICOS_INTEGRATION_STAGES", "approved").split(",") if s.strip()}
        self._stop = threading.Event()
        self._init_db()
        self._thread = None
        if self.urls:
            self._thread = threading.Thread(target=self._loop, name="aicos-integration-outbox", daemon=True)
            self._thread.start()

    def _init_db(self):
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS integration_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dedup_key TEXT NOT NULL UNIQUE,
                    target_url TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    event_id INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    delivered_at REAL,
                    last_error TEXT DEFAULT ''
                )
                """
            )
            try:
                self.conn.execute("ALTER TABLE integration_outbox ADD COLUMN abandoned_at REAL")
            except Exception:
                pass   # already present — normal on every restart
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_integration_due "
                "ON integration_outbox(delivered_at,abandoned_at,next_attempt_at,id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_integration_done "
                "ON integration_outbox(delivered_at,abandoned_at)"
            )
            self.conn.commit()

    def _envelope(self, stage: str, event: dict) -> dict:
        return {
            "schema": "aicos.integration.event.v1",
            "aicos_version": AICOS_VERSION,
            "stage": stage,
            "sent_at": time.time(),
            "event": {
                "id": event.get("id"), "event_uuid": event.get("event_uuid"),
                "source_id": event.get("source_id"), "agent": event.get("agent"),
                "type": event.get("type"), "severity": event.get("severity"),
                "confidence": event.get("confidence"), "district": event.get("district"),
                "lat": event.get("lat"), "lon": event.get("lon"), "status": event.get("status"),
                "ts": event.get("ts"), "extra": event.get("extra") or {},
            },
        }

    def queue(self, stage: str, event: dict) -> int:
        stage = str(stage).lower()
        if not self.urls or stage not in self.stages:
            return 0
        payload = json.dumps(self._envelope(stage, event), separators=(",", ":"), ensure_ascii=False)
        with self.lock:
            before = self.conn.total_changes
            for url in self.urls:
                dedup = hashlib.sha256(
                    f"{url}|{stage}|{event.get('id')}|{event.get('status')}".encode()
                ).hexdigest()
                self.conn.execute(
                    "INSERT OR IGNORE INTO integration_outbox"
                    "(dedup_key,target_url,stage,event_id,payload,next_attempt_at) VALUES(?,?,?,?,?,?)",
                    (dedup, url, stage, int(event.get("id") or 0), payload, time.time()),
                )
            self.conn.commit()
            return self.conn.total_changes - before

    def _send(self, row) -> tuple[bool, str, bool]:
        """Return ``(delivered, detail, permanent)`` using the real HTTP code.

        ``urllib`` raises :class:`HTTPError` before a response object is returned
        for non-2xx responses, so production classification must inspect
        ``exc.code`` rather than parse ``str(exc)``.
        """
        body = str(row["payload"]).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"AICOS/{AICOS_VERSION}",
        }
        if self.secret:
            ts = str(int(time.time()))
            signature = hmac.new(
                self.secret.encode(), ts.encode() + b"." + body, hashlib.sha256
            ).hexdigest()
            headers["X-AICOS-Timestamp"] = ts
            headers["X-AICOS-Signature"] = "sha256=" + signature
        req = urllib.request.Request(str(row["target_url"]), data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                code = int(resp.status)
                ok = 200 <= code < 300
                return ok, f"HTTP {code}", False if ok else code in self.PERMANENT_HTTP_STATUSES
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
            detail = f"HTTP {code}: {getattr(exc, 'reason', '')}".rstrip()
            try:
                body_text = exc.read(512).decode("utf-8", errors="replace").strip()
                if body_text:
                    detail = f"{detail} — {body_text[:300]}"
            except Exception:
                pass
            return False, detail[:500], code in self.PERMANENT_HTTP_STATUSES
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return False, str(exc)[:500], False

    @classmethod
    def _normalise_send(cls, result) -> tuple[bool, str, bool]:
        """Normalize custom/test sender results without ever guessing success.

        Historical senders returned ``(ok, detail)``; v6.0.2 production sender
        returns ``(ok, detail, permanent)``.  Any other return shape is a
        transient failure.  This prevents a truthy string/dict from silently
        deleting an integration event.
        """
        if isinstance(result, tuple) and len(result) == 3:
            ok, detail, permanent = result
            return bool(ok), str(detail or ""), bool(permanent)
        if isinstance(result, tuple) and len(result) == 2:
            ok, detail = result
            detail = str(detail or "")
            permanent = False
            match = cls._HTTP_DETAIL_RE.match(detail)
            if not ok and match:
                permanent = int(match.group(1)) in cls.PERMANENT_HTTP_STATUSES
            return bool(ok), detail, permanent
        return False, f"unexpected sender result: {type(result).__name__}", False

    def flush_once(self, sender=None, limit: int = 20) -> int:
        sender = sender or self._send
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM integration_outbox WHERE delivered_at IS NULL "
                "AND abandoned_at IS NULL AND next_attempt_at <= ? ORDER BY id LIMIT ?",
                (time.time(), max(1, min(int(limit), 100))),
            ).fetchall()
        delivered = 0
        for row in rows:
            try:
                result = sender(row)
            except Exception as exc:
                result = (False, f"sender exception: {exc}", False)
            ok, detail, permanent = self._normalise_send(result)
            with self.lock:
                if ok:
                    self.conn.execute(
                        "UPDATE integration_outbox SET delivered_at=?,last_error='' WHERE id=?",
                        (time.time(), row["id"]),
                    )
                    delivered += 1
                else:
                    attempts = int(row["attempts"] or 0) + 1
                    if permanent or attempts >= self.max_attempts:
                        self.conn.execute(
                            "UPDATE integration_outbox SET attempts=?,abandoned_at=?,last_error=? WHERE id=?",
                            (attempts, time.time(), detail[:500], row["id"]),
                        )
                        self.abandoned += 1
                        log.error(
                            "integration delivery abandoned after %d attempt(s): %s",
                            attempts, detail[:300],
                        )
                    else:
                        delay = min(900.0, 2.0 ** min(attempts, 9))
                        self.conn.execute(
                            "UPDATE integration_outbox SET attempts=?,next_attempt_at=?,last_error=? WHERE id=?",
                            (attempts, time.time() + delay, detail[:500], row["id"]),
                        )
                self.conn.commit()
        self._maybe_prune()
        return delivered

    def _maybe_prune(self) -> None:
        """Bound integration history without hiding active delivery failures."""
        now = time.time()
        if now - self._last_prune < 300.0:
            return
        self._last_prune = now
        cutoff = now - self.history_seconds
        with self.lock:
            cur = self.conn.execute(
                "DELETE FROM integration_outbox "
                "WHERE (delivered_at IS NOT NULL AND delivered_at < ?) "
                "   OR (abandoned_at IS NOT NULL AND abandoned_at < ?)",
                (cutoff, cutoff),
            )
            removed = cur.rowcount or 0
            total = int(self.conn.execute(
                "SELECT COUNT(*) n FROM integration_outbox"
            ).fetchone()["n"])
            if total > self.max_rows:
                # Trim completed history first. Active rows are the only copy
                # of pending integration delivery and must not be silently lost.
                need = total - self.max_rows
                cur2 = self.conn.execute(
                    "DELETE FROM integration_outbox WHERE id IN ("
                    " SELECT id FROM integration_outbox "
                    " WHERE delivered_at IS NOT NULL OR abandoned_at IS NOT NULL "
                    " ORDER BY COALESCE(delivered_at,abandoned_at), id LIMIT ?)",
                    (need,),
                )
                removed += cur2.rowcount or 0
            if removed:
                self.conn.commit()
                log.info("integration outbox pruned: %d row(s) removed", removed)

    def _loop(self):
        while not self._stop.wait(2.0):
            self.flush_once()

    def status(self) -> dict:
        with self.lock:
            pending = int(self.conn.execute(
                "SELECT COUNT(*) n FROM integration_outbox "
                "WHERE delivered_at IS NULL AND abandoned_at IS NULL"
            ).fetchone()["n"])
            delivered = int(self.conn.execute(
                "SELECT COUNT(*) n FROM integration_outbox WHERE delivered_at IS NOT NULL"
            ).fetchone()["n"])
            retrying = int(self.conn.execute(
                "SELECT COUNT(*) n FROM integration_outbox "
                "WHERE delivered_at IS NULL AND abandoned_at IS NULL AND attempts > 0"
            ).fetchone()["n"])
            abandoned = int(self.conn.execute(
                "SELECT COUNT(*) n FROM integration_outbox WHERE abandoned_at IS NOT NULL"
            ).fetchone()["n"])
        return {
            "configured_targets": len(self.urls),
            "stages": sorted(self.stages),
            "hmac_signing": bool(self.secret),
            "pending": pending,
            "delivered": delivered,
            "retrying": retrying,
            "abandoned": abandoned,
            "max_attempts": self.max_attempts,
        }

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
