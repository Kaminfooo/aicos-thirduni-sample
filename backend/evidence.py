"""Tamper-evident incident evidence signing for AICOS.

Each revision is an Ed25519-signed manifest containing immutable event facts,
SHA-256 hashes of media, and the previous global manifest hash. The result is
an append-only cryptographic chain suitable for later independent verification.
"""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path

from crypto_utils import canonical_json, file_sha256, load_private_key, load_public_key, sha256_hex, sign_envelope, verify_envelope
from cryptography.hazmat.primitives import serialization

EVIDENCE_SCHEMA = "aicos.evidence.v1"


class EvidenceError(RuntimeError):
    pass


class EvidenceManager:
    def __init__(self, conn, db_lock, snapshot_dir: str) -> None:
        self.conn = conn
        self.db_lock = db_lock
        self.snapshot_dir = Path(snapshot_dir)
        self.required = os.environ.get("AICOS_EVIDENCE_SIGNING_REQUIRED", "false").lower() in {"1", "true", "yes"}
        private_path = os.environ.get("AICOS_EVIDENCE_PRIVATE_KEY", "").strip()
        public_path = os.environ.get("AICOS_EVIDENCE_PUBLIC_KEY", "").strip()
        self.private_key = load_private_key(private_path) if private_path else None
        if public_path:
            self.public_key = load_public_key(public_path)
        elif self.private_key is not None:
            self.public_key = self.private_key.public_key()
        else:
            self.public_key = None
        if self.required and self.private_key is None:
            raise EvidenceError("AICOS_EVIDENCE_PRIVATE_KEY is required for signed-evidence mode")
        self._init_db()

    def _init_db(self) -> None:
        with self.db_lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_manifests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    envelope_json TEXT NOT NULL,
                    manifest_hash TEXT NOT NULL UNIQUE,
                    prev_hash TEXT DEFAULT '',
                    key_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL DEFAULT '',
                    UNIQUE(event_id, revision)
                )
                """
            )
            try:
                self.conn.execute("ALTER TABLE evidence_manifests ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
            except Exception:
                pass
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_event ON evidence_manifests(event_id, revision)")
            self.conn.commit()

    def _media_paths(self, event_id: int) -> tuple[Path, Path]:
        return self.snapshot_dir / f"{int(event_id)}.jpg", self.snapshot_dir / f"{int(event_id)}.gif"

    def sign_event(self, event_id: int) -> dict | None:
        if self.private_key is None:
            return None
        with self.db_lock:
            row = self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
            if row is None:
                raise EvidenceError("event not found")
            previous = self.conn.execute(
                "SELECT manifest_hash FROM evidence_manifests ORDER BY id DESC LIMIT 1"
            ).fetchone()
            revision_row = self.conn.execute(
                "SELECT COALESCE(MAX(revision), 0) AS n FROM evidence_manifests WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        event = dict(row)
        extra = event.get("extra") or "{}"
        try:
            extra = json.loads(extra)
        except Exception:
            extra = {}
        snap, clip = self._media_paths(event_id)
        try:
            with self.db_lock:
                audit_rows = self.conn.execute(
                    "SELECT ts, action, note FROM audit WHERE event_id = ? ORDER BY id", (event_id,)
                ).fetchall()
        except Exception:
            audit_rows = []
        audit = [{"ts": float(r["ts"]), "action": str(r["action"]), "note": str(r["note"] or "")} for r in audit_rows]
        content = {
            "evidence_version": 2,
            "event_id": int(event_id),
            "event_uuid": event.get("event_uuid"),
            "detected_at": float(event["ts"]),
            "source_id": event.get("source_id"),
            "agent": event.get("agent"),
            "type": event.get("type"),
            "severity": int(event.get("severity", 0)),
            "confidence": float(event.get("confidence", 0.0)),
            "district": event.get("district", ""),
            "lat": float(event.get("lat", 0.0)),
            "lon": float(event.get("lon", 0.0)),
            "status": str(event.get("status", "")),
            "audit": audit,
            "extra": extra,
            "snapshot_sha256": file_sha256(snap),
            "clip_sha256": file_sha256(clip),
        }
        content_hash = sha256_hex(canonical_json(content))
        with self.db_lock:
            existing = self.conn.execute(
                "SELECT * FROM evidence_manifests WHERE event_id = ? AND content_hash = ? ORDER BY revision DESC LIMIT 1",
                (event_id, content_hash),
            ).fetchone()
        if existing is not None:
            d = dict(existing)
            return {"revision": d["revision"], "manifest_hash": d["manifest_hash"],
                    "envelope": json.loads(d["envelope_json"]), "duplicate": True}
        payload = dict(content)
        payload["previous_manifest_sha256"] = str(previous["manifest_hash"]) if previous else ""
        payload["signed_at"] = time.time()
        envelope = sign_envelope(EVIDENCE_SCHEMA, payload, self.private_key)
        blob = canonical_json(envelope)
        manifest_hash = sha256_hex(blob)
        revision = int(revision_row["n"] if revision_row else 0) + 1
        with self.db_lock:
            self.conn.execute(
                """
                INSERT INTO evidence_manifests
                    (event_id, revision, created_at, envelope_json, manifest_hash, prev_hash, key_id, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, revision, time.time(), blob.decode("utf-8"), manifest_hash,
                 payload["previous_manifest_sha256"], envelope["key_id"], content_hash),
            )
            self.conn.commit()
        return {"revision": revision, "manifest_hash": manifest_hash, "envelope": envelope}

    def latest(self, event_id: int) -> dict | None:
        with self.db_lock:
            row = self.conn.execute(
                "SELECT * FROM evidence_manifests WHERE event_id = ? ORDER BY revision DESC LIMIT 1",
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["envelope"] = json.loads(d.pop("envelope_json"))
        return d

    def verify(self, event_id: int) -> dict:
        item = self.latest(event_id)
        if item is None:
            return {"ok": False, "reason": "no signed evidence manifest"}
        if self.public_key is None:
            return {"ok": False, "reason": "no evidence public key configured"}
        try:
            payload = verify_envelope(item["envelope"], self.public_key, EVIDENCE_SCHEMA)
            actual_manifest_hash = sha256_hex(canonical_json(item["envelope"]))
            if actual_manifest_hash != item["manifest_hash"]:
                raise EvidenceError("stored manifest hash mismatch")
            snap, clip = self._media_paths(event_id)
            if payload.get("snapshot_sha256") != file_sha256(snap):
                raise EvidenceError("snapshot hash mismatch")
            if payload.get("clip_sha256") != file_sha256(clip):
                raise EvidenceError("clip hash mismatch")
            prev_hash = str(payload.get("previous_manifest_sha256") or "")
            if prev_hash:
                with self.db_lock:
                    exists = self.conn.execute(
                        "SELECT 1 FROM evidence_manifests WHERE manifest_hash = ? LIMIT 1", (prev_hash,)
                    ).fetchone()
                if exists is None:
                    raise EvidenceError("previous manifest in chain is missing")
            return {"ok": True, "revision": item["revision"], "manifest_hash": item["manifest_hash"], "key_id": item["key_id"]}
        except Exception as exc:
            return {"ok": False, "reason": str(exc), "manifest_hash": item.get("manifest_hash")}


    def verify_chain(self) -> dict:
        """Verify every stored manifest signature and global hash-chain link.

        This proves the manifest ledger has not been reordered, removed or
        edited. Media retention is intentionally checked separately by
        ``verify(event_id)`` because old JPEG/GIF evidence may be lawfully
        pruned while the signed manifest ledger remains.
        """
        if self.public_key is None:
            return {"ok": False, "reason": "no evidence public key configured"}
        with self.db_lock:
            rows = self.conn.execute("SELECT * FROM evidence_manifests ORDER BY id ASC").fetchall()
        previous = ""
        checked = 0
        for row in rows:
            item = dict(row)
            try:
                envelope = json.loads(item["envelope_json"])
                payload = verify_envelope(envelope, self.public_key, EVIDENCE_SCHEMA)
                actual = sha256_hex(canonical_json(envelope))
                if actual != item["manifest_hash"]:
                    raise EvidenceError("manifest hash mismatch")
                if str(payload.get("previous_manifest_sha256") or "") != previous:
                    raise EvidenceError("chain link mismatch")
                previous = item["manifest_hash"]
                checked += 1
            except Exception as exc:
                return {"ok": False, "checked": checked, "failed_manifest_id": item.get("id"), "reason": str(exc)}
        return {"ok": True, "checked": checked, "head": previous}

    def export_package(self, event_id: int) -> bytes:
        item = self.latest(event_id)
        if item is None:
            raise EvidenceError("no signed evidence manifest")
        verification = self.verify(event_id)
        chain = self.verify_chain()
        snap, clip = self._media_paths(event_id)
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(item, indent=2, ensure_ascii=False))
            zf.writestr("verification.json", json.dumps({"event": verification, "chain": chain}, indent=2, ensure_ascii=False))
            if self.public_key is not None:
                pem = self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                zf.writestr("evidence-public-key.pem", pem)
            if snap.is_file():
                zf.write(snap, arcname="snapshot.jpg")
            if clip.is_file():
                zf.write(clip, arcname="clip.gif")
            zf.writestr("README.txt",
                "AICOS signed evidence package. Verify manifest.json with evidence-public-key.pem, "
                "then compare the SHA-256 media hashes in the signed payload.\n")
        return out.getvalue()
