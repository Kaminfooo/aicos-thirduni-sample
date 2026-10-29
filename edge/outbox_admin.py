#!/usr/bin/env python3
"""Offline administration utility for the AICOS edge delivery spool.

Use this on an edge node to inspect delivery state, export quarantined events,
or replay a dead-letter item after fixing credentials/schema/license problems.
It never contacts the backend itself; replay simply moves the event back to the
active durable outbox for the running worker to deliver.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path


def connect(path: str) -> sqlite3.Connection:
    if not os.path.isfile(path):
        raise SystemExit(f"outbox database not found: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def status(conn: sqlite3.Connection) -> dict:
    def count(table: str, where: str = "1=1") -> int:
        try:
            return int(conn.execute(f"SELECT COUNT(*) n FROM {table} WHERE {where}").fetchone()["n"])
        except sqlite3.OperationalError:
            return 0
    try:
        payload = int(conn.execute("SELECT COALESCE(SUM(LENGTH(payload)),0) n FROM outbox").fetchone()["n"])
    except sqlite3.OperationalError:
        payload = 0
    try:
        dlq_payload = int(conn.execute("SELECT COALESCE(SUM(LENGTH(payload)),0) n FROM dead_letter").fetchone()["n"])
    except sqlite3.OperationalError:
        dlq_payload = 0
    return {
        "queued": count("outbox"),
        "protected_queued": count("outbox", "protected=1"),
        "dead_letter": count("dead_letter"),
        "protected_dead_letter": count("dead_letter", "severity>=4"),
        "queued_payload_bytes": payload,
        "dead_letter_payload_bytes": dlq_payload,
    }


def list_dead(conn: sqlite3.Connection, limit: int) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT id,event_uuid,created_at,quarantined_at,severity,attempts,last_error,reason "
            "FROM dead_letter ORDER BY id DESC LIMIT ?", (max(1, min(limit, 1000)),)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in rows]


def export_dead(conn: sqlite3.Connection, dest: str) -> int:
    rows = conn.execute("SELECT * FROM dead_letter ORDER BY id").fetchall()
    payload = [dict(r) for r in rows]
    Path(dest).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(payload)


def requeue(conn: sqlite3.Connection, dlq_id: int) -> bool:
    row = conn.execute("SELECT * FROM dead_letter WHERE id=?", (dlq_id,)).fetchone()
    if row is None:
        return False
    try:
        payload = json.loads(row["payload"])
    except Exception:
        raise SystemExit("dead-letter payload is not valid JSON; export it for manual recovery")
    event_uuid = str(payload.get("event_uuid") or row["event_uuid"] or "")
    if not event_uuid:
        raise SystemExit("dead-letter item has no event_uuid and cannot be replayed safely")
    try:
        severity = max(1, min(5, int(payload.get("severity", row["severity"] or 1))))
    except (TypeError, ValueError):
        severity = 1
    conn.execute(
        "INSERT INTO outbox(event_uuid,created_at,payload,severity,protected,attempts,next_attempt_at,last_error) "
        "VALUES(?,?,?,?,?,0,?,?) ON CONFLICT(event_uuid) DO UPDATE SET "
        "payload=excluded.payload,severity=excluded.severity,protected=excluded.protected," 
        "attempts=0,next_attempt_at=excluded.next_attempt_at,last_error=excluded.last_error",
        (event_uuid, time.time(), row["payload"], severity, 1 if severity >= 4 else 0,
         time.time(), "manual replay from dead-letter queue"),
    )
    conn.execute("DELETE FROM dead_letter WHERE id=?", (dlq_id,))
    conn.commit()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect/recover the AICOS edge delivery spool")
    ap.add_argument("--db", default=os.environ.get("AICOS_OUTBOX_PATH", "aicos-outbox.db"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p_list = sub.add_parser("list-dead"); p_list.add_argument("--limit", type=int, default=50)
    p_export = sub.add_parser("export-dead"); p_export.add_argument("output")
    p_requeue = sub.add_parser("requeue"); p_requeue.add_argument("id", type=int)
    sub.add_parser("requeue-all")
    args = ap.parse_args()
    conn = connect(args.db)
    try:
        if args.cmd == "status":
            print(json.dumps(status(conn), indent=2))
        elif args.cmd == "list-dead":
            print(json.dumps(list_dead(conn, args.limit), indent=2, ensure_ascii=False))
        elif args.cmd == "export-dead":
            n = export_dead(conn, args.output); print(f"exported {n} dead-letter item(s) to {args.output}")
        elif args.cmd == "requeue":
            if not requeue(conn, args.id):
                print("dead-letter id not found", file=sys.stderr); return 2
            print(f"requeued dead-letter id {args.id}")
        elif args.cmd == "requeue-all":
            ids = [int(r[0]) for r in conn.execute("SELECT id FROM dead_letter ORDER BY id").fetchall()]
            moved = sum(1 for item_id in ids if requeue(conn, item_id))
            print(f"requeued {moved} dead-letter item(s)")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
