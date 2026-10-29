"""
AICOS Incident Report Generator
-------------------------------
Builds a professional incident report from a stored event: incident
details, narrative, related events in the same area/time window, and
the full operator audit trail.

Two modes:
  1. Template mode (default) — fully offline, deterministic text.
  2. AI-polished mode — if AICOS_LLM_API_KEY is set, the draft is
     rewritten by an LLM into polished prose (facts are preserved).
     NOTE: this sends event METADATA (never video) to the AI provider.
     Leave the key unset for air-gapped / sovereign deployments.

Environment (all optional):
  AICOS_LLM_API_KEY   Anthropic API key; empty = template mode
  AICOS_LLM_MODEL     default "claude-sonnet-4-6"
  AICOS_REPORT_LANG   "en" (default) or "ar" — AI mode writes Arabic
"""

from __future__ import annotations

import json
import os

from versioning import AICOS_VERSION
import sqlite3
import time

RELATED_WINDOW_SECONDS = 1800  # ±30 minutes
RULE = "-" * 64
BAR = "=" * 64


def _fmt(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _fmt_t(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    if "extra" in d:
        try:
            d["extra"] = json.loads(d.get("extra") or "{}")
        except (TypeError, ValueError):
            d["extra"] = {}
    return d


def gather(conn: sqlite3.Connection, lock, event_id: int):
    """Collect the event, related events and audit rows. None if missing."""
    with lock:
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        ev = _row(row)
        related = conn.execute(
            """
            SELECT * FROM events
            WHERE id != ? AND ABS(ts - ?) <= ?
              AND (district = ? OR source_id = ?)
            ORDER BY ts
            LIMIT 25
            """,
            (event_id, ev["ts"], RELATED_WINDOW_SECONDS,
             ev.get("district", ""), ev.get("source_id", "")),
        ).fetchall()
        try:
            audit = conn.execute(
                "SELECT * FROM audit WHERE event_id = ? ORDER BY ts", (event_id,)
            ).fetchall()
        except sqlite3.OperationalError:  # older DB without audit table
            audit = []
    return {
        "event": ev,
        "related": [_row(r) for r in related],
        "audit": [dict(a) for a in audit],
    }


def _narrative(ev: dict, related_count: int) -> str:
    cam = ev.get("extra", {}).get("camera_name", "")
    cam_part = f" ({cam})" if cam else ""
    loc_bits = []
    if ev.get("district"):
        loc_bits.append(f"{ev['district']} district")
    if ev.get("lat") or ev.get("lon"):
        loc_bits.append(f"coordinates {ev.get('lat', 0.0):.5f}, {ev.get('lon', 0.0):.5f}")
    loc = " at ".join(loc_bits) if loc_bits else "an unspecified location"
    sev = int(ev.get("severity", 1))
    hitl = (" As a severity 4-5 event, it required operator review before any dispatch."
            if sev >= 4 else "")
    rel = (f" {related_count} related event(s) were recorded in the same area "
           f"within 30 minutes and are listed below." if related_count else "")
    return (
        f"On {_fmt(ev['ts'])}, the AICOS {ev.get('agent', 'system')} agent detected "
        f"\"{ev.get('type', 'event')}\" at {loc}, reported by source "
        f"{ev.get('source_id', 'unknown')}{cam_part} with a confidence of "
        f"{round(float(ev.get('confidence', 0)) * 100)}%. The event was classified "
        f"severity {sev} on a 1-5 scale.{hitl} Current status: "
        f"{str(ev.get('status', 'detected')).upper()}.{rel}"
    )


def render_text(data: dict) -> str:
    ev = data["event"]
    related = data["related"]
    audit = data["audit"]
    inc = f"INC-{int(ev['id']):06d}"
    cam = ev.get("extra", {}).get("camera_name", "")

    lines = [
        BAR,
        f" AICOS INCIDENT REPORT - {inc}",
        BAR,
        f"Generated:      {_fmt(time.time())}",
        f"Classification: {str(ev.get('agent', '')).upper()}  |  Severity {ev.get('severity')}/5"
        f"  |  Status: {str(ev.get('status', '')).upper()}",
        "",
        "1. INCIDENT DETAILS",
        RULE,
        f"Type:        {ev.get('type', '')}",
        f"Date/time:   {_fmt(ev['ts'])}",
        f"District:    {ev.get('district') or '-'}",
        f"Coordinates: {ev.get('lat', 0.0):.5f}, {ev.get('lon', 0.0):.5f}",
        f"Source:      {ev.get('source_id', '')}" + (f" - {cam}" if cam else ""),
        f"Confidence:  {round(float(ev.get('confidence', 0)) * 100)}%",
        "",
        "2. NARRATIVE",
        RULE,
        _narrative(ev, len(related)),
        "",
        "3. RELATED EVENTS (SAME AREA, +/- 30 MIN)",
        RULE,
    ]
    if related:
        for r in related:
            lines.append(
                f"{_fmt_t(r['ts'])}  {r.get('source_id', ''):<10} "
                f"{r.get('type', '')} (sev {r.get('severity')}, "
                f"{round(float(r.get('confidence', 0)) * 100)}%)"
            )
    else:
        lines.append("(none recorded)")
    lines += ["", "4. OPERATOR ACTIONS (AUDIT TRAIL)", RULE]
    if audit:
        for a in audit:
            lines.append(f"{_fmt_t(a['ts'])}  {str(a.get('action', '')):<10} {a.get('note', '')}")
    else:
        lines.append("(no audit rows recorded)")
    lines += [
        "",
        RULE,
        "DRAFT generated automatically by AICOS from sensor events.",
        "To be reviewed, corrected and signed by the responsible officer.",
        f"Report ID {inc}  |  AICOS v{AICOS_VERSION}",
    ]
    return "\n".join(lines)


def maybe_llm_polish(text: str, data: dict) -> str:
    """Rewrite the draft with an LLM if configured; on ANY failure,
    return the template text unchanged."""
    api_key = os.environ.get("AICOS_LLM_API_KEY", "").strip()
    if not api_key:
        return text
    lang = os.environ.get("AICOS_REPORT_LANG", "en").strip().lower()
    lang_line = ("Write the report in Modern Standard Arabic."
                 if lang == "ar" else "Write the report in clear professional English.")
    prompt = (
        "Rewrite the following automated incident report draft as a clear, "
        "professional, factual incident report suitable for police / city "
        "operations records. Keep ALL facts, identifiers, timestamps and "
        "numbers exactly as given. Do not invent any detail. Keep the final "
        "disclaimer lines. " + lang_line + "\n\n" + text
    )
    try:
        import requests
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            timeout=20,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": os.environ.get("AICOS_LLM_MODEL", "claude-sonnet-4-6"),
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if resp.status_code == 200:
            parts = resp.json().get("content", [])
            out = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            if out.strip():
                return out.strip() + "\n"
    except Exception:
        pass
    return text
