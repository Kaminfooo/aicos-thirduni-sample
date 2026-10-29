"""
AICOS Escalation Layer
----------------------
Sends alerts to the people who need to act: a webhook (which fronts any
dispatch system, Zapier, n8n, or an internal API), Telegram, and email.

Deliberate design: AICOS does NOT call emergency services automatically.
A detection is a recommendation; a human confirms it. So there are two
distinct moments, and you choose which ones notify:

    detected   a high-severity event needs an operator to look at it
    approved   an operator confirmed it — this is the dispatch signal

Configure in .env (all optional; unset channels are simply skipped):

    AICOS_NOTIFY_ON=approved        detected | approved | both | off
    AICOS_NOTIFY_MIN_SEVERITY=4
    AICOS_NOTIFY_WEBHOOK_URL=https://...
    AICOS_TELEGRAM_BOT_TOKEN=123:abc
    AICOS_TELEGRAM_CHAT_ID=-1001234567890
    AICOS_SMTP_HOST=smtp.example.com
    AICOS_SMTP_PORT=587
    AICOS_SMTP_USER=alerts@example.com
    AICOS_SMTP_PASSWORD=...
    AICOS_SMTP_TO=control-room@city.example.com

Every send runs on a background thread and swallows its own errors: a
failing notification must never delay or lose an event.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from email.message import EmailMessage

log = logging.getLogger("aicos.notify")

TIMEOUT = 8
VALID_MODES = {"off", "detected", "approved", "both"}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _mode() -> str:
    mode = _env("AICOS_NOTIFY_ON", "approved").lower()
    return mode if mode in VALID_MODES else "approved"


def _min_severity() -> int:
    try:
        return int(_env("AICOS_NOTIFY_MIN_SEVERITY", "4"))
    except ValueError:
        return 4


def should_notify(stage: str, severity: int) -> bool:
    """stage is 'detected' or 'approved'."""
    mode = _mode()
    if mode == "off":
        return False
    if mode != "both" and mode != stage:
        return False
    return int(severity or 0) >= _min_severity()


def _headline(stage: str, event: dict) -> str:
    verb = "CONFIRMED — DISPATCH" if stage == "approved" else "NEEDS REVIEW"
    return (f"AICOS {verb}: {event.get('type', 'event')} "
            f"(severity {event.get('severity')})")


def _body_lines(event: dict) -> list:
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.get("ts", time.time())))
    lines = [
        f"Incident:   INC-{int(event.get('id', 0)):06d}",
        f"Type:       {event.get('type', '')}",
        f"Severity:   {event.get('severity')} / 5",
        f"Confidence: {round(float(event.get('confidence', 0)) * 100)}%",
        f"Camera:     {event.get('source_id', '')}",
        f"District:   {event.get('district') or '-'}",
        f"Time:       {when}",
    ]
    lat, lon = event.get("lat") or 0.0, event.get("lon") or 0.0
    if lat or lon:
        lines.append(f"Location:   {lat:.5f}, {lon:.5f}")
        lines.append(f"Map:        https://www.google.com/maps?q={lat},{lon}")
    return lines


# ---------------------------------------------------------------------------
# Channels — each returns True on success, never raises
# ---------------------------------------------------------------------------


def _send_webhook(stage: str, event: dict) -> bool:
    url = _env("AICOS_NOTIFY_WEBHOOK_URL")
    if not url:
        return False
    payload = {"stage": stage, "headline": _headline(stage, event), "event": event}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "AICOS/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.warning("webhook failed: %s", exc)
        return False


def _send_telegram(stage: str, event: dict) -> bool:
    token = _env("AICOS_TELEGRAM_BOT_TOKEN")
    chat_id = _env("AICOS_TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    text = _headline(stage, event) + "\n\n" + "\n".join(_body_lines(event))
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        with urllib.request.urlopen(url, data=data, timeout=TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.warning("telegram failed: %s", exc)
        return False


def _send_email(stage: str, event: dict) -> bool:
    host = _env("AICOS_SMTP_HOST")
    to = _env("AICOS_SMTP_TO")
    user = _env("AICOS_SMTP_USER")
    if not host or not to:
        return False
    try:
        port = int(_env("AICOS_SMTP_PORT", "587"))
    except ValueError:
        port = 587
    msg = EmailMessage()
    msg["Subject"] = _headline(stage, event)
    msg["From"] = user or "aicos@localhost"
    msg["To"] = to
    msg.set_content("\n".join(_body_lines(event)))
    try:
        with smtplib.SMTP(host, port, timeout=TIMEOUT) as smtp:
            smtp.ehlo()
            try:
                smtp.starttls()
                smtp.ehlo()
            except smtplib.SMTPException:
                pass          # server without TLS — continue on the plain session
            password = _env("AICOS_SMTP_PASSWORD")
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
        return True
    except (smtplib.SMTPException, OSError) as exc:
        log.warning("email failed: %s", exc)
        return False


CHANNELS = (_send_webhook, _send_telegram, _send_email)


def _dispatch(stage: str, event: dict) -> None:
    sent = []
    for channel in CHANNELS:
        try:
            if channel(stage, event):
                sent.append(channel.__name__.replace("_send_", ""))
        except Exception as exc:                     # never let a channel escape
            log.warning("channel %s raised: %s", channel.__name__, exc)
    if sent:
        log.info("escalated INC-%06d (%s) via %s",
                 int(event.get("id", 0)), stage, ", ".join(sent))


def notify(stage: str, event: dict) -> bool:
    """Fire notifications on a background thread. Returns True if a send was
    started (i.e. the rules matched), not whether delivery succeeded."""
    if not should_notify(stage, event.get("severity", 0)):
        return False
    threading.Thread(target=_dispatch, args=(stage, dict(event)),
                     daemon=True, name="aicos-notify").start()
    return True
