#!/usr/bin/env python3
"""Offline administration CLI for AICOS signed assets.

Private keys used here should be kept OFF production servers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from crypto_utils import load_private_key, load_public_key, sign_envelope, verify_envelope, write_keypair
from licensing import LICENSE_SCHEMA
from certification import CERT_SCHEMA
from evidence import EVIDENCE_SCHEMA


def _write(path: str, value: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(p)


def cmd_keygen(a):
    key_id, pub = write_keypair(a.private, a.public)
    print(f"key_id={key_id}\npublic={pub}\nPRIVATE KEY: keep offline; do not copy into source control")


def cmd_license(a):
    now = time.time()
    payload = {
        "license_id": a.license_id or f"LIC-{uuid.uuid4().hex[:12].upper()}",
        "customer": a.customer,
        "deployment_id": a.deployment,
        "edition": a.edition,
        "issued_at": now,
        "not_before": now,
        "expires_at": now + a.days * 86400 if a.days else 0,
        "grace_days": a.grace_days,
        "max_cameras": a.max_cameras,
        "modules": a.modules or ["*"],
        "features": a.features or ["forensic_search", "signed_evidence", "camera_certification"],
        "support_tier": a.support_tier,
        "machine_ids": a.machine_id or [],
        "camera_ids": a.camera_id or [],
    }
    _write(a.out, sign_envelope(LICENSE_SCHEMA, payload, load_private_key(a.private_key)))


def _detector_assessments(survey: dict) -> dict:
    measured = survey.get("detector_assessments")
    if isinstance(measured, dict) and measured:
        return measured
    recommended = set(survey.get("recommended_detectors") or survey.get("detectors") or [])
    quality = float(survey.get("assurance_score", 0) or 0)
    out = {}
    names = ["person_down", "violence", "crowd", "panic", "intrusion", "loitering", "unattended_object", "snatch_theft", "traffic"]
    aliases = {"fight": "violence", "vehicles": "traffic", "abandoned_object": "unattended_object", "snatch": "snatch_theft"}
    normalized = {aliases.get(x, x) for x in recommended}
    for name in names:
        supported = name in normalized
        score = int(max(0, min(100, quality - (5 if name in {"violence", "panic", "snatch_theft"} else 0)))) if supported else 0
        out[name] = {"status": "certified" if supported and score >= 70 else ("conditional" if supported else "unsupported"), "score": score}
    return out


def cmd_certificate(a):
    survey = json.loads(Path(a.survey).read_text(encoding="utf-8"))
    # Accept either one camera record or a survey envelope containing cameras.
    if "cameras" in survey:
        matches = [x for x in survey["cameras"] if str(x.get("camera_id") or x.get("name")) == a.camera]
        if not matches:
            raise SystemExit(f"camera {a.camera!r} not found in survey")
        survey = matches[0]
    camera_id = str(survey.get("camera_id") or survey.get("name") or a.camera)
    now = time.time()
    payload = {
        "certificate_id": a.certificate_id or f"CERT-{uuid.uuid4().hex[:12].upper()}",
        "camera_id": camera_id,
        "issued_to": a.customer,
        "deployment_id": a.deployment,
        "issued_at": now,
        "not_before": now,
        "expires_at": now + a.days * 86400,
        "assurance_score": int(survey.get("assurance_score", 0) or 0),
        "metrics": survey.get("metrics", survey),
        "detectors": _detector_assessments(survey),
        "survey_sha256": __import__("hashlib").sha256(Path(a.survey).read_bytes()).hexdigest(),
    }
    _write(a.out, sign_envelope(CERT_SCHEMA, payload, load_private_key(a.private_key)))


def cmd_verify(a):
    env = json.loads(Path(a.file).read_text(encoding="utf-8"))
    schema = env.get("schema")
    if schema not in {LICENSE_SCHEMA, CERT_SCHEMA, EVIDENCE_SCHEMA}:
        raise SystemExit(f"unsupported schema: {schema}")
    payload = verify_envelope(env, load_public_key(a.public_key), schema)
    print(json.dumps({"ok": True, "schema": schema, "payload": payload}, indent=2, ensure_ascii=False))


def parser():
    p = argparse.ArgumentParser(description="AICOS offline signing administration")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("keygen"); s.add_argument("--private", required=True); s.add_argument("--public", required=True); s.set_defaults(fn=cmd_keygen)
    s = sp.add_parser("issue-license")
    s.add_argument("--private-key", required=True); s.add_argument("--out", required=True); s.add_argument("--customer", required=True); s.add_argument("--deployment", required=True)
    s.add_argument("--license-id"); s.add_argument("--edition", default="Government Sovereign"); s.add_argument("--days", type=int, default=365); s.add_argument("--grace-days", type=int, default=30)
    s.add_argument("--max-cameras", type=int, default=100); s.add_argument("--module", dest="modules", action="append"); s.add_argument("--feature", dest="features", action="append")
    s.add_argument("--support-tier", default="government"); s.add_argument("--machine-id", action="append"); s.add_argument("--camera-id", action="append"); s.set_defaults(fn=cmd_license)
    s = sp.add_parser("issue-camera-cert")
    s.add_argument("--private-key", required=True); s.add_argument("--survey", required=True); s.add_argument("--camera", required=True); s.add_argument("--customer", required=True); s.add_argument("--deployment", required=True); s.add_argument("--out", required=True)
    s.add_argument("--certificate-id"); s.add_argument("--days", type=int, default=365); s.set_defaults(fn=cmd_certificate)
    s = sp.add_parser("verify")
    s.add_argument("--public-key", required=True); s.add_argument("--file", required=True); s.set_defaults(fn=cmd_verify)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.fn(args)
