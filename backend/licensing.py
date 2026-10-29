"""Offline signed-license verification for AICOS.

No phone-home is required. A government deployment receives a signed JSON
license file and the runtime receives only the AICOS vendor PUBLIC key.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

from crypto_utils import load_public_key, verify_envelope

LICENSE_SCHEMA = "aicos.license.v1"

# Event types are mapped to commercial analytics modules. Unknown event types
# fall back to the agent's broad module rather than accidentally bypassing a
# restricted deployment.
_TYPE_MODULES = (
    (("fire", "flame", "smoke"), "fire_smoke"),
    (("person down", "fall", "medical"), "person_down"),
    (("fight", "altercation", "violence", "assault"), "violence"),
    (("crowd",), "crowd"),
    (("panic",), "panic"),
    (("intrusion",), "intrusion"),
    (("loiter",), "loitering"),
    (("unattended", "abandoned"), "unattended_object"),
    (("snatch", "theft"), "snatch_theft"),
    (("wrong-way", "wrong way", "stopped vehicle", "congestion", "vehicle"), "traffic"),
)


def module_for_event(event_type: str, agent: str = "") -> str:
    text = str(event_type or "").lower()
    for needles, module in _TYPE_MODULES:
        if any(n in text for n in needles):
            return module
    return {"traffic": "traffic", "emergency": "emergency", "security": "security"}.get(
        str(agent or "").lower(), "core"
    )


class LicenseError(RuntimeError):
    pass


class LicenseManager:
    def __init__(self) -> None:
        self.required = os.environ.get("AICOS_LICENSE_REQUIRED", "false").lower() in {"1", "true", "yes"}
        self.path = os.environ.get("AICOS_LICENSE_FILE", "").strip()
        self.public_key_path = os.environ.get("AICOS_LICENSE_PUBLIC_KEY", "").strip()
        self.machine_id = os.environ.get("AICOS_MACHINE_ID", "").strip()
        self.payload: dict | None = None
        self.error = ""
        self._load()

    def _load(self) -> None:
        if not self.path:
            if self.required:
                raise LicenseError("AICOS_LICENSE_FILE is required")
            return
        if not self.public_key_path:
            raise LicenseError("AICOS_LICENSE_PUBLIC_KEY is required when a license file is configured")
        try:
            envelope = json.loads(Path(self.path).read_text(encoding="utf-8"))
            public_key = load_public_key(self.public_key_path)
            payload = verify_envelope(envelope, public_key, LICENSE_SCHEMA)
            self._validate_claims(payload)
            self.payload = payload
        except Exception as exc:
            self.error = str(exc)
            raise LicenseError(f"invalid AICOS license: {exc}") from exc

    def _validate_claims(self, payload: dict) -> None:
        now = time.time()
        nb = float(payload.get("not_before", 0) or 0)
        exp = float(payload.get("expires_at", 0) or 0)
        grace_days = max(0.0, float(payload.get("grace_days", 0) or 0))
        if nb and now < nb:
            raise LicenseError("license is not active yet")
        if exp and now > exp + grace_days * 86400:
            raise LicenseError("license has expired beyond its grace period")
        machine_ids = [str(x) for x in payload.get("machine_ids", []) if str(x)]
        if machine_ids and self.machine_id not in machine_ids:
            raise LicenseError("this machine is not authorized by the license")
        if int(payload.get("max_cameras", 0) or 0) < 0:
            raise LicenseError("max_cameras cannot be negative")

    @property
    def enabled(self) -> bool:
        return self.payload is not None

    def status(self) -> dict:
        if not self.payload:
            return {"mode": "development-unlicensed", "required": self.required}
        p = self.payload
        now = time.time()
        exp = float(p.get("expires_at", 0) or 0)
        grace_days = float(p.get("grace_days", 0) or 0)
        phase = "active"
        if exp and now > exp:
            phase = "grace" if now <= exp + grace_days * 86400 else "expired"
        return {
            "mode": "signed-offline-license",
            "license_id": p.get("license_id"),
            "customer": p.get("customer"),
            "deployment_id": p.get("deployment_id"),
            "edition": p.get("edition", "Government"),
            "phase": phase,
            "expires_at": exp or None,
            "max_cameras": int(p.get("max_cameras", 0) or 0),
            "modules": list(p.get("modules", [])),
            "features": list(p.get("features", [])),
            "support_tier": p.get("support_tier", "standard"),
        }


    def assert_feature(self, feature: str) -> None:
        if not self.payload:
            if self.required:
                raise LicenseError("a valid signed license is required")
            return
        self._validate_claims(self.payload)
        features = {str(x) for x in self.payload.get("features", [])}
        if features and "*" not in features and feature not in features:
            raise LicenseError(f"feature {feature!r} is not licensed")

    def assert_camera_allowed(self, source_id: str, known_sources: Iterable[str]) -> None:
        if not self.payload:
            if self.required:
                raise LicenseError("a valid signed license is required")
            return
        self._validate_claims(self.payload)
        p = self.payload
        allowed_sources = {str(x) for x in p.get("camera_ids", []) if str(x)}
        if allowed_sources and source_id not in allowed_sources:
            raise LicenseError(f"camera {source_id!r} is not listed in this license")
        known = {str(x) for x in known_sources}
        max_cameras = int(p.get("max_cameras", 0) or 0)
        if max_cameras and source_id not in known and len(known) >= max_cameras:
            raise LicenseError(f"licensed camera-channel limit ({max_cameras}) reached")

    def assert_event_allowed(self, source_id: str, event_type: str, agent: str,
                             known_sources: Iterable[str]) -> None:
        if not self.payload:
            if self.required:
                raise LicenseError("a valid signed license is required")
            return
        self._validate_claims(self.payload)
        p = self.payload
        allowed_sources = {str(x) for x in p.get("camera_ids", []) if str(x)}
        if allowed_sources and source_id not in allowed_sources:
            raise LicenseError(f"camera {source_id!r} is not listed in this license")
        known = {str(x) for x in known_sources}
        max_cameras = int(p.get("max_cameras", 0) or 0)
        if max_cameras and source_id not in known and len(known) >= max_cameras:
            raise LicenseError(f"licensed camera-channel limit ({max_cameras}) reached")
        modules = {str(x) for x in p.get("modules", [])}
        if modules and "*" not in modules:
            needed = module_for_event(event_type, agent)
            broad = str(agent or "").lower()
            if needed not in modules and broad not in modules and "core" not in modules:
                raise LicenseError(f"analytics module {needed!r} is not licensed")
