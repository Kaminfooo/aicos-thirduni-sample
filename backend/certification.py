"""Signed per-camera operational certificates for AICOS.

Certificates are generated from a measured camera survey and signed offline.
The runtime can optionally refuse events from cameras/detectors that are not
currently certified.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from crypto_utils import load_public_key, verify_envelope
from licensing import module_for_event

CERT_SCHEMA = "aicos.camera-cert.v1"


class CameraCertificateError(RuntimeError):
    pass


class CameraCertificateRegistry:
    def __init__(self) -> None:
        self.required = os.environ.get("AICOS_REQUIRE_CAMERA_CERT", "false").lower() in {"1", "true", "yes"}
        self.directory = os.environ.get("AICOS_CAMERA_CERT_DIR", "").strip()
        self.public_key_path = os.environ.get("AICOS_CERT_PUBLIC_KEY", "").strip()
        self.certificates: dict[str, dict] = {}
        if self.directory:
            self.reload()
        elif self.required:
            raise CameraCertificateError("AICOS_CAMERA_CERT_DIR is required")

    def reload(self) -> None:
        if not self.public_key_path:
            raise CameraCertificateError("AICOS_CERT_PUBLIC_KEY is required")
        public_key = load_public_key(self.public_key_path)
        directory = Path(self.directory)
        if not directory.is_dir():
            raise CameraCertificateError(f"certificate directory not found: {directory}")
        loaded: dict[str, dict] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                payload = verify_envelope(envelope, public_key, CERT_SCHEMA)
                camera_id = str(payload.get("camera_id", "")).strip()
                if not camera_id:
                    raise ValueError("camera_id missing")
                loaded[camera_id] = payload
            except Exception as exc:
                raise CameraCertificateError(f"invalid camera certificate {path.name}: {exc}") from exc
        self.certificates = loaded

    def _validate_time(self, cert: dict) -> None:
        now = time.time()
        nb = float(cert.get("not_before", 0) or 0)
        exp = float(cert.get("expires_at", 0) or 0)
        if nb and now < nb:
            raise CameraCertificateError("camera certificate is not active yet")
        if exp and now > exp:
            raise CameraCertificateError("camera certificate has expired")

    def assert_camera_current(self, camera_id: str) -> None:
        if not self.required and not self.certificates:
            return
        cert = self.certificates.get(str(camera_id))
        if cert is None:
            if self.required:
                raise CameraCertificateError(f"camera {camera_id!r} has no operational certificate")
            return
        self._validate_time(cert)

    def assert_event_allowed(self, camera_id: str, event_type: str, agent: str) -> None:
        if not self.required and not self.certificates:
            return
        cert = self.certificates.get(str(camera_id))
        if cert is None:
            if self.required:
                raise CameraCertificateError(f"camera {camera_id!r} has no operational certificate")
            return
        self._validate_time(cert)
        module = module_for_event(event_type, agent)
        detectors = cert.get("detectors", {}) or {}
        assessment = detectors.get(module)
        if assessment is None:
            # Broad modules are intentionally not accepted as proof that a
            # detector was tested on this particular view.
            if self.required:
                raise CameraCertificateError(f"detector {module!r} is not certified for {camera_id}")
            return
        status = str(assessment.get("status", "")).lower() if isinstance(assessment, dict) else str(assessment).lower()
        if status not in {"certified", "conditional"}:
            raise CameraCertificateError(f"detector {module!r} is {status or 'not certified'} on {camera_id}")

    def summary(self) -> dict:
        items = []
        for camera_id, cert in sorted(self.certificates.items()):
            items.append({
                "camera_id": camera_id,
                "certificate_id": cert.get("certificate_id"),
                "expires_at": cert.get("expires_at"),
                "assurance_score": cert.get("assurance_score"),
                "detectors": cert.get("detectors", {}),
            })
        return {"required": self.required, "count": len(items), "certificates": items}
