"""AICOS release version loader for edge services.

VERSION at the repository root is the single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_version() -> str:
    override = os.environ.get("AICOS_VERSION", "").strip()
    if override:
        return override
    candidates = [
        Path(__file__).resolve().parents[1] / "VERSION",
        Path.cwd() / "VERSION",
    ]
    for path in candidates:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return "0+unknown"


AICOS_VERSION = _load_version()
