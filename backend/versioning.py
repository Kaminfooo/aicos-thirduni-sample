"""AICOS release version loader.

The repository-level VERSION file is the single source of truth.  Production
containers copy it to /app/VERSION; source-tree runs read it from the project
root.  AICOS_VERSION can override it for packaging/tests, but no runtime
module hard-codes a release number.
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
        Path("/app/VERSION"),
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
