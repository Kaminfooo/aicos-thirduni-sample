#!/usr/bin/env python3
"""Verify SHA256SUMS.txt for an unpacked AICOS release."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print("SHA256SUMS.txt not found", file=sys.stderr)
        return 2
    failures = []
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            expected, rel = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed manifest line: {line}")
            continue
        path = ROOT / rel
        if not path.is_file():
            failures.append(f"missing: {rel}")
            continue
        actual = digest(path)
        checked += 1
        if actual != expected:
            failures.append(f"mismatch: {rel}")
    if failures:
        print("release verification FAILED")
        for item in failures:
            print(" -", item)
        return 1
    print(f"release verification OK: {checked} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
