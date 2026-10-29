#!/usr/bin/env python3
"""
AICOS Preflight
---------------
Verifies a deployment BEFORE you run the worker, so problems surface in
seconds instead of after a long download or a twenty-minute test run.

Checks, in order of how often they bite:

  1. Python environment      virtualenv active, version sane
  2. Dependencies            core, YOLO and probe groups, each reported
  3. Configuration           cameras.yaml parses and means what you think
  4. Camera sources          every source actually reachable
  5. Backend                 health endpoint and API key both verified
  6. Model weights           present, loadable, correct bias layout
  7. CPU capacity            measured, with a process_every_n_frames advice
  8. Disk                    room for snapshots and the event database

Usage:
    python preflight.py
    python preflight.py --config ../config/cameras.yaml \\
                        --backend http://127.0.0.1:8080 --key <api-key>

Exit code is 0 when nothing FAILED, 1 otherwise — so it can gate a
deployment script.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

from versioning import AICOS_VERSION

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MARK = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail="", hint=""):
        self.rows.append((status, name, detail, hint))
        line = f"[{MARK[status]}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)
        if hint and status != PASS:
            print(f"           -> {hint}")
        sys.stdout.flush()

    @property
    def failures(self):
        return [r for r in self.rows if r[0] == FAIL]

    @property
    def warnings(self):
        return [r for r in self.rows if r[0] == WARN]


# ---------------------------------------------------------------------------
# 1. environment
# ---------------------------------------------------------------------------


def check_python(rep: Report) -> None:
    in_venv = (hasattr(sys, "real_prefix")
               or sys.prefix != getattr(sys, "base_prefix", sys.prefix))
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if not in_venv:
        rep.add(WARN, "python environment", f"Python {ver}, NOT in a virtualenv",
                "run: source .venv/bin/activate  (pip installs will otherwise be "
                "blocked by 'externally-managed-environment')")
    elif sys.version_info < (3, 9):
        rep.add(FAIL, "python environment", f"Python {ver} is too old",
                "AICOS needs Python 3.9 or newer")
    else:
        rep.add(PASS, "python environment", f"Python {ver} in virtualenv")


DEP_GROUPS = {
    "core (worker + backend)": [("cv2", "opencv-python-headless"),
                                ("numpy", "numpy"),
                                ("yaml", "PyYAML"),
                                ("requests", "requests")],
    "YOLO detectors": [("ultralytics", "ultralytics"), ("torch", "torch")],
    "violence probe (optional)": [("transformers", "transformers"), ("av", "av")],
}


def check_dependencies(rep: Report) -> dict:
    present = {}
    for group, mods in DEP_GROUPS.items():
        missing = []
        for module, package in mods:
            try:
                importlib.import_module(module)
                present[module] = True
            except ImportError:
                present[module] = False
                missing.append(package)
            except Exception as exc:
                # imports that fail for reasons other than absence — the classic
                # is cv2 needing system graphics libraries on a headless server
                present[module] = False
                hint = ("sudo apt -y install libgl1 libglib2.0-0t64"
                        if "libGL" in str(exc) or module == "cv2"
                        else f"import error: {exc}")
                rep.add(FAIL, f"dependency {module}", str(exc)[:70], hint)
                missing.append(package)
        optional = "optional" in group
        if not missing:
            rep.add(PASS, f"dependencies: {group}", "all present")
        elif optional:
            rep.add(WARN, f"dependencies: {group}", "missing " + ", ".join(missing),
                    "only needed for violence_probe.py — pip install " + " ".join(missing))
        else:
            rep.add(FAIL, f"dependencies: {group}", "missing " + ", ".join(missing),
                    "pip install " + " ".join(missing))
    return present


# ---------------------------------------------------------------------------
# 2. configuration
# ---------------------------------------------------------------------------


def check_config(rep: Report, path: str):
    if not os.path.isfile(path):
        rep.add(FAIL, "config file", f"not found: {path}",
                "cp ../config/cameras.example.yaml ../config/cameras.yaml")
        return None
    try:
        import yaml
    except ImportError:
        rep.add(FAIL, "config file", "cannot parse — PyYAML missing")
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:
        rep.add(FAIL, "config file", f"invalid YAML: {str(exc)[:80]}",
                "check indentation; tabs are not valid YAML")
        return None

    cameras = cfg.get("cameras") or []
    if not cameras:
        rep.add(FAIL, "config file", "no cameras defined")
        return cfg
    ids = [c.get("id") for c in cameras]
    if len(set(ids)) != len(ids):
        rep.add(FAIL, "config cameras", "duplicate camera ids",
                "each camera needs a unique id — events are keyed on it")
    missing_url = [c.get("id") for c in cameras if not c.get("rtsp_url")]
    if missing_url:
        rep.add(FAIL, "config cameras", f"no rtsp_url on: {missing_url}")
    rep.add(PASS, "config file", f"{len(cameras)} camera(s): {', '.join(str(i) for i in ids)}")

    settings = cfg.get("settings") or {}
    no_coords = [c.get("id") for c in cameras
                 if not (c.get("lat") or c.get("lon"))]
    if no_coords:
        rep.add(WARN, "camera coordinates", f"lat/lon are 0 on: {no_coords}",
                "events from these cameras will not appear on the dispatch map")

    ycfg = ((settings.get("detectors") or {}).get("yolo") or {})
    if ycfg.get("enabled"):
        every_n = int(settings.get("process_every_n_frames", 12) or 12)
        rep.add(PASS, "detectors", f"YOLO enabled, imgsz={ycfg.get('imgsz', 640)}, "
                                   f"analyzing 1 frame in {every_n}")
    else:
        rep.add(WARN, "detectors", "YOLO disabled — only the motion demo will run",
                "set settings.detectors.yolo.enabled: true")

    for cam in cameras:
        for zone in (cam.get("zones") or []):
            pts = zone.get("points") or []
            if len(pts) < 3:
                rep.add(FAIL, f"zone '{zone.get('name')}'",
                        f"needs at least 3 points, has {len(pts)}")
            elif any(not (0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0)
                     for p in pts):
                rep.add(FAIL, f"zone '{zone.get('name')}'", "points outside 0.0-1.0",
                        "zone points are FRACTIONS of the frame, not pixels")
    return cfg


# ---------------------------------------------------------------------------
# 3. camera sources
# ---------------------------------------------------------------------------


def check_sources(rep: Report, cfg: dict, deps: dict) -> None:
    if not cfg:
        return
    for cam in (cfg.get("cameras") or []):
        cam_id = cam.get("id", "?")
        src = str(cam.get("rtsp_url", ""))
        low = src.lower()

        if src.isdigit():
            rep.add(PASS, f"source {cam_id}", f"local camera device {src}")
            continue

        if low.startswith(("http://", "https://")):
            base = low.split("?")[0]
            if base.endswith(IMAGE_EXTS):
                ok, detail = _probe_http_image(src)
                rep.add(PASS if ok else FAIL, f"source {cam_id}",
                        detail, "" if ok else "open the URL in a browser to confirm it")
            elif "youtube.com/" in low or "youtu.be/" in low:
                found = shutil.which("yt-dlp")
                rep.add(PASS if found else FAIL, f"source {cam_id}",
                        "YouTube live" + ("" if found else " — yt-dlp not installed"),
                        "" if found else "pip install yt-dlp  (note: YouTube blocks "
                                         "most datacenter IPs)")
            else:
                rep.add(WARN, f"source {cam_id}", "remote stream — not verified here",
                        "the worker will report if it cannot connect")
            continue

        if low.startswith(("rtsp://", "rtsps://", "rtmp://")):
            if not deps.get("cv2"):
                rep.add(WARN, f"source {cam_id}", "RTSP — cannot test, cv2 missing")
                continue
            ok, detail = _probe_stream(src)
            rep.add(PASS if ok else FAIL, f"source {cam_id}", detail,
                    "" if ok else "check the URL, credentials and camera VLAN access")
            continue

        # local file
        if not os.path.isfile(src):
            rep.add(FAIL, f"source {cam_id}", f"file not found: {src}",
                    "the worker would treat this as a live stream and retry forever")
        elif not src.lower().endswith(VIDEO_EXTS):
            rep.add(WARN, f"source {cam_id}", f"unusual extension: {src}")
        elif not deps.get("cv2"):
            rep.add(WARN, f"source {cam_id}", "file exists — cannot open, cv2 missing")
        else:
            ok, detail = _probe_stream(src)
            rep.add(PASS if ok else FAIL, f"source {cam_id}", detail,
                    "" if ok else "file exists but cannot be decoded — is it a valid video?")


def _probe_http_image(url: str):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AICOS-preflight"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(4096)
            if resp.status != 200:
                return False, f"HTTP {resp.status}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, f"unreachable: {str(exc)[:50]}"
    if raw[:2] == b"\xff\xd8":
        return True, "snapshot camera reachable (JPEG)"
    if raw[:8].startswith(b"\x89PNG"):
        return True, "snapshot camera reachable (PNG)"
    return False, "reachable but did not return an image"


def _probe_stream(src: str):
    import cv2
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        return False, "cannot open"
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return False, "opened but no frames"
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    detail = f"{w}x{h}"
    if fps:
        detail += f" @ {fps:.0f}fps"
    if count:
        detail += f", {count} frames"
    if w < 640:
        return True, detail + "  (LOW RESOLUTION — people may be too small to detect)"
    return True, detail


# ---------------------------------------------------------------------------
# 4. backend
# ---------------------------------------------------------------------------


def check_secrets(rep: Report, device_key: str, operator_key: str,
                  allow_shared_operator_keys: bool | None = None,
                  viewer_key: str = "") -> None:
    """Validate credentials using the same rules as the backend.

    In Government RBAC mode (shared keys disabled), AICOS_OPERATOR_KEY is
    intentionally blank and must not make preflight fail.  When shared keys
    are explicitly enabled for a pilot/legacy deployment, operator/viewer
    credentials are validated and must remain distinct from the device key.
    """
    weak = {"change-me", "changeme", "password", "secret",
            "local-test-key-12345", "aicos", "test"}
    if allow_shared_operator_keys is None:
        allow_shared_operator_keys = os.environ.get(
            "AICOS_ALLOW_SHARED_OPERATOR_KEYS", "false"
        ).lower() in {"1", "true", "yes"}

    def validate(label: str, value: str, var: str, required: bool = True) -> bool:
        if not value:
            if required:
                rep.add(FAIL, label, f"{var} is not set",
                        "generate one with: python -c \"import secrets; "
                        "print(secrets.token_urlsafe(32))\"")
                return False
            rep.add(PASS, label, f"{var} disabled in RBAC mode")
            return True
        if value.lower() in weak or len(value) < 24:
            rep.add(FAIL, label, f"{var} is weak ({len(value)} chars, 24 minimum)",
                    "generate a strong independent credential")
            return False
        rep.add(PASS, label, f"{var} set, {len(value)} chars")
        return True

    device_ok = validate("device key", device_key, "AICOS_API_KEY", required=True)
    if allow_shared_operator_keys:
        operator_ok = validate("operator key", operator_key, "AICOS_OPERATOR_KEY", required=True)
        viewer_ok = validate("viewer key", viewer_key, "AICOS_VIEWER_KEY", required=False) if viewer_key else True
        supplied = [("device", device_key), ("operator", operator_key)]
        if viewer_key:
            supplied.append(("viewer", viewer_key))
        values = [v for _, v in supplied if v]
        if len(values) != len(set(values)):
            rep.add(FAIL, "role separation", "two or more shared credentials are identical",
                    "device, operator and viewer credentials must be generated independently")
        elif device_ok and operator_ok and viewer_ok:
            rep.add(PASS, "role separation", "device/operator/viewer credentials are distinct")
    else:
        if operator_key or viewer_key:
            rep.add(WARN, "shared operator keys",
                    "RBAC mode is enabled; AICOS_OPERATOR_KEY/AICOS_VIEWER_KEY are ignored",
                    "remove unused shared keys from the environment to reduce credential exposure")
        else:
            rep.add(PASS, "shared operator keys", "disabled; local RBAC sessions are required")
        if device_ok:
            rep.add(PASS, "role separation", "device credential cannot authorize operator actions")


def check_outbox(rep: Report) -> None:
    """A queued event carries a snapshot and a clip — roughly 0.6 MB each."""
    path = os.environ.get("AICOS_OUTBOX_PATH", "aicos-outbox.db")
    try:
        max_mb = float(os.environ.get("AICOS_OUTBOX_MAX_MB", "256"))
        max_rows = int(os.environ.get("AICOS_OUTBOX_MAX_ROWS", "5000"))
        min_free_mb = float(os.environ.get("AICOS_OUTBOX_MIN_FREE_MB", "512"))
        if max_mb <= 0 or max_rows <= 0 or min_free_mb < 0:
            raise ValueError("limits must be positive")
    except ValueError:
        rep.add(FAIL, "outbox limits",
                "AICOS_OUTBOX_MAX_MB / _MAX_ROWS / _MIN_FREE_MB are invalid",
                "set positive numeric storage limits")
        return
    try:
        free_mb = shutil.disk_usage(
            os.path.dirname(os.path.abspath(path)) or ".").free / 1e6
    except OSError:
        free_mb = None
    detail = f"payload cap {max_mb:.0f} MB / {max_rows} events, reserve {min_free_mb:.0f} MB free"
    if free_mb is not None and max_mb > free_mb * 0.5:
        rep.add(WARN, "outbox limits", detail,
                f"only {free_mb:.0f} MB free — a full outbox would take more "
                f"than half of it. Lower AICOS_OUTBOX_MAX_MB.")
    else:
        rep.add(PASS, "outbox limits",
                detail + (f", {free_mb:.0f} MB free" if free_mb is not None else ""))
    if os.path.isfile(path):
        size_mb = sum(
            os.path.getsize(candidate) for candidate in (path, path + "-wal", path + "-shm")
            if os.path.isfile(candidate)
        ) / 1e6
        if size_mb > max_mb * 0.8:
            rep.add(WARN, "outbox backlog", f"{size_mb:.1f} MB already queued",
                    "the backend has been unreachable — check it before starting")
        else:
            rep.add(PASS, "outbox backlog", f"{size_mb:.1f} MB queued")


def check_backend(rep: Report, base_url: str, api_key: str) -> None:
    base = (base_url or "").rstrip("/")
    if not base:
        rep.add(WARN, "backend", "no URL given — skipped")
        return
    try:
        with urllib.request.urlopen(base + "/api/health", timeout=8) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        version = body.get("version", "?")
        rep.add(PASS, "backend health", f"{base} responding, version {version}")
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        rep.add(FAIL, "backend health", f"{base} unreachable: {str(exc)[:50]}",
                "on the server: docker compose up -d && curl localhost:8080/api/health")
        return

    if not api_key:
        rep.add(WARN, "backend API key", "not supplied — cannot verify",
                "pass --key, or export AICOS_API_KEY")
        return

    # Send a deliberately invalid body: 401 means the key was rejected,
    # 422 means the key was accepted and only validation failed. Nothing
    # is ever stored either way.
    req = urllib.request.Request(
        base + "/api/events", data=b"{}", method="POST",
        headers={"Content-Type": "application/json", "X-AICOS-KEY": api_key})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            code = resp.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except (urllib.error.URLError, OSError) as exc:
        rep.add(WARN, "backend API key", f"could not verify: {str(exc)[:40]}")
        return

    if code == 401:
        rep.add(FAIL, "backend API key", "rejected (401)",
                "key must match AICOS_API_KEY in the backend .env — "
                "grep '^AICOS_API_KEY=' ../.env | cut -d= -f2")
    elif code == 422:
        rep.add(PASS, "backend API key", "accepted")
    else:
        rep.add(WARN, "backend API key", f"unexpected response {code}")


# ---------------------------------------------------------------------------
# 5. model weights
# ---------------------------------------------------------------------------


def check_models(rep: Report, cfg: dict, deps: dict) -> str:
    if not cfg:
        return ""
    settings = cfg.get("settings") or {}
    ycfg = ((settings.get("detectors") or {}).get("yolo") or {})
    weights = str(ycfg.get("model", "yolov8n-pose.pt"))
    if not ycfg.get("enabled"):
        return ""
    if not deps.get("ultralytics"):
        rep.add(FAIL, "pose model", "ultralytics not installed")
        return ""
    if os.path.isfile(weights):
        size = os.path.getsize(weights) / 1e6
        rep.add(PASS, "pose model", f"{weights} present ({size:.1f} MB)")
    else:
        rep.add(WARN, "pose model", f"{weights} not local — will download on first run",
                "needs internet on this machine the first time")

    vcfg = ((settings.get("detectors") or {}).get("violence") or {})
    if vcfg.get("enabled"):
        vw = str(vcfg.get("weights", ""))
        if os.path.isfile(vw):
            rep.add(PASS, "violence model", f"{vw} present "
                                            f"({os.path.getsize(vw)/1e6:.1f} MB)")
        else:
            rep.add(FAIL, "violence model", f"weights not found: {vw}",
                    "see docs/VIOLENCE-MODEL-GUIDE.md to download them, or set "
                    "violence.enabled: false")
    return weights


# ---------------------------------------------------------------------------
# 6. capacity
# ---------------------------------------------------------------------------


def check_capacity(rep: Report, cfg: dict, deps: dict, weights: str) -> None:
    cores = os.cpu_count() or 1
    if not (deps.get("ultralytics") and deps.get("cv2") and weights):
        rep.add(WARN, "cpu capacity", f"{cores} core(s) — inference not benchmarked")
        return
    try:
        import numpy as np
        from ultralytics import YOLO
        try:
            import torch
            nnp = getattr(torch.backends, "nnpack", None)
            if nnp is not None:
                try:
                    nnp.enabled = False
                except Exception:
                    pass
        except Exception:
            pass
        settings = cfg.get("settings") or {}
        ycfg = ((settings.get("detectors") or {}).get("yolo") or {})
        imgsz = int(ycfg.get("imgsz", 640))
        model = YOLO(weights)
        blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model.predict(blank, imgsz=imgsz, verbose=False)          # warm up
        t0 = time.time()
        runs = 3
        for _ in range(runs):
            model.predict(blank, imgsz=imgsz, verbose=False)
        per = (time.time() - t0) / runs
    except Exception as exc:
        rep.add(WARN, "cpu capacity", f"benchmark failed: {str(exc)[:60]}")
        return

    fps = 1.0 / per if per else 0.0
    cams = len(cfg.get("cameras") or []) or 1
    detail = f"{cores} core(s), {per*1000:.0f} ms per frame ({fps:.1f} fps total)"
    budget = fps / cams
    if budget < 0.5:
        rep.add(WARN, "cpu capacity", detail,
                f"only {budget:.1f} analyses/sec per camera across {cams} camera(s) — "
                f"reduce imgsz, raise process_every_n_frames, or run fewer cameras")
    else:
        settings = cfg.get("settings") or {}
        every_n = int(settings.get("process_every_n_frames", 12) or 12)
        # a 30fps source analyzed every N frames needs 30/N analyses per second
        need = 30.0 / max(1, every_n)
        advice = ""
        if need > budget:
            suggested = int(30.0 / max(0.1, budget)) + 5
            advice = (f"at process_every_n_frames={every_n} this machine would fall "
                      f"behind; try {suggested}")
        rep.add(PASS if not advice else WARN, "cpu capacity", detail, advice)


def check_disk(rep: Report, path: str = ".") -> None:
    try:
        usage = shutil.disk_usage(os.path.abspath(path))
    except OSError as exc:
        rep.add(WARN, "disk space", f"could not check: {exc}")
        return
    free_gb = usage.free / 1e9
    detail = f"{free_gb:.1f} GB free"
    if free_gb < 1:
        rep.add(FAIL, "disk space", detail, "snapshots and the event DB need room")
    elif free_gb < 5:
        rep.add(WARN, "disk space", detail, "consider lowering SNAPSHOT_RETENTION_DAYS")
    else:
        rep.add(PASS, "disk space", detail)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="AICOS preflight checks")
    ap.add_argument("--config", default="../config/cameras.yaml")
    ap.add_argument("--backend", default=os.environ.get("BACKEND_URL", ""))
    ap.add_argument("--key", default=os.environ.get("AICOS_API_KEY", ""),
                    help="device/ingest key")
    ap.add_argument("--operator-key", default=os.environ.get("AICOS_OPERATOR_KEY", ""),
                    help="legacy shared operator key (optional when RBAC mode is enabled)")
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()

    version = AICOS_VERSION
    print("=" * 62)
    print(f" AICOS PREFLIGHT — kit version {version}")
    print("=" * 62)
    rep = Report()
    check_python(rep)
    deps = check_dependencies(rep)
    shared_enabled = os.environ.get("AICOS_ALLOW_SHARED_OPERATOR_KEYS", "false").lower() in {"1", "true", "yes"}
    check_secrets(rep, args.key, args.operator_key, shared_enabled, os.environ.get("AICOS_VIEWER_KEY", ""))
    check_outbox(rep)
    cfg = check_config(rep, args.config)
    check_sources(rep, cfg, deps)
    check_backend(rep, args.backend, args.key)
    weights = check_models(rep, cfg, deps)
    if not args.skip_benchmark:
        check_capacity(rep, cfg, deps, weights)
    check_disk(rep, os.path.dirname(os.path.abspath(args.config)) or ".")

    print("=" * 62)
    if rep.failures:
        print(f" {len(rep.failures)} FAILURE(S) — fix these before running the worker:")
        for _, name, detail, _ in rep.failures:
            print(f"   - {name}: {detail}")
        print("=" * 62)
        return 1
    if rep.warnings:
        print(f" READY, with {len(rep.warnings)} warning(s) — the worker will run.")
    else:
        print(" ALL CHECKS PASSED — ready to run the worker.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
