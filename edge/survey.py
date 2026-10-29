#!/usr/bin/env python3
"""
AICOS Camera Survey
-------------------
Batch-tests a list of camera sources and reports, for each one, whether it
works and WHICH DETECTORS it can realistically support.

This is the tool for Phase 1 of a pilot: a city hands you fifty camera
URLs, and instead of opening them one by one you get a ranked table plus
a ready-to-run cameras.yaml. It is also how you triage public test
cameras without wasting an afternoon.

The recommendations come from measurements, not guesses:

  person size    how tall people appear as a fraction of the frame. Pose
                 keypoints get unreliable below ~0.18, so person-down is
                 only offered above that.
  continuity     snapshot cameras deliver an image every few seconds, so
                 fight and panic — which need frame-to-frame motion —
                 are never offered for them.
  population     a camera that never sees eight people cannot support the
                 default crowd threshold, so a lower one is suggested.

Usage:
    # one URL per line, optionally "name,url"
    python survey.py cameras.txt
    python survey.py cameras.txt --samples 8 --out ../config/cameras-survey.yaml
    python survey.py --url http://cam.example/snap.jpg --url /root/videos/street1.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from urllib.parse import urlsplit, urlunsplit

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")

# person size thresholds, from what the detectors actually need
SIZE_POSE = 0.18      # person-down needs reliable pose keypoints
SIZE_MOTION = 0.12    # fight / snatch need believable motion geometry
SIZE_MIN = 0.06       # below this, nothing is trustworthy
LOW_RES_WIDTH = 500   # public agency snapshots are often ~350px — too coarse


class Result:
    def __init__(self, name, url, kind):
        self.name = name
        self.url = url
        self.kind = kind
        self.ok = False
        self.error = ""
        self.width = 0
        self.height = 0
        self.samples = 0
        self.with_people = 0
        self.max_people = 0
        self.sizes = []
        self.vehicles = 0
        self.brightness = []
        self.sharpness = []
        self.native_fps = 0.0

    @property
    def median_size(self) -> float:
        if not self.sizes:
            return 0.0
        s = sorted(self.sizes)
        return s[len(s) // 2]

    @property
    def hit_rate(self) -> float:
        return (self.with_people / self.samples) if self.samples else 0.0

    @property
    def continuous(self) -> bool:
        return self.kind in ("file", "stream", "device")

    @property
    def median_brightness(self) -> float:
        if not self.brightness:
            return 0.0
        a = sorted(self.brightness)
        return float(a[len(a) // 2])

    @property
    def median_sharpness(self) -> float:
        if not self.sharpness:
            return 0.0
        a = sorted(self.sharpness)
        return float(a[len(a) // 2])


def classify(url: str) -> str:
    if isinstance(url, str) and url.isdigit():
        return "device"
    low = url.lower()
    if low.startswith(("rtsp://", "rtsps://", "rtmp://")):
        return "stream"
    if low.startswith(("http://", "https://")):
        base = low.split("?")[0]
        if base.endswith(IMAGE_EXTS):
            return "snapshot"
        return "stream"
    return "file" if os.path.isfile(url) else "missing"


# ---------------------------------------------------------------------------
# measurement (pure — takes frames, returns nothing but numbers)
# ---------------------------------------------------------------------------


def analyze_frame(frame, engine, result: Result) -> None:
    """Run the model on one frame and fold the numbers into result."""
    h, w = frame.shape[:2]
    result.width = result.width or w
    result.height = result.height or h
    try:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result.brightness.append(float(gray.mean()))
        result.sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    except Exception:
        pass
    dets = engine.infer(frame)
    people = [d for d in dets if d.get("cls", 0) == 0]
    result.vehicles += sum(1 for d in dets if d.get("cls", 0) in (2, 3, 5, 7))
    result.samples += 1
    if not people:
        return
    result.with_people += 1
    result.max_people = max(result.max_people, len(people))
    tallest = max((d["box"][3] - d["box"][1]) / max(1.0, h) for d in people)
    result.sizes.append(tallest)


def recommend(r: Result) -> tuple:
    """Return (verdict, [detector names], notes) for a surveyed camera."""
    if not r.ok:
        return "UNUSABLE", [], r.error or "could not read from this source"

    notes_pre = []
    if r.width and r.width < LOW_RES_WIDTH:
        notes_pre.append(f"only {r.width}x{r.height} — most public agency "
                         f"cameras are published at this size and are too "
                         f"coarse for reliable analytics")

    if r.samples and not r.with_people:
        # a camera can be useless for people and still useful for traffic
        if r.vehicles:
            notes_pre.append(f"saw {r.vehicles} vehicle detection(s) across "
                             f"{r.samples} samples")
            return "VEHICLES ONLY", ["vehicles"], "; ".join(notes_pre)
        notes_pre.append("nobody appeared during the survey — try a busier "
                         "time of day")
        return "NO PEOPLE", [], "; ".join(notes_pre)

    size = r.median_size
    if size < SIZE_MIN:
        return "TOO FAR", [], (f"people are {size:.2f} of frame — too small to "
                               f"analyze; try imgsz 960 and conf 0.25, or a "
                               f"closer camera")

    dets = []
    notes = list(notes_pre)
    dets.append("crowd")
    if r.max_people < 8:
        notes.append(f"only ever saw {r.max_people} people — lower crowd.warn "
                     f"to about {max(3, r.max_people)}")
    if size >= SIZE_POSE:
        dets.append("person_down")
    else:
        notes.append(f"people are {size:.2f} of frame — person-down needs "
                     f"{SIZE_POSE:.2f}+ for reliable pose")
    if r.continuous and size >= SIZE_MOTION:
        dets += ["fight", "panic"]
    elif not r.continuous:
        notes.append("snapshot camera — fight and panic need a continuous "
                     "stream and are not offered")
    dets += ["intrusion", "loitering"]
    if r.vehicles:
        dets.append("vehicles")

    verdict = "GOOD" if size >= SIZE_POSE else "WORKABLE"
    return verdict, dets, "; ".join(notes)


# ---------------------------------------------------------------------------
# acquisition
# ---------------------------------------------------------------------------


def survey_snapshot(r: Result, engine, samples: int, interval: float,
                    imgsz: int) -> None:
    import cv2
    import numpy as np
    import requests
    session = requests.Session()
    for i in range(samples):
        if i:
            time.sleep(interval)
        try:
            resp = session.get(r.url, timeout=12,
                               headers={"User-Agent": "AICOS-survey"})
        except Exception as exc:
            r.error = f"fetch failed: {str(exc)[:40]}"
            continue
        if resp.status_code != 200 or not resp.content:
            r.error = f"HTTP {resp.status_code}"
            continue
        frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            r.error = "response was not a decodable image"
            continue
        r.ok = True
        r.error = ""
        h, w = frame.shape[:2]
        r.width, r.height = w, h
        if w > imgsz:
            frame = cv2.resize(frame, (imgsz, int(h * imgsz / w)))
        analyze_frame(frame, engine, r)


def survey_stream(r: Result, engine, samples: int, every: int,
                  imgsz: int) -> None:
    import cv2
    src = int(r.url) if r.url.isdigit() else r.url
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        r.error = "cannot open"
        cap.release()
        return
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if 0.5 <= fps <= 240.0:
            r.native_fps = fps
    except Exception:
        r.native_fps = 0.0
    n = 0
    deadline = time.time() + 90
    while r.samples < samples and time.time() < deadline:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        n += 1
        if n % every:
            continue
        r.ok = True
        h, w = frame.shape[:2]
        r.width, r.height = w, h
        if w > imgsz:
            frame = cv2.resize(frame, (imgsz, int(h * imgsz / w)))
        analyze_frame(frame, engine, r)
    cap.release()
    if not r.ok and not r.error:
        r.error = "opened but delivered no frames"


# ---------------------------------------------------------------------------


def load_sources(args) -> list:
    entries = []
    for url in (args.url or []):
        entries.append((None, url))
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "," in line and not line.split(",")[0].lower().startswith(
                        ("http", "rtsp", "/")):
                    name, url = line.split(",", 1)
                    entries.append((name.strip(), url.strip()))
                else:
                    entries.append((None, line))
    out = []
    for i, (name, url) in enumerate(entries, 1):
        out.append(Result(name or f"CAM-{i:03d}", url, classify(url)))
    return out


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> int:
    return int(round(max(lo, min(hi, v))))


def assurance_score(r: Result) -> int:
    if not r.ok:
        return 0
    resolution = min(1.0, (r.width * r.height) / float(1280 * 720)) if r.width and r.height else 0.0
    brightness = r.median_brightness
    light = max(0.0, 1.0 - abs(brightness - 125.0) / 125.0) if brightness else 0.0
    sharp = min(1.0, r.median_sharpness / 180.0) if r.median_sharpness else 0.0
    person = min(1.0, r.median_size / SIZE_POSE) if r.median_size else (0.75 if r.vehicles else 0.25)
    continuity = 1.0 if r.continuous else 0.65
    return _clamp(100.0 * (0.22 * resolution + 0.13 * light + 0.15 * sharp + 0.30 * person + 0.20 * continuity))


def detector_scores(r: Result) -> dict:
    verdict, dets, _ = recommend(r)
    dets = set(dets)
    base = assurance_score(r)
    aliases = {"fight": "violence", "vehicles": "traffic"}
    out = {}
    for det in sorted(dets):
        name = aliases.get(det, det)
        score = base
        reasons = []
        if name in {"violence", "panic", "snatch"}:
            if r.native_fps:
                fps_component = min(100, int(r.native_fps / 10.0 * 100))
                score = int(round(0.65 * base + 0.35 * fps_component))
                if r.native_fps < 10:
                    reasons.append(f"native FPS {r.native_fps:.1f} is below the 10+ FPS motion target")
            elif not r.continuous:
                score = min(score, 45)
                reasons.append("source is not continuous")
        if name == "person_down" and r.median_size < SIZE_POSE:
            score = min(score, 60)
            reasons.append("person size is below the preferred pose threshold")
        status = "certified" if score >= 70 else ("conditional" if score >= 50 else "unsupported")
        out[name] = {"score": int(score), "status": status, "reasons": reasons}
    return out


def _redact_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        if not parts.scheme or not parts.netloc:
            return Path(value).name if os.path.isfile(value) else value
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        return urlunsplit((parts.scheme, host + port, parts.path, "", ""))
    except Exception:
        return "[redacted]"


def emit_json(results, path: str) -> None:
    cameras = []
    for r in results:
        verdict, dets, notes = recommend(r)
        cameras.append({
            "camera_id": r.name,
            "source_type": r.kind,
            "source_redacted": _redact_url(r.url),
            "source_fingerprint": hashlib.sha256(str(r.url).encode("utf-8")).hexdigest(),
            "verdict": verdict,
            "assurance_score": assurance_score(r),
            "recommended_detectors": dets,
            "detector_assessments": detector_scores(r),
            "metrics": {
                "width": r.width, "height": r.height, "samples": r.samples,
                "person_hit_rate": round(r.hit_rate, 4), "max_people": r.max_people,
                "median_person_size": round(r.median_size, 4), "vehicle_detections": r.vehicles,
                "median_brightness": round(r.median_brightness, 2),
                "median_sharpness": round(r.median_sharpness, 2),
                "native_fps": round(r.native_fps, 2), "continuous": r.continuous,
            },
            "notes": notes,
        })
    doc = {"schema": "aicos.camera-survey.v1", "generated_at": time.time(), "cameras": cameras}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote signed-certificate input survey to {path}")


def emit_yaml(results, path: str, imgsz: int, conf: float) -> None:
    usable = [(r, recommend(r)) for r in results]
    usable = [(r, v, d, n) for r, (v, d, n) in usable if d]
    if not usable:
        print("\nno usable cameras — nothing written")
        return
    lines = [
        "settings:",
        "  process_every_n_frames: 20",
        "  event_cooldown_seconds: 30",
        "  resize_width: 640",
        "  snapshot_interval_seconds: 2",
        "  event_snapshots: true",
        "  detectors:",
        "    motion: { enabled: false }",
        "    yolo:",
        "      enabled: true",
        '      model: "yolov8n-pose.pt"',
        f"      conf: {conf}",
        f"      imgsz: {imgsz}",
        "",
        "cameras:",
    ]
    for r, verdict, dets, notes in usable:
        lines.append(f'  - id: "{r.name}"')
        lines.append(f'    name: "{r.name}"')
        lines.append(f'    rtsp_url: "{r.url}"')
        lines.append('    district: "[SET ME]"')
        lines.append("    lat: 0.0")
        lines.append("    lon: 0.0")
        lines.append("    detectors:")
        lines.append("      yolo:")
        for name in ("person_down", "fight", "panic", "crowd",
                     "intrusion", "loitering", "vehicles"):
            on = "true" if name in dets else "false"
            lines.append(f"        {name}: {{ enabled: {on} }}")
        if r.max_people and r.max_people < 8 and "crowd" in dets:
            lines.append(f"        crowd: {{ enabled: true, warn: "
                         f"{max(3, r.max_people)}, critical: "
                         f"{max(5, r.max_people + 2)} }}")
        if notes:
            lines.append(f"    # survey: {notes}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {len(usable)} camera(s) to {path}")
    print("Set district / lat / lon before using it.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Survey camera sources")
    ap.add_argument("file", nargs="?", help="text file, one source per line")
    ap.add_argument("--url", action="append", help="a source (repeatable)")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--interval", type=float, default=3.0,
                    help="seconds between snapshot fetches")
    ap.add_argument("--every", type=int, default=15,
                    help="for streams: analyze 1 of every N frames")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", help="write a cameras.yaml for the usable ones")
    ap.add_argument("--json-out", help="write a credential-redacted camera assurance survey JSON")
    args = ap.parse_args()

    results = load_sources(args)
    if not results:
        ap.error("give a file or at least one --url")

    from detectors import UltralyticsEngine
    engine = UltralyticsEngine("yolov8n-pose.pt", args.conf, args.imgsz,
                               classes=[0, 2, 3, 5, 7])

    print(f"surveying {len(results)} source(s), {args.samples} samples each")
    print("=" * 78)
    for r in results:
        print(f"\n{r.name}  [{r.kind}]  {r.url[:60]}")
        sys.stdout.flush()
        try:
            if r.kind == "missing":
                r.error = "file not found"
            elif r.kind == "snapshot":
                survey_snapshot(r, engine, args.samples, args.interval, args.imgsz)
            else:
                survey_stream(r, engine, args.samples, args.every, args.imgsz)
        except Exception as exc:
            r.error = f"{type(exc).__name__}: {str(exc)[:50]}"
        verdict, dets, notes = recommend(r)
        if r.ok:
            print(f"  {r.width}x{r.height} | samples {r.samples} | "
                  f"people in {r.hit_rate*100:.0f}% | max {r.max_people} | "
                  f"size {r.median_size:.2f} | vehicles {r.vehicles} | "
                  f"fps {r.native_fps:.1f} | assurance {assurance_score(r)}/100")
        print(f"  -> {verdict}" + (f": {', '.join(dets)}" if dets else ""))
        if notes:
            print(f"     note: {notes}")

    print("\n" + "=" * 78)
    print(f"{'camera':<14}{'verdict':<11}{'size':>6}{'max':>5}  detectors")
    print("-" * 78)
    for r in sorted(results, key=lambda x: -x.median_size):
        verdict, dets, _ = recommend(r)
        print(f"{r.name:<14}{verdict:<11}{r.median_size:>6.2f}"
              f"{r.max_people:>5}  {', '.join(dets) if dets else '-'}")
    if args.out:
        emit_yaml(results, args.out, args.imgsz, args.conf)
    if args.json_out:
        emit_json(results, args.json_out)
    return 0 if any(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
