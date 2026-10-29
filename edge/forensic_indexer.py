#!/usr/bin/env python3
"""AICOS privacy-first forensic observation indexer.

Samples an authorized camera/video source and sends NON-BIOMETRIC object
metadata to the AICOS backend for later forensic search. No face embeddings,
identity templates, or image crops are transmitted by default.

Indexed attributes:
  * person / vehicle kind
  * coarse dominant color
  * approximate motion direction
  * normalized bounding box and detector confidence

Example:
  python forensic_indexer.py --source rtsp://... --camera CAM-001 \
      --backend https://aicos.internal --key "$AICOS_API_KEY"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from detectors import UltralyticsEngine

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _dominant_color(frame, box, kind: str) -> str:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    bh = y2 - y1
    # For people use torso/clothing, deliberately avoiding a face crop.
    if kind == "person":
        y1 = y1 + int(0.22 * bh)
        y2 = y1 + max(1, int(0.38 * bh))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    # Shrink to reduce CPU and use median HSV for resilience to small logos.
    crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # Drop extreme highlights/shadows where hue is unstable.
    good = hsv[(hsv[:, 2] > 25) & (hsv[:, 2] < 245)]
    if len(good) >= 30:
        hsv = good
    H, S, V = [float(np.median(hsv[:, i])) for i in range(3)]
    if V < 50:
        return "black"
    if S < 35 and V > 195:
        return "white"
    if S < 45:
        return "gray"
    # OpenCV hue is 0..179.
    if H < 7 or H >= 172:
        return "red"
    if H < 18:
        return "orange"
    if H < 33:
        return "yellow"
    if H < 82:
        return "green"
    if H < 98:
        return "cyan"
    if H < 130:
        return "blue"
    if H < 158:
        return "purple"
    return "pink"


class DirectionTracker:
    """Very small nearest-centroid tracker used only for coarse direction."""
    def __init__(self, ttl: float = 4.0):
        self.ttl = ttl
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, kind: str, cx: float, cy: float, now: float) -> tuple[int, str]:
        for tid in [k for k, v in self.tracks.items() if now - v["ts"] > self.ttl]:
            del self.tracks[tid]
        best = None
        for tid, tr in self.tracks.items():
            if tr["kind"] != kind:
                continue
            d = math.hypot(cx - tr["cx"], cy - tr["cy"])
            if d <= 0.18 and (best is None or d < best[0]):
                best = (d, tid, tr)
        if best is None:
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {"kind": kind, "cx": cx, "cy": cy, "ts": now}
            return tid, "stationary"
        _, tid, tr = best
        dt = max(1e-3, now - tr["ts"])
        dx, dy = (cx - tr["cx"]) / dt, (cy - tr["cy"]) / dt
        speed = math.hypot(dx, dy)
        if speed < 0.012:
            direction = "stationary"
        elif abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        tr.update(cx=cx, cy=cy, ts=now)
        return tid, direction


class ObservationOutbox:
    def __init__(self, backend: str, key: str, camera_id: str, path: str):
        self.url = backend.rstrip("/") + "/api/forensics/observations"
        self.key = key
        self.camera_id = camera_id
        self.session = requests.Session()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS batches(id INTEGER PRIMARY KEY, created_at REAL, payload TEXT, attempts INTEGER DEFAULT 0, next_attempt REAL DEFAULT 0)")
        self.db.commit()

    def _send(self, payload: dict) -> tuple[bool, str]:
        try:
            r = self.session.post(self.url, json=payload, headers={"X-AICOS-KEY": self.key}, timeout=8)
            if r.status_code in (200, 201):
                return True, ""
            return False, f"HTTP {r.status_code}: {r.text[:180]}"
        except requests.RequestException as exc:
            return False, str(exc)[:240]

    def submit(self, observations: list[dict]) -> None:
        if not observations:
            return
        payload = {"source_id": self.camera_id, "observations": observations}
        ok, err = self._send(payload)
        if ok:
            return
        with self.lock:
            self.db.execute("INSERT INTO batches(created_at,payload,next_attempt) VALUES(?,?,?)",
                            (time.time(), json.dumps(payload, separators=(",", ":")), time.time() + 2))
            self.db.commit()
        print(f"warning: forensic batch queued: {err}", file=sys.stderr)

    def flush(self) -> None:
        now = time.time()
        with self.lock:
            rows = self.db.execute("SELECT id,payload,attempts FROM batches WHERE next_attempt <= ? ORDER BY id LIMIT 20", (now,)).fetchall()
        for row_id, blob, attempts in rows:
            try: payload = json.loads(blob)
            except Exception:
                with self.lock:
                    self.db.execute("DELETE FROM batches WHERE id=?", (row_id,)); self.db.commit()
                continue
            ok, err = self._send(payload)
            with self.lock:
                if ok:
                    self.db.execute("DELETE FROM batches WHERE id=?", (row_id,))
                else:
                    delay = min(300, 2 ** min(int(attempts) + 1, 8))
                    self.db.execute("UPDATE batches SET attempts=attempts+1,next_attempt=? WHERE id=?", (time.time() + delay, row_id))
                self.db.commit()

    def close(self):
        try: self.flush()
        finally:
            self.session.close(); self.db.close()


def _appearance_key(kind: str, color: str, bbox_norm: list[float]) -> str:
    # Deliberately coarse and non-biometric: object class + clothing/body color
    # + rough apparent size bucket. It is a search aid, not an identity token.
    area = max(0.0, float(bbox_norm[2]) * float(bbox_norm[3]))
    size = "small" if area < 0.025 else ("medium" if area < 0.12 else "large")
    return f"{kind}:{color}:{size}"


def run(args) -> int:
    engine = UltralyticsEngine(args.model, args.conf, args.imgsz, classes=[0, 2, 3, 5, 7])
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("cannot open source", file=sys.stderr); return 2
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (0.5 <= fps <= 240): fps = 25.0
    stride = max(1, int(round(fps / max(0.1, args.sample_fps))))
    tracker = DirectionTracker()
    outbox = ObservationOutbox(args.backend, args.key, args.camera,
                               args.outbox or f"forensics-{args.camera}.db")
    batch: list[dict] = []
    frame_no = 0
    last_flush = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None: break
            frame_no += 1
            if frame_no % stride: continue
            now = time.time()
            h, w = frame.shape[:2]
            for det in engine.infer(frame):
                cls = int(det.get("cls", 0))
                kind = "person" if cls == 0 else VEHICLE_CLASSES.get(cls)
                if not kind: continue
                x1, y1, x2, y2 = [float(v) for v in det["box"]]
                cx, cy = ((x1 + x2) / 2 / max(1, w), (y1 + y2) / 2 / max(1, h))
                tid, direction = tracker.update("person" if cls == 0 else "vehicle", cx, cy, now)
                color = _dominant_color(frame, (x1, y1, x2, y2), "person" if cls == 0 else "vehicle")
                bbox_norm = [round(x1/max(1,w),4), round(y1/max(1,h),4),
                             round((x2-x1)/max(1,w),4), round((y2-y1)/max(1,h),4)]
                object_kind = "person" if cls == 0 else "vehicle"
                batch.append({
                    "ts": now, "kind": object_kind,
                    "confidence": float(det.get("conf", 0.0)), "color": color,
                    "direction": direction, "track_id": f"{args.camera}:{tid}",
                    "appearance_key": _appearance_key(object_kind, color, bbox_norm),
                    "bbox": bbox_norm,
                    "extra": {"object_class": kind, "privacy": "non-biometric-coarse-appearance"},
                })
            if len(batch) >= args.batch_size or time.time() - last_flush >= args.flush_seconds:
                outbox.submit(batch); batch = []; outbox.flush(); last_flush = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        if batch: outbox.submit(batch)
        outbox.close(); cap.release()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="AICOS privacy-first forensic object indexer")
    ap.add_argument("--source", required=True); ap.add_argument("--camera", required=True)
    ap.add_argument("--backend", required=True); ap.add_argument("--key", default=os.environ.get("AICOS_API_KEY", ""))
    ap.add_argument("--model", default="yolov8n.pt"); ap.add_argument("--conf", type=float, default=0.35); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--sample-fps", type=float, default=1.0, help="forensic metadata samples per second")
    ap.add_argument("--batch-size", type=int, default=100); ap.add_argument("--flush-seconds", type=float, default=10.0); ap.add_argument("--outbox")
    args = ap.parse_args()
    if len(args.key) < 24: ap.error("--key/AICOS_API_KEY must be a strong device key")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
