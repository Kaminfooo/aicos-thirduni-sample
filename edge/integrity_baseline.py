#!/usr/bin/env python3
"""Deliberately establish or refresh the certified camera-view baseline.

This tool is intentionally separate from the always-on worker: a government
operator/installer must approve a known-good view instead of silently accepting
whatever a camera happens to show at first boot.
"""
from __future__ import annotations
import argparse
import os
import time
import cv2
from camera_integrity import IntegrityMonitor, quality_metrics


def main():
    ap = argparse.ArgumentParser(description="Create an AICOS camera integrity baseline")
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--source", required=True, help="RTSP/HTTP/video file or local webcam index")
    ap.add_argument("--out-dir", default=os.environ.get("AICOS_INTEGRITY_BASELINE_DIR", "aicos-integrity-baselines"))
    ap.add_argument("--warmup", type=int, default=10, help="frames to discard before capture")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("camera source could not be opened")
    frame = None
    for _ in range(max(1, args.warmup)):
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release(); raise SystemExit("camera source did not produce a frame")
    cap.release()
    q = quality_metrics(frame)
    if q["sharpness"] < 8 or (q["dark_fraction"] > .98 and q["brightness"] < 8):
        raise SystemExit(f"refusing poor baseline: {q}")
    mon = IntegrityMonitor(args.camera_id, args.out_dir, auto_create=False)
    result = mon.establish(frame, {"created_at": time.time(), "quality": q})
    print(f"baseline created: {mon.image_path}")
    print(result)

if __name__ == "__main__":
    main()
