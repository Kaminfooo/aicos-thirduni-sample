#!/usr/bin/env python3
"""
AICOS fight diagnostic
----------------------
Shows, for every analyzed frame, exactly what the fight detector measures
on each pair of people and which condition is failing.

The detector needs a pair to be close together AND either

  grappling  both bodies moving fast without travelling anywhere, or
  striking   both sets of hands moving fast while the bodies stay put

Two people trading blows on the spot barely move their bounding boxes, so
"striking" exists for exactly that case. It depends on wrist keypoints,
which get unreliable when people are small in frame — this tool tells you
whether your camera is producing them and at what speed.

Usage:
    python fight_debug.py "http://192.168.1.12:8080/video" --rotate 90
    python fight_debug.py clip.mp4 --imgsz 640 --duration 60
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

import cv2

from detectors import YoloBehaviorSuite


class LatestFrame:
    """Newest frame only — a live camera does not wait for inference."""

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.stop = threading.Event()
        self.dropped = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.stop.set()
                break
            with self._lock:
                if self.frame is not None:
                    self.dropped += 1
                self.frame = frame

    def get(self):
        with self._lock:
            frame, self.frame = self.frame, None
            return frame

    def close(self):
        self.stop.set()
        self._thread.join(timeout=2)


def rotate_frame(frame, rot: int):
    rot = int(rot) % 360
    code = {90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rot)
    return frame if code is None else cv2.rotate(frame, code)


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose fight detection")
    ap.add_argument("video")
    ap.add_argument("--rotate", type=int, default=0,
                    help="90 if the camera is mounted sideways")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="wrist keypoints need resolution; 320 often loses them")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--model", default="yolov8s-pose.pt")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--duration", type=float, default=60)
    args = ap.parse_args()

    is_file = os.path.isfile(args.video)
    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open: " + args.video)

    print(f"model {args.model}  imgsz {args.imgsz}  conf {args.conf}  "
          f"rotate {args.rotate}")
    print("Have TWO people stand close and trade blows for 15 s.")
    print("-" * 76)

    suite = YoloBehaviorSuite(camera={}, config={
        "model": args.model, "conf": args.conf, "imgsz": args.imgsz,
        "crowd": {"enabled": False}, "person_down": {"enabled": False},
        "panic": {"enabled": False}, "intrusion": {"enabled": False},
        "loitering": {"enabled": False},
        "fight": {"enabled": True, "seconds": args.seconds},
    })
    cfg = suite.cfg["fight"]
    if "min_limb_speed" not in cfg:
        raise SystemExit(
            "This detectors.py predates limb tracking, so striking cannot be\n"
            "detected at all. Install the current edge/detectors.py and retry.")
    min_limb = float(cfg["min_limb_speed"])
    min_speed = float(cfg.get("min_speed", 0.5))
    max_travel = float(cfg.get("max_travel", 0.8))

    latest = None if is_file else LatestFrame(cap)
    started = time.time()
    n = events = 0
    peak_limb = 0.0
    with_kp = 0
    pairs_seen = 0
    try:
        while True:
            if is_file:
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                if latest.stop.is_set():
                    break
                if args.duration and (time.time() - started) > args.duration:
                    print(f"\n(reached --duration {args.duration:.0f}s)")
                    break
                frame = latest.get()
                if frame is None:
                    time.sleep(0.02)
                    continue
            frame = rotate_frame(frame, args.rotate)
            n += 1
            vt = time.time() - started
            h, w = frame.shape[:2]
            if w > args.imgsz:
                frame = cv2.resize(frame, (args.imgsz, int(h * args.imgsz / w)))
            found = suite.detect(frame, None, now=1_000_000.0 + vt)

            live = [t for t in suite.tracker.tracks.values() if t.missed == 0]
            for tr in live:
                if tr.kp_hist:
                    with_kp += 1
                peak_limb = max(peak_limb, tr.limb_speed())

            if len(live) >= 2:
                pairs_seen += 1
                a, b = live[0], live[1]
                (ax, ay), (bx, by) = a.center(), b.center()
                avg_h = max(1.0, (a.height() + b.height()) / 2.0)
                gap = math.hypot(ax - bx, ay - by) / avg_h
                la, lb = a.limb_speed(), b.limb_speed()
                close = "CLOSE" if gap < float(cfg.get("proximity", 1.0)) else "apart"
                strike = "STRIKE" if (la > min_limb and lb > min_limb
                                      and a.net_drift() < max_travel
                                      and b.net_drift() < max_travel) else "-"
                max_ratio = float(cfg.get("max_path_ratio", 0.45))
                grap = "GRAPPLE" if (a.speed_norm() > min_speed
                                     and b.speed_norm() > min_speed
                                     and a.path_ratio() < max_ratio
                                     and b.path_ratio() < max_ratio) else "-"
                print(f"t={vt:5.1f}s gap={gap:4.2f} {close:5}  "
                      f"hands {la:4.2f}/{lb:4.2f} (need >{min_limb})  "
                      f"drift {a.net_drift():4.2f}/{b.net_drift():4.2f}  "
                      f"{grap} {strike}")
            elif n % 10 == 0:
                print(f"t={vt:5.1f}s  people={len(live)} — need two in frame")
            for ev in found:
                events += 1
                print("   " + "=" * 60)
                print(f"   >>> EVENT: {ev['type']} (conf {ev['confidence']})")
                print("   " + "=" * 60)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n(stopped)")
    finally:
        if latest is not None:
            latest.close()
        cap.release()

    print("-" * 76)
    print(f"frames analyzed: {n}   frames with two people: {pairs_seen}")
    print(f"wrist keypoints seen on {with_kp} tracked people")
    print(f"fastest hand speed measured: {peak_limb:.2f}  "
          f"(threshold is {min_limb})")
    print(f"events fired: {events}")
    if events:
        print("VERDICT: fight detection works at these settings.")
    elif with_kp == 0:
        print("VERDICT: no wrist keypoints at all — the people are too small")
        print("         or too blurred for the pose model. Raise --imgsz to 640")
        print("         or 960, or move the camera closer.")
    elif peak_limb < min_limb:
        suggest = max(0.3, round(peak_limb * 0.6, 1))
        print(f"VERDICT: hands were tracked but never reached {min_limb}.")
        print(f"         Peak was {peak_limb:.2f}. Set fight.min_limb_speed to")
        print(f"         about {suggest} in cameras.yaml and test again — but")
        print(f"         check afterwards that ordinary gesturing stays silent.")
    elif pairs_seen == 0:
        print("VERDICT: never saw two people at once — both must be in frame.")
    else:
        print("VERDICT: the signals were reached but not held long enough.")
        print("         Keep it up for the full --seconds window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
