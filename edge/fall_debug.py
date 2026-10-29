"""
AICOS person-down diagnostic
----------------------------
Answers the question "why didn't it fire?" by showing, for every analyzed
frame, exactly what the detector saw: how many people, each person's
box shape and speed, and whether the down-timer is running.

It clocks the analysis on VIDEO time rather than wall time, so results do
not depend on how fast this machine happens to run inference.

Usage:
    python fall_debug.py /root/videos/fall.mp4
    python fall_debug.py /root/videos/fall.mp4 --every 2 --seconds 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2

from detectors import YoloBehaviorSuite

BASE = 1_000_000.0     # arbitrary clock origin


class LatestFrame:
    """Keeps only the newest frame from a live camera.

    A live stream does not wait. Reading frames in order means falling
    further behind with every inference, until the analysis is minutes
    stale — the fall has already happened and been missed. This thread
    discards everything except the current frame, so analysis is of NOW.
    """

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose person-down detection")
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=3,
                    help="analyze 1 of every N frames")
    ap.add_argument("--seconds", type=float, default=1.0,
                    help="down-duration threshold being tested")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--duration", type=float, default=0,
                    help="stop automatically after N seconds on a live "
                         "camera, so you do not have to time Ctrl+C")
    ap.add_argument("--model", default="yolov8n-pose.pt",
                    help="yolov8n-pose.pt (fastest), yolov8s-pose.pt or "
                         "yolov8m-pose.pt (better at unusual poses like a "
                         "body on the ground)")
    args = ap.parse_args()

    is_file = os.path.isfile(args.video)
    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open: " + args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if is_file:
        print(f"file: {total} frames @ {fps:.1f} fps = {total/fps:.1f}s | "
              f"analyzing 1 in {args.every} -> {fps/args.every:.1f} samples/sec")
    else:
        print("LIVE camera: timing from the wall clock, always analyzing the "
              "newest frame.")
        print("Walk in, fall SIDEWAYS across the view, stay still 10 s, "
              "then press Ctrl+C.")
    print(f"testing person_down.seconds = {args.seconds}")
    print("-" * 72)

    print(f"model: {args.model}  conf: {args.conf}  imgsz: {args.imgsz}")
    suite = YoloBehaviorSuite(camera={}, config={
        "model": args.model,
        "conf": args.conf, "imgsz": args.imgsz,
        "crowd": {"enabled": False}, "fight": {"enabled": False},
        "panic": {"enabled": False}, "intrusion": {"enabled": False},
        "loitering": {"enabled": False},
        "person_down": {"enabled": True, "seconds": args.seconds},
    })

    n = 0
    events = 0
    ever_down = False
    max_ratio = 0.0
    started = time.time()
    latest = None if is_file else LatestFrame(cap)
    try:
      while True:
          if is_file:
              ok, frame = cap.read()
              if not ok:
                  break
              n += 1
              if n % args.every:
                  continue
              vt = n / fps
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
              n += 1
              vt = time.time() - started      # real elapsed time, not frame count
          h, w = frame.shape[:2]
          if w > args.imgsz:
              frame = cv2.resize(frame, (args.imgsz, int(h * args.imgsz / w)))
          found = suite.detect(frame, None, now=BASE + vt)

          live = [t for t in suite.tracker.tracks.values() if t.missed == 0]
          parts = []
          for tr in live:
              x1, y1, x2, y2 = tr.box()
              ratio = (x2 - x1) / max(1.0, (y2 - y1))
              max_ratio = max(max_ratio, ratio)
              posture = "PRONE" if ratio > 1.35 else "upright"
              parts.append(f"#{tr.tid} w/h={ratio:.2f} drift={tr.net_drift():.2f} "
                           f"{posture}")

          # The down-timer is anchored to a ground POSITION, not a track id, so
          # read it from the suite rather than from any single track.
          sites = getattr(suite, "_down_sites", {})
          held = 0.0
          for site in sites.values():
              held = max(held, site["last"] - site["first"])
          if sites:
              ever_down = True
          timer = f"   [down-timer {held:.1f}s / {args.seconds:.0f}s]" if sites else ""
          print(f"t={vt:5.2f}s  people={len(live)}  " + " | ".join(parts) + timer)
          for ev in found:
              events += 1
              print("          " + "=" * 58)
              print(f"          >>> EVENT FIRED: {ev['type']}")
              print(f"          >>> severity {ev['severity']}, "
                    f"confidence {ev['confidence']} — DETECTION WORKS")
              print("          " + "=" * 58)
          sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n(stopped)")
    if latest is not None:
        print(f"\n(analyzed {n} frames in {time.time()-started:.0f}s — "
              f"{latest.dropped} stale frames dropped to stay current)")
        latest.close()
    cap.release()
    print("-" * 72)
    print(f"events fired: {events}")
    print(f"widest person box seen: w/h = {max_ratio:.2f}  "
          f"(a lying person usually exceeds 1.35)")
    if events:
        print("VERDICT: detection works on this clip at these settings.")
    elif ever_down:
        print("VERDICT: the down-posture WAS recognized but not held long enough.")
        print("         Use a longer clip, or lower person_down.seconds for testing.")
    else:
        print("VERDICT: the down-posture was never recognized.")
        print("         Either she is never fully horizontal in frame (a forward")
        print("         collapse toward the camera stays 'tall'), or the pose")
        print("         keypoints are unreliable at this angle. Try a clip filmed")
        print("         side-on, where a lying body is clearly horizontal.")


if __name__ == "__main__":
    main()
