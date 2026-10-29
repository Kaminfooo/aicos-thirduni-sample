#!/usr/bin/env python3
"""
AICOS violence model probe
--------------------------
Shows what the TRAINED fight model scores, frame by frame, on your camera.

The geometric fight detector can be diagnosed with fight_debug.py. This is
the equivalent for the trained model: it answers whether the model
recognises what your people are doing as violence, and at what confidence,
so the threshold comes from measurement rather than guesswork.

Usage:
    python violence_debug.py "http://192.168.1.12:8080/video" --rotate 90
    python violence_debug.py clip.mp4 --conf 0.2
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2


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
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(int(rot) % 360)
    return frame if code is None else cv2.rotate(frame, code)


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe the trained fight model")
    ap.add_argument("video")
    ap.add_argument("--repo", default="Musawer14/fight_detection_yolov8")
    ap.add_argument("--weights", default="", help="local .pt, overrides --repo")
    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.10,
                    help="deliberately LOW so you can see scores that the "
                         "production threshold would discard")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--duration", type=float, default=60)
    args = ap.parse_args()

    from violence import YoloViolenceModel, ViolenceDetector
    det = ViolenceDetector(config={"repo": args.repo, "weights": args.weights,
                                   "conf": args.conf, "imgsz": args.imgsz})
    path = det._resolve_weights()
    model = YoloViolenceModel(path, args.conf, args.imgsz)
    names = model.names
    print(f"model classes: {names}")
    violent = [i for i, n in names.items()
               if "viol" in n.lower() and not n.lower().startswith("non")]
    if not violent:
        violent = [max(names)] if names else [1]
    cls = violent[0]
    print(f"scoring class {cls} = '{names.get(cls)}'")
    print("Have two people act out a fight. Scores print every frame.")
    print("-" * 70)

    is_file = os.path.isfile(args.video)
    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open: " + args.video)
    latest = None if is_file else LatestFrame(cap)

    started = time.time()
    n = hits = 0
    peak = 0.0
    scores = []
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
            score = model.best_score(frame, cls)
            peak = max(peak, score)
            if score > 0:
                hits += 1
                scores.append(score)
            bar = "#" * int(score * 40)
            print(f"t={time.time()-started:5.1f}s  violence={score:4.2f} {bar}")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n(stopped)")
    finally:
        if latest is not None:
            latest.close()
        cap.release()

    print("-" * 70)
    print(f"frames scored: {n}   frames the model called violent: {hits}")
    print(f"highest score: {peak:.2f}")
    if scores:
        scores.sort()
        print(f"median score when it fired: {scores[len(scores)//2]:.2f}")
    if peak == 0.0:
        print("VERDICT: the model never saw violence in this footage.")
        print("         Staged fighting often does not resemble the real")
        print("         footage these models are trained on. Either accept")
        print("         that fight detection needs real incidents to validate,")
        print("         or fine-tune on your own clips.")
    elif peak < 0.45:
        print(f"VERDICT: it responded, peaking at {peak:.2f}, below the 0.45")
        print(f"         production threshold. Setting violence.conf to about")
        print(f"         {max(0.2, round(peak*0.7, 2))} would fire — but then run")
        print(f"         the CONTROL test (two people talking) and confirm it")
        print(f"         stays at zero. If it does not, leave violence off.")
    else:
        print("VERDICT: the model recognises this as violence at production")
        print("         settings. Enable it, then run the control test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
