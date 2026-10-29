"""
AICOS camera calibration probe
------------------------------
Runs the detection model over a video file or stream and reports what it
actually sees: person counts, person sizes, detection consistency. Use it
on every new camera or clip BEFORE wiring detectors — it tells you which
imgsz / conf the camera needs and which behaviors can realistically fire.

Usage:
    python probe.py /path/to/clip.mp4
    python probe.py /path/to/clip.mp4 --imgsz 960 --conf 0.25
    python probe.py "rtsp://user:pass@ip:554/..." --seconds 30
"""
import argparse
import time

import cv2

from detectors import UltralyticsEngine


def _verdict(heights: list, frames_with: int, sampled: int, max_p: int,
             imgsz: int, conf: float) -> None:
    print("-" * 50)
    print(f"settings: imgsz={imgsz} conf={conf}")
    print(f"sampled {sampled} frames | frames with people: {frames_with} | "
          f"max simultaneous: {max_p}")
    if heights:
        heights.sort()
        med = heights[len(heights) // 2]
        print(f"tallest-person fraction: median {med:.2f}  max {heights[-1]:.2f}")
        if med < 0.10:
            print("VERDICT: people are SMALL -> use imgsz 960 and conf 0.25 for this camera")
        elif med < 0.18:
            print("VERDICT: workable; pose-based person_down will be marginal at this distance")
        else:
            print("VERDICT: good size for all behaviors including person_down")
    else:
        print("VERDICT: NO PEOPLE DETECTED -> wrong scene, or raise imgsz / lower conf")


def probe_snapshot(url: str, args) -> None:
    """Poll a refreshing-JPEG camera and report what the model sees."""
    import numpy as np
    import requests

    eng = UltralyticsEngine("yolov8n-pose.pt", conf=args.conf, imgsz=args.imgsz)
    session = requests.Session()
    polls = args.polls
    interval = args.interval
    print(f"snapshot camera: polling {polls}x every {interval}s "
          f"(~{int(polls * interval)}s total)")

    sampled = frames_with = max_p = fails = 0
    heights: list = []
    for i in range(polls):
        if i:
            time.sleep(interval)
        frame = None
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and resp.content:
                frame = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
            else:
                print(f"poll {i+1}: HTTP {resp.status_code}")
        except requests.RequestException as exc:
            print(f"poll {i+1}: fetch failed - {exc}")
        if frame is None:
            fails += 1
            continue
        sampled += 1
        h, w = frame.shape[:2]
        if w > args.imgsz:
            frame = cv2.resize(frame, (args.imgsz, int(h * args.imgsz / w)))
            h, w = frame.shape[:2]
        dets = eng.infer(frame)
        p = len(dets)
        max_p = max(max_p, p)
        if p:
            frames_with += 1
            tallest = max((d["box"][3] - d["box"][1]) / h for d in dets)
            heights.append(tallest)
        print(f"poll {i+1}: persons={p}" +
              (f"  tallest={heights[-1]:.2f} of frame" if p else ""))
    if fails:
        print(f"({fails} poll(s) returned no usable image)")
    if sampled == 0:
        print("-" * 50)
        print("VERDICT: camera unreachable or not an image URL - check the URL in a browser")
        return
    _verdict(heights, frames_with, sampled, max_p, args.imgsz, args.conf)
    print("NOTE: snapshot cameras refresh slowly. Crowd, person-down, intrusion")
    print("      and loitering work; fight and panic need a continuous stream.")


def main() -> None:
    ap = argparse.ArgumentParser(description="AICOS calibration probe")
    ap.add_argument("source", help="video file path, rtsp:// URL, or webcam index")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--every", type=int, default=15, help="sample 1 of N frames")
    ap.add_argument("--seconds", type=float, default=0,
                    help="stop after N seconds (0 = full file / until Ctrl+C)")
    ap.add_argument("--polls", type=int, default=10,
                    help="snapshot cameras: how many images to fetch")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="snapshot cameras: seconds between fetches")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    # Snapshot cameras publish a single refreshing JPEG — poll them instead
    # of trying to read a continuous stream.
    if isinstance(src, str) and src.lower().startswith(("http://", "https://")) \
            and src.split("?")[0].lower().endswith((".jpg", ".jpeg", ".png")):
        probe_snapshot(src, args)
        return

    if isinstance(src, str) and ("youtube.com/" in src or "youtu.be/" in src):
        import shutil
        import subprocess
        if shutil.which("yt-dlp") is None:
            raise SystemExit("yt-dlp not installed - run: pip install yt-dlp")
        out = subprocess.run(["yt-dlp", "-g", "-f", "best[height<=720]/best", src],
                             capture_output=True, text=True, timeout=60)
        urls = [l.strip() for l in (out.stdout or "").splitlines() if l.startswith("http")]
        if not urls:
            raise SystemExit("yt-dlp could not resolve: " + (out.stderr or "")[:200])
        src = urls[0]
        print("resolved live stream URL")
    eng = UltralyticsEngine("yolov8n-pose.pt", conf=args.conf, imgsz=args.imgsz)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit("cannot open source: " + str(src))

    t0 = time.time()
    n = sampled = frames_with = max_p = 0
    heights: list = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.seconds and time.time() - t0 > args.seconds:
                break
            n += 1
            if n % args.every:
                continue
            sampled += 1
            h, w = frame.shape[:2]
            if w > args.imgsz:
                frame = cv2.resize(frame, (args.imgsz, int(h * args.imgsz / w)))
                h, w = frame.shape[:2]
            dets = eng.infer(frame)
            p = len(dets)
            max_p = max(max_p, p)
            if p:
                frames_with += 1
                tallest = max((d["box"][3] - d["box"][1]) / h for d in dets)
                heights.append(tallest)
                print(f"frame {n}: persons={p}  tallest={tallest:.2f} of frame")
    finally:
        cap.release()

    _verdict(heights, frames_with, sampled, max_p, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
