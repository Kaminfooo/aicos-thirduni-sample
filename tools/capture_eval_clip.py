#!/usr/bin/env python3
"""Record a short labeled evaluation clip from a live AICOS camera source.

Designed for the Thirduni person-down evaluation dataset. It works with the
same kinds of OpenCV sources AICOS already accepts, including the Android
IP Webcam MJPEG endpoint such as ``http://192.168.1.12:8080/video``.

Examples:

    # Negative clip: normal walking
    python tools/capture_eval_clip.py \
      --url http://192.168.1.12:8080/video \
      --label negative --name walking --seconds 10

    # Positive clip: controlled person-down scenario
    python tools/capture_eval_clip.py \
      --url http://192.168.1.12:8080/video \
      --label positive --name person_down_slow --seconds 10

Files are written under ``data/person_down_eval/<label>/`` by default.
Do not stage a dangerous fall. Use a controlled, low-risk transition onto a
mat/bed/soft surface, or have another adult perform a safe scenario.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import time
from eval_support import sha256, sidecar


def safe_name(value: str) -> str:
    value = value.strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_.-") or "clip"


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture a labeled AICOS evaluation clip")
    ap.add_argument("--url", required=True, help="OpenCV camera source, e.g. IP Webcam /video URL")
    ap.add_argument("--label", required=True, choices=("positive", "negative"))
    ap.add_argument("--name", required=True, help="scenario name, e.g. walking or person_down_slow")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--lead-in", type=int, default=3, help="countdown before recording")
    ap.add_argument("--width", type=int, default=640, help="output width; 0 keeps source size")
    ap.add_argument("--fps", type=float, default=15.0, help="output FPS used when source FPS is unreliable")
    ap.add_argument("--dataset", type=Path, default=Path("data/person_down_eval"))
    ap.add_argument("--camera-id", default="CAM-PHONE-01")
    ap.add_argument("--read-timeout-ms", type=int, default=10000,
                    help="bounded network read timeout; allow brief Wi-Fi stalls (default 10000)")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required: pip install opencv-python") from exc

    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise SystemExit("--seconds must be finite and > 0")
    if not math.isfinite(args.fps) or args.fps <= 0 or args.width < 0 or args.lead_in < 0:
        raise SystemExit("FPS must be finite and > 0; width and lead-in must be nonnegative")
    if not 1000 <= args.read_timeout_ms <= 60000:
        raise SystemExit('--read-timeout-ms must be between 1000 and 60000')

    # Network reads must not block the recording process indefinitely.
    if args.url.lower().startswith(('http://', 'https://', 'rtsp://')):
        cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, args.read_timeout_ms])
    else:
        cap = cv2.VideoCapture(int(args.url) if args.url.isdigit() else args.url)
    if not cap.isOpened():
        print(f"Could not open camera source: {args.url}", file=sys.stderr)
        print("For IP Webcam, verify the phone and PC are on the same LAN and that /video opens in a browser.", file=sys.stderr)
        return 2

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        print("Camera opened but no frame was received.", file=sys.stderr)
        return 3

    src_h, src_w = frame.shape[:2]
    if args.width and src_w > args.width:
        out_w = args.width
        out_h = max(2, int(round(src_h * out_w / src_w)))
    else:
        out_w, out_h = src_w, src_h

    # Use even dimensions so MP4 encoding does not silently truncate a row/column.
    out_w, out_h = max(2, out_w // 2 * 2), max(2, out_h // 2 * 2)
    # H.264 availability varies across OpenCV builds; mp4v is deliberately portable.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_dir = args.dataset / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name(args.name)}_{stamp}.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        print(f"Could not create output video: {out_path}", file=sys.stderr)
        return 4

    # Drain the live stream during the countdown; sample newest frames thereafter.
    edge_dir = str(Path(__file__).resolve().parents[1] / 'edge')
    sys.path.insert(0, edge_dir)
    from rtsp_worker import LatestFrame
    grabber = LatestFrame(cap)

    print(f"Connected: {src_w}x{src_h} -> recording {out_w}x{out_h}")
    print(f"Scenario: {args.label}/{args.name}")
    for n in range(args.lead_in, 0, -1):
        if grabber.stop.is_set():
            print('Stream ended during countdown; recording will be marked invalid.', file=sys.stderr)
            break
        print(f"Recording starts in {n}...", flush=True)
        time.sleep(1)

    countdown_stream_failed = grabber.stop.is_set()
    start = time.monotonic()
    frames = 0
    next_write = start
    frame_interval = 1.0 / args.fps
    frame_times = []
    interrupted = False

    try:
        while not countdown_stream_failed and (time.monotonic() - start) < args.seconds:
            frame = grabber.get()
            if frame is None:
                if grabber.stop.is_set():
                    break
                time.sleep(0.01)
                continue
            now = time.monotonic()
            if now < next_write:
                continue
            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h))
            writer.write(frame)
            frames += 1
            frame_times.append(now - start)
            # No synthetic duplicated frames and no rapid catch-up after a stall.
            next_write = now + frame_interval
    except KeyboardInterrupt:
        interrupted = True
    finally:
        capture_end = time.monotonic()
        writer.release()
        grabber.close()
        if not grabber._thread.is_alive():
            cap.release()

    actual = capture_end - start
    gaps = [b - a for a, b in zip(frame_times, frame_times[1:])]
    probe = cv2.VideoCapture(str(out_path))
    decoded = 0
    try:
        while True:
            ok, frame = probe.read()
            if not ok or frame is None:
                break
            decoded += 1
    finally:
        probe.release()
    valid = not interrupted and decoded == frames and frames >= 2
    metadata = {'schema_version': 1, 'status': 'valid' if valid else 'invalid',
                'stream_ended_during_countdown': countdown_stream_failed,
                'read_timeout_ms': args.read_timeout_ms,
                'label': args.label, 'scenario': safe_name(args.name), 'camera_id': args.camera_id,
                'video_sha256': sha256(out_path), 'encoded_fps': args.fps,
                'encoded_duration_seconds': frames / args.fps, 'capture_wall_seconds': actual,
                'frame_times_seconds': frame_times, 'frames_written': frames, 'frames_decoded': decoded,
                'max_gap_seconds': max(gaps, default=0),
                'large_gap_count': sum(gap > max(0.5, 3 / args.fps) for gap in gaps),
                'timestamp_definition': 'PC receipt/selection time relative to recording start; not camera exposure time',
                'note': 'Replay must use this sidecar; fixed-FPS video playback alone can compress capture stalls.'}
    sidecar(out_path, 'capture').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"Saved: {out_path}")
    print(f"Frames: {frames} | wall duration: {actual:.2f}s | encoded FPS: {args.fps:g}")
    print(f"Timing sidecar: {sidecar(out_path, 'capture')} | max gap: {max(gaps, default=0):.3f}s")
    if not valid:
        print('Recording is invalid; excluded from evaluation. Keep the file/sidecar for diagnosis.', file=sys.stderr)
        return 5
    print("Next: repeat with another scenario, then run tools/evaluate_person_down.py on the dataset folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
