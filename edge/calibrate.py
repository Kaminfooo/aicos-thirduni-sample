#!/usr/bin/env python3
"""
AICOS Calibration Helper
------------------------
Grabs one frame from a camera and saves it with a labelled grid overlaid,
so you can read off the image coordinates of your landmarks instead of
guessing them.

Workflow:

  1. python calibrate.py "http://192.168.1.12:8080/video"
     -> writes calibration-grid.png

  2. Open that image. Every gridline is labelled with its fraction of the
     frame (0.0 to 1.0). Find four landmarks ON THE GROUND — a wall
     corner, a doorstep, a tree base, the end of a paving line — and read
     off each one's x,y fraction from the grid.

  3. Find the same four spots in satellite view (Google Maps: long-press
     a point to read its coordinates) and note the lat,lon.

  4. Put the four pairs into cameras.yaml under that camera:

        calibration:
          - image: [0.31, 0.88]
            world: [34.87120, 3.31400]
          ... four in total

Rules that matter:
  * Ground-level points only. A rooftop corner or the top of a pole will
    skew the result — the maths assumes one flat surface.
  * Spread them across the view. Four points in a line cannot define the
    transform and will be rejected.
  * Pick things that are unambiguous in BOTH the camera image and the
    satellite image. That is what limits your accuracy in practice.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

GRID_STEP = 0.1


def gridlines(step: float = GRID_STEP):
    """Fractions at which to draw lines. Kept separate so it can be tested."""
    n = int(round(1.0 / step))
    return [round(i * step, 3) for i in range(n + 1)]


def label_positions(width: int, height: int, step: float = GRID_STEP):
    """Pixel positions and label text for every gridline."""
    out = []
    for f in gridlines(step):
        x = int(round(f * (width - 1)))
        y = int(round(f * (height - 1)))
        out.append(("v", x, f"{f:.1f}"))
        out.append(("h", y, f"{f:.1f}"))
    return out


def grab_frame(source: str, warmup: int = 5):
    """Read a frame from a file, stream, webcam index or snapshot URL."""
    import cv2

    low = source.lower()
    if low.startswith(("http://", "https://")) and \
            low.split("?")[0].endswith((".jpg", ".jpeg", ".png")):
        import numpy as np
        import requests
        resp = requests.get(source, timeout=15,
                            headers={"User-Agent": "AICOS-calibrate"})
        if resp.status_code != 200 or not resp.content:
            raise SystemExit(f"snapshot fetch failed: HTTP {resp.status_code}")
        frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit("that URL did not return a decodable image")
        return frame

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open: {source}")
    frame = None
    for _ in range(max(1, warmup)):        # let auto-exposure settle
        ok, got = cap.read()
        if ok and got is not None:
            frame = got
        time.sleep(0.05)
    cap.release()
    if frame is None:
        raise SystemExit("opened the source but got no frames")
    return frame


def draw_grid(frame, step: float = GRID_STEP):
    import cv2
    h, w = frame.shape[:2]
    out = frame.copy()
    for kind, pos, text in label_positions(w, h, step):
        major = text in ("0.0", "0.5", "1.0")
        colour = (0, 255, 255) if major else (0, 200, 0)
        thick = 2 if major else 1
        if kind == "v":
            cv2.line(out, (pos, 0), (pos, h - 1), colour, thick)
            cv2.putText(out, text, (min(pos + 4, w - 34), 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        else:
            cv2.line(out, (0, pos), (w - 1, pos), colour, thick)
            cv2.putText(out, text, (4, max(pos - 5, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    cv2.putText(out, "x = across, y = down.  Read landmark positions as fractions.",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                cv2.LINE_AA)
    return out


def check_existing(path: str) -> None:
    """If a cameras.yaml is given, report how well its calibration fits."""
    import yaml
    from geo import GroundCalibration, CalibrationError
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    found = False
    for cam in (cfg.get("cameras") or []):
        pts = cam.get("calibration")
        if not pts:
            continue
        found = True
        cam_id = cam.get("id", "?")
        try:
            cal = GroundCalibration(pts)
        except CalibrationError as exc:
            print(f"  {cam_id}: INVALID — {exc}")
            continue
        worst = cal.residual_metres(pts)
        verdict = ("excellent" if worst < 2 else
                   "usable" if worst < 5 else
                   "poor — re-check your landmarks")
        print(f"  {cam_id}: {len(pts)} points, worst fit {worst:.1f} m ({verdict})")
        corners = [(0.5, 0.95), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]
        for nx, ny in corners:
            got = cal.to_world(nx, ny)
            if got:
                print(f"      image ({nx}, {ny}) -> {got[0]:.5f}, {got[1]:.5f}")
    if not found:
        print("  no camera in that file has a calibration block yet")


def local_to_world(anchor_lat: float, anchor_lon: float, east_m: float,
                   north_m: float):
    """Convert metres east/north of an anchor into latitude and longitude.

    For indoor spaces, or anywhere a tape measure beats satellite view:
    pick one floor point as the origin, measure the others from it, and
    anchor the whole set at the building's coordinates. Relative accuracy
    is as good as your tape; absolute accuracy is as good as the anchor.
    """
    lat = anchor_lat + north_m / 110540.0
    lon = anchor_lon + east_m / (111320.0 * (math.cos(math.radians(anchor_lat)) or 1e-9))
    return (lat, lon)


def build_local_yaml(anchor, pairs):
    """pairs: [((img_x, img_y), (east_m, north_m)), ...] -> YAML text."""
    lat0, lon0 = anchor
    lines = ["    calibration:"]
    for (ix, iy), (east, north) in pairs:
        lat, lon = local_to_world(lat0, lon0, east, north)
        lines.append(f"      - image: [{ix}, {iy}]")
        lines.append(f"        world: [{lat:.7f}, {lon:.7f}]"
                     f"   # {east} m east, {north} m north of anchor")
    return "\n".join(lines)


def parse_pair(text: str):
    """'0.20,0.85 = 0,0' -> ((0.20, 0.85), (0.0, 0.0))"""
    if "=" not in text:
        raise SystemExit(f"--pair needs 'imgx,imgy = east,north' — got: {text}")
    left, right = text.split("=", 1)
    try:
        ix, iy = [float(v) for v in left.strip().split(",")]
        east, north = [float(v) for v in right.strip().split(",")]
    except ValueError:
        raise SystemExit(f"could not read numbers from: {text}")
    if not (0.0 <= ix <= 1.0 and 0.0 <= iy <= 1.0):
        raise SystemExit(f"image point ({ix}, {iy}) must be fractions 0.0-1.0")
    return ((ix, iy), (east, north))


def main() -> int:
    ap = argparse.ArgumentParser(description="AICOS calibration helper")
    ap.add_argument("source", nargs="?",
                    help="camera URL, file path, or webcam index")
    ap.add_argument("--out", default="calibration-grid.png")
    ap.add_argument("--step", type=float, default=GRID_STEP,
                    help="gridline spacing as a fraction (default 0.1)")
    ap.add_argument("--check", help="a cameras.yaml to validate instead")
    ap.add_argument("--anchor", help="lat,lon of your measuring origin "
                                     "(for --pair, e.g. indoor spaces)")
    ap.add_argument("--pair", action="append",
                    help="'imgx,imgy = east_m,north_m' relative to --anchor "
                         "(repeatable, give 4+)")
    args = ap.parse_args()

    if args.pair:
        if not args.anchor:
            ap.error("--pair needs --anchor lat,lon")
        try:
            alat, alon = [float(v) for v in args.anchor.split(",")]
        except ValueError:
            ap.error("--anchor must be 'lat,lon'")
        pairs = [parse_pair(p) for p in args.pair]
        if len(pairs) < 4:
            ap.error(f"need at least 4 --pair values, got {len(pairs)}")
        block = build_local_yaml((alat, alon), pairs)
        print("Paste this under the camera in cameras.yaml:\n")
        print(block)
        from geo import GroundCalibration, CalibrationError
        pts = []
        for (ix, iy), (east, north) in pairs:
            lat, lon = local_to_world(alat, alon, east, north)
            pts.append({"image": [ix, iy], "world": [lat, lon]})
        try:
            cal = GroundCalibration(pts)
            print(f"\nfit check: {len(pts)} points, worst "
                  f"{cal.residual_metres(pts):.2f} m")
        except CalibrationError as exc:
            print(f"\nPROBLEM: {exc}")
            return 1
        return 0

    if args.check:
        print(f"checking calibrations in {args.check}")
        check_existing(args.check)
        return 0
    if not args.source:
        ap.error("give a camera source, or --check a cameras.yaml")

    import cv2
    frame = grab_frame(args.source)
    h, w = frame.shape[:2]
    cv2.imwrite(args.out, draw_grid(frame, args.step))
    print(f"frame captured: {w}x{h}")
    print(f"saved: {os.path.abspath(args.out)}")
    print()
    print("Next:")
    print("  1. Open that image and find 4 landmarks ON THE GROUND, spread")
    print("     across the view (never four in a line).")
    print("  2. Read each landmark's x,y as fractions from the grid.")
    print("  3. Find the same spots in satellite view for their lat,lon.")
    print("  4. Add them to cameras.yaml, then verify with:")
    print(f"     python calibrate.py --check ../config/cameras-phone.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
