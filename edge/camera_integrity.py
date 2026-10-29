"""Camera view-integrity and drift monitoring.

A camera that is bumped, re-aimed, covered or badly defocused can invalidate
zones, ground calibration and detector assumptions without disconnecting from
the network. This module compares a periodically sampled current frame against
a stored site-approved baseline using feature matching plus image-quality
signals. It does not upload the baseline to the central backend.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


def _resize_gray(frame, width: int = 640):
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("empty frame")
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, max(1, int(h * width / w))))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def quality_metrics(frame) -> dict:
    gray = _resize_gray(frame)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    darkness = float(np.mean(gray < 25))
    overexposure = float(np.mean(gray > 245))
    return {
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
        "dark_fraction": round(darkness, 4),
        "bright_fraction": round(overexposure, 4),
    }


def compare_frames(baseline, current) -> dict:
    """Return a conservative camera-integrity assessment.

    ORB/RANSAC estimates whether the scene geometry moved. Quality checks catch
    covered/black/defocused views where geometric matching is unreliable.
    Thresholds are deliberately conservative; production deployments should
    tune them during site acceptance, not assume one threshold fits every view.
    """
    b = _resize_gray(baseline)
    c = _resize_gray(current)
    if b.shape != c.shape:
        c = cv2.resize(c, (b.shape[1], b.shape[0]))
    q = quality_metrics(current)
    if q["dark_fraction"] > 0.98 and q["brightness"] < 8.0:
        return {"status": "obscured", "score": 5, "reason": "view is almost entirely dark", **q}
    if q["bright_fraction"] > 0.98 and q["brightness"] > 247.0:
        return {"status": "obscured", "score": 5, "reason": "view is almost entirely overexposed", **q}
    if q["sharpness"] < 8.0:
        return {"status": "degraded", "score": 25, "reason": "image is severely blurred/defocused", **q}

    orb = cv2.ORB_create(nfeatures=900, fastThreshold=10)
    kb, db = orb.detectAndCompute(b, None)
    kc, dc = orb.detectAndCompute(c, None)
    if db is None or dc is None or len(kb) < 12 or len(kc) < 12:
        return {"status": "degraded", "score": 30, "reason": "insufficient stable scene features", "keypoints_baseline": len(kb or []), "keypoints_current": len(kc or []), **q}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(db, dc), key=lambda m: m.distance)[:250]
    if len(matches) < 10:
        return {"status": "drift_suspected", "score": 20, "reason": "too few scene matches to baseline", "matches": len(matches), **q}

    src = np.float32([kb[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kc[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or mask is None:
        return {"status": "drift_suspected", "score": 20, "reason": "baseline geometry could not be recovered", "matches": len(matches), **q}
    inlier_ratio = float(mask.ravel().mean())
    h, w = b.shape[:2]
    corners = np.float32([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]])
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    original = corners.reshape(-1, 2)
    shifts = np.linalg.norm(warped - original, axis=1)
    diag = max(1.0, math.hypot(w, h))
    shift_pct = float(np.median(shifts) / diag)
    median_match_distance = float(np.median([m.distance for m in matches]))

    # Blend geometry and match stability into an operator-friendly 0-100 score.
    score = 100.0 - min(70.0, shift_pct * 700.0) - min(25.0, max(0.0, 0.45 - inlier_ratio) * 70.0)
    score = int(max(0, min(100, round(score))))
    if shift_pct >= 0.075:
        status, reason = "drift_suspected", "camera view moved materially from certified baseline"
    elif shift_pct >= 0.035 or inlier_ratio < 0.24:
        status, reason = "warning", "camera view differs from certified baseline"
    else:
        status, reason = "ok", "camera view remains consistent with certified baseline"
    return {
        "status": status, "score": score, "reason": reason,
        "inlier_ratio": round(inlier_ratio, 4), "corner_shift_pct": round(shift_pct, 5),
        "matches": len(matches), "median_match_distance": round(median_match_distance, 2), **q,
    }


class IntegrityMonitor:
    def __init__(self, camera_id: str, directory: str, auto_create: bool = True):
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(camera_id))
        self.camera_id = str(camera_id)
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_path = self.root / f"{safe}.baseline.jpg"
        self.meta_path = self.root / f"{safe}.baseline.json"
        self.auto_create = bool(auto_create)

    def has_baseline(self) -> bool:
        return self.image_path.is_file()

    def establish(self, frame, metadata: dict | None = None) -> dict:
        ok = cv2.imwrite(str(self.image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise OSError("could not write integrity baseline")
        meta = {"camera_id": self.camera_id, **(metadata or {})}
        self.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"status": "baseline", "score": 100, "reason": "certified view baseline established"}

    def assess(self, frame) -> dict:
        if not self.has_baseline():
            if not self.auto_create:
                return {"status": "uncalibrated", "score": 0, "reason": "no certified camera-view baseline"}
            return self.establish(frame)
        baseline = cv2.imread(str(self.image_path))
        if baseline is None:
            return {"status": "uncalibrated", "score": 0, "reason": "baseline file is unreadable"}
        return compare_frames(baseline, frame)
