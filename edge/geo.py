"""
AICOS Ground Calibration
------------------------
Converts a position in the camera image into real-world coordinates, so an
incident is reported where the PERSON is rather than where the camera is.

Without calibration, every event from a camera lands on the camera's own
coordinates — accurate to the size of its field of view, typically 20-100
metres. With four reference points, accuracy drops to a few metres.

How it works: a fixed camera looking at flat ground sees the world through
a projective transform. Given four or more points whose image position AND
real position are both known, that transform can be recovered exactly, and
then any pixel can be converted to a ground position.

Calibrating a camera (this is what a site survey is for):

  1. Freeze a frame from the camera and pick 4+ landmarks on the GROUND
     that are easy to identify — a corner of a building, a lamppost base,
     the end of a road marking, a manhole cover. Spread them across the
     view; do not choose four points in a line.
  2. Note each landmark's position in the image as a FRACTION of width
     and height, so the calibration survives a resolution change.
  3. Find the same landmarks in satellite view and read off their real
     coordinates.
  4. Put the pairs in cameras.yaml:

        calibration:
          - image: [0.21, 0.86]
            world: [34.87131, 3.31402]
          - image: [0.78, 0.83]
            world: [34.87145, 3.31489]
          - image: [0.65, 0.42]
            world: [34.87210, 3.31474]
          - image: [0.29, 0.44]
            world: [34.87198, 3.31418]

Points must lie on the ground plane. A rooftop corner or the top of a pole
will skew the result, because the maths assumes a single flat surface.

No third-party dependency: the homography is solved here in plain Python,
so this works on any edge device without numpy or OpenCV.
"""

from __future__ import annotations

import math

# metres per degree, near enough for the few hundred metres a camera covers
_M_PER_DEG_LAT = 110540.0
_M_PER_DEG_LON = 111320.0


class CalibrationError(ValueError):
    """Raised when the supplied reference points cannot define a transform."""


# ---------------------------------------------------------------------------
# linear algebra (small, exact, dependency-free)
# ---------------------------------------------------------------------------


def _solve(matrix, rhs):
    """Solve a square system by Gaussian elimination with partial pivoting."""
    n = len(matrix)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise CalibrationError(
                "reference points are degenerate — are four of them in a "
                "straight line, or two at the same spot?")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / pv
            if factor:
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
    return [aug[i][i + 1 if False else n] / aug[i][i] for i in range(n)]


def solve_homography(src, dst):
    """Recover the 3x3 projective transform taking src points to dst points.

    Needs at least 4 correspondences. With more than 4 it solves the
    least-squares fit, which averages out small survey errors.
    """
    if len(src) != len(dst):
        raise CalibrationError("mismatched point lists")
    if len(src) < 4:
        raise CalibrationError(
            f"need at least 4 reference points, got {len(src)}")

    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v])
        rhs.append(v)

    # normal equations: (A^T A) h = A^T b  — exact for 4 points, least
    # squares for more
    n = 8
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for row, b in zip(rows, rhs):
        for i in range(n):
            atb[i] += row[i] * b
            for j in range(n):
                ata[i][j] += row[i] * row[j]
    h = _solve(ata, atb)
    return h + [1.0]


def apply_homography(h, x, y):
    """Map one point through the transform. None if it projects to infinity."""
    denom = h[6] * x + h[7] * y + h[8]
    if abs(denom) < 1e-12:
        return None
    return ((h[0] * x + h[1] * y + h[2]) / denom,
            (h[3] * x + h[4] * y + h[5]) / denom)


# ---------------------------------------------------------------------------


class GroundCalibration:
    """Turns a normalized image position into latitude and longitude."""

    def __init__(self, points):
        pairs = []
        for entry in (points or []):
            img = entry.get("image")
            world = entry.get("world")
            if not img or not world or len(img) < 2 or len(world) < 2:
                raise CalibrationError(
                    "each calibration entry needs image: [x, y] and "
                    "world: [lat, lon]")
            ix, iy = float(img[0]), float(img[1])
            if not (0.0 <= ix <= 1.0 and 0.0 <= iy <= 1.0):
                raise CalibrationError(
                    f"image point ({ix}, {iy}) is outside 0.0-1.0 — these are "
                    f"FRACTIONS of the frame, not pixels")
            pairs.append(((ix, iy), (float(world[0]), float(world[1]))))

        if len(pairs) < 4:
            raise CalibrationError(
                f"need at least 4 reference points, got {len(pairs)}")

        # Work in metres relative to the first point: latitude and longitude
        # are large numbers with small differences, which makes the fit
        # numerically fragile.
        self.lat0, self.lon0 = pairs[0][1]
        self._cos_lat = math.cos(math.radians(self.lat0)) or 1e-9

        src = [p[0] for p in pairs]
        dst = [self._to_metres(lat, lon) for _, (lat, lon) in pairs]
        self.h = solve_homography(src, dst)
        self.point_count = len(pairs)

    def _to_metres(self, lat, lon):
        return ((lon - self.lon0) * _M_PER_DEG_LON * self._cos_lat,
                (lat - self.lat0) * _M_PER_DEG_LAT)

    def _from_metres(self, east, north):
        return (self.lat0 + north / _M_PER_DEG_LAT,
                self.lon0 + east / (_M_PER_DEG_LON * self._cos_lat))

    def to_world(self, nx: float, ny: float):
        """Normalized image position (0-1) -> (lat, lon), or None."""
        try:
            point = apply_homography(self.h, float(nx), float(ny))
        except (TypeError, ValueError):
            return None
        if point is None:
            return None
        lat, lon = self._from_metres(point[0], point[1])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return (lat, lon)

    def residual_metres(self, points) -> float:
        """Largest error, in metres, when mapping the reference points back.

        A good calibration is under a couple of metres. A large value means
        a landmark was mis-identified or is not on the ground plane.
        """
        worst = 0.0
        for entry in points:
            ix, iy = float(entry["image"][0]), float(entry["image"][1])
            lat, lon = float(entry["world"][0]), float(entry["world"][1])
            got = self.to_world(ix, iy)
            if got is None:
                return float("inf")
            ge, gn = self._to_metres(*got)
            we, wn = self._to_metres(lat, lon)
            worst = max(worst, math.hypot(ge - we, gn - wn))
        return worst


def build(points, camera_id: str = "?"):
    """Construct a calibration, or return None and log why it failed.

    Never raises: a bad calibration must degrade to camera-level location,
    not stop the camera from working.
    """
    if not points:
        return None
    import logging
    log = logging.getLogger("aicos.edge")
    try:
        cal = GroundCalibration(points)
    except CalibrationError as exc:
        log.error("[%s] calibration ignored: %s", camera_id, exc)
        return None
    residual = cal.residual_metres(points)
    if residual > 10.0:
        log.warning("[%s] calibration fits poorly (worst point off by %.1f m) "
                    "— check the landmarks are on the ground and correctly "
                    "identified", camera_id, residual)
    else:
        log.info("[%s] ground calibration active: %d points, worst fit %.1f m",
                 camera_id, cal.point_count, residual)
    return cal
