"""
AICOS Behavior Detection Suite (YOLO pack)
------------------------------------------
Real abnormal-behavior detection on top of a YOLO pose model, exposed
through the same Detector interface used by rtsp_worker.py.

Behaviors (all configurable in cameras.yaml -> settings.detectors.yolo):
  person_down   Person collapsed / lying immobile        -> emergency, SEV 4
  fight         Possible fight / altercation (heuristic) -> security,  SEV 3
  crowd         Crowd build-up / large crowd             -> security,  SEV 2-3
  panic         Several people running simultaneously    -> security,  SEV 4
  intrusion     Person inside a restricted zone          -> security,  SEV 3-4
  loitering     Person staying in a zone too long        -> security,  SEV 2

Design notes
  * ONE model inference per analyzed frame, shared by all analyzers.
  * All timing uses wall-clock timestamps, so behavior thresholds are
    independent of camera FPS or process_every_n_frames.
  * The suite keeps per-camera state (tracker, zone timers): create one
    instance PER CAMERA (rtsp_worker.build_detectors does this).
  * "fight" and "panic" are motion heuristics: they exist to bring a
    human eye to the feed fast, not to convict anyone. Severity and
    wording reflect that ("Possible...").

Install (laptop/server):   pip install -r requirements-yolo.txt
First run auto-downloads the model weights (~7 MB for yolov8n-pose).
Jetson: install NVIDIA's PyTorch wheel for your JetPack first.
"""

from __future__ import annotations

import math
import time

# COCO-17 keypoint indexes used for the "person down" torso check
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_MIN_CONF = 0.30

DEFAULTS = {
    "model": "yolov8n-pose.pt",
    "conf": 0.35,
    "imgsz": 640,
    "person_down": {
        "enabled": True,
        "seconds": 4.0,
        "gap_tolerance": 3.0,   # bridge detector dropouts of up to this long
        "max_drift": 0.7,       # how far they may drift (in body lengths) and still count as still
        "collapse_drop": 0.40,  # height loss that counts as going to the ground
        "collapse_window": 6.0, # they must have been upright this recently
        "still_drift": 0.35,    # motionless enough to still be the person who fell
        "upright_ratio": 0.70,  # boxes narrower than this are someone standing
        "site_max_seconds": 300.0,  # a site never lives longer than this
    },
    "fight": {
        "enabled": True,
        "seconds": 4.0,
        "proximity": 1.0,         # centre distance as a fraction of body height
        "min_speed": 0.5,         # body-heights/sec — above detector jitter
        "max_path_ratio": 0.45,   # below this = moving without going anywhere
        "min_gap_variation": 0.10,  # the pair's gap must open and close
        "min_limb_speed": 0.7,      # ONE pair of hands moving this fast counts,
                                    # even if the bodies stay put (striking)
        "secondary_limb_ratio": 0.25,  # the other person must be reacting, not
                                       # standing perfectly still
        "striking_hot_ratio": 0.35, # striking is bursty; grappling is sustained
        "max_travel": 0.25,         # ...but only if they are going nowhere.
                                    # Measured on real footage: a fight on the
                                    # spot drifts 0.01-0.15; walking hits 0.35,
                                    # and walkers swing their arms too.
    },
    "crowd": {"enabled": True, "warn": 8, "critical": 15},
    "panic": {"enabled": True, "min_runners": 3, "run_speed": 1.8},
    "intrusion": {"enabled": True},
    "loitering": {"enabled": True, "seconds": 120.0},
    # --- opt-in behaviors (enable per camera after validating on that view) ---
    "abandoned_object": {
        "enabled": False,
        "seconds": 60.0,          # unattended for this long -> alert
        "owner_radius": 0.12,     # a person this close (fraction of width) = attended
        "move_tolerance": 0.03,   # how far it may drift and still count as the same object
    },
    "snatch": {
        "enabled": False,
        "contact": 0.9,           # they must come this close first (body heights)
        "contact_samples": 2,     # and STAY close — a fly-past is not contact
        "flee_speed": 1.6,        # one leaves at running speed
        "window": 6.0,            # flight must follow contact within this long
        "separation": 2.5,        # and the gap must open to this many body heights
    },
}

DEFAULTS["vehicles"] = {
    "enabled": False,
    "stopped_seconds": 45.0,     # a vehicle motionless this long on a road
    "move_tolerance": 0.02,      # drift allowed while still counting as stopped
    "congestion_count": 6,       # this many near-motionless vehicles at once
    "wrong_way": {
        "enabled": False,
        "direction": "",         # "left"|"right"|"up"|"down" — normal traffic flow
        "min_speed": 0.15,       # fraction of frame width per second
    },
}
DEFAULTS["night"] = {
    "hours": "",                 # e.g. "21:00-06:00"; empty = no night rules
    "severity_boost": 1,         # security events get this much more severe
}

# COCO classes for carried items, used by abandoned-object detection
BAG_CLASSES = (24, 26, 28)   # backpack, handbag, suitcase
# COCO vehicle classes
VEHICLE_CLASSES = (2, 3, 5, 7)   # car, motorcycle, bus, truck


def _merge(base: dict, override: dict) -> dict:
    out = {}
    for k, v in base.items():
        if isinstance(v, dict):
            out[k] = _merge(v, (override or {}).get(k, {}) or {})
        else:
            out[k] = (override or {}).get(k, v)
    return out


# ---------------------------------------------------------------------------
# Geometry helpers (pure python — no numpy/cv2 required here)
# ---------------------------------------------------------------------------


def point_in_polygon(x: float, y: float, points) -> bool:
    """Ray-casting point-in-polygon. points = [(x, y), ...]."""
    inside = False
    n = len(points)
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _in_active_hours(spec: str, now: float) -> bool:
    """spec like '21:00-06:00' (may wrap midnight); empty = always active."""
    if not spec:
        return True
    try:
        a, b = spec.split("-")
        h1, m1 = (int(v) for v in a.split(":"))
        h2, m2 = (int(v) for v in b.split(":"))
    except (ValueError, AttributeError):
        return True
    t = time.localtime(now)
    cur = t.tm_hour * 60 + t.tm_min
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if start <= end:
        return start <= cur < end
    return cur >= start or cur < end   # wraps midnight


class Zone:
    def __init__(self, spec: dict):
        self.name = str(spec.get("name", "zone"))
        self.kind = str(spec.get("type", "intrusion")).lower()
        self.points = [(float(p[0]), float(p[1])) for p in (spec.get("points") or [])]
        self.active_hours = str(spec.get("active_hours", "") or "")

    def valid(self) -> bool:
        return len(self.points) >= 3

    def contains(self, nx: float, ny: float) -> bool:
        return point_in_polygon(nx, ny, self.points)

    def active(self, now: float) -> bool:
        return _in_active_hours(self.active_hours, now)


# ---------------------------------------------------------------------------
# Minimal multi-object tracker (greedy nearest-center matching)
# ---------------------------------------------------------------------------


class Track:
    __slots__ = ("tid", "hist", "missed", "down_since", "down_miss",
                 "down_fired_at", "zone_enter", "zone_fired", "run_hot",
                 "peak_h", "peak_ts", "collapsed", "ever_upright", "kp_hist")

    def __init__(self, tid: int, t: float, box):
        self.tid = tid
        self.hist = [(t, box)]          # [(ts, (x1, y1, x2, y2)), ...]
        self.peak_h = max(1.0, box[3] - box[1])   # tallest seen = upright height
        self.peak_ts = t
        self.collapsed = False
        # A fall is a TRANSITION. Something that has never been upright —
        # a curtain, a bag, a chair the model mistakes for a person — did
        # not fall, however motionless and however prone-shaped it looks.
        self.ever_upright = (box[2] - box[0]) < 0.70 * max(1.0, box[3] - box[1])
        self.kp_hist = []       # [(ts, [(x, y), ...wrists]), ...]
        self.missed = 0
        self.down_since = None
        self.down_miss = 0
        self.down_fired_at = 0.0
        self.zone_enter = {}            # zone name -> entered ts
        self.zone_fired = {}            # zone name -> last fired ts
        self.run_hot = 0

    def push(self, t: float, box) -> None:
        self.hist.append((t, box))
        if len(self.hist) > 14:
            self.hist.pop(0)
        self.missed = 0
        height = max(1.0, box[3] - box[1])
        if (box[2] - box[0]) < 0.70 * height:
            self.ever_upright = True
        if height >= self.peak_h:
            self.peak_h = height
            self.peak_ts = t
        else:
            # decay slowly, so someone genuinely walking further away
            # re-baselines instead of looking permanently collapsed
            self.peak_h *= 0.9995

    def push_keypoints(self, t: float, kpts) -> None:
        """Remember where this person's hands were."""
        if not kpts or len(kpts) <= KP_R_WRIST:
            return
        cx, cy = self.center()
        wrists = []
        for idx in (KP_L_WRIST, KP_R_WRIST):
            x, y, conf = kpts[idx]
            if conf >= KP_MIN_CONF:
                # relative to the body: walking carries the hands along with
                # the person, and that is not striking
                wrists.append((float(x) - cx, float(y) - cy))
        if not wrists:
            return
        self.kp_hist.append((t, wrists))
        if len(self.kp_hist) > 8:
            self.kp_hist.pop(0)

    def limb_speed(self) -> float:
        """How fast the hands are moving, in body lengths per second.

        A fight between two people standing their ground barely moves their
        bounding boxes — the arms do the work. Whole-body speed misses it
        entirely, which is why this exists.
        """
        if len(self.kp_hist) < 2:
            return 0.0
        total = 0.0
        samples = 0
        for i in range(1, len(self.kp_hist)):
            (t0, a), (t1, b) = self.kp_hist[i - 1], self.kp_hist[i]
            dt = t1 - t0
            if dt <= 0.01:
                continue
            n = min(len(a), len(b))
            if not n:
                continue
            step = sum(math.hypot(b[j][0] - a[j][0], b[j][1] - a[j][1])
                       for j in range(n)) / n
            total += step / dt
            samples += 1
        if not samples:
            return 0.0
        return (total / samples) / self.body_length()

    def height_collapse(self) -> float:
        """How much shorter this person is than when they were upright.

        Orientation-independent, which the width-to-height test is not: a
        body lying with its head toward or away from the camera stays
        NARROW in the image and looks just like someone standing. What
        always changes is height — a person on the ground takes up far
        less vertical space than the same person on their feet.
        """
        return max(0.0, 1.0 - (self.height() / max(1.0, self.peak_h)))

    def box(self):
        return self.hist[-1][1]

    def center(self):
        x1, y1, x2, y2 = self.box()
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def height(self) -> float:
        x1, y1, x2, y2 = self.box()
        return max(1.0, y2 - y1)

    def body_length(self) -> float:
        """The person's long axis, whichever way they are oriented.

        Speeds are normalized by this rather than by height. A prone
        person's box is short, so dividing by height made a few pixels of
        detector jitter read as fast movement — and the stillness check
        then rejected exactly the people person-down exists to find.
        """
        x1, y1, x2, y2 = self.box()
        return max(1.0, y2 - y1, x2 - x1)

    def feet(self):
        x1, y1, x2, y2 = self.box()
        return ((x1 + x2) / 2.0, y2)

    def _centers(self):
        return [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for _, b in self.hist]

    def path_length(self) -> float:
        pts = self._centers()
        return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                   for i in range(1, len(pts)))

    def speed_norm(self) -> float:
        """Mean speed in body-heights per second, measured ALONG the path.

        Path-based (not net displacement) so a person struggling on the spot
        still reads as fast-moving — that is the signal a fight produces.
        """
        if len(self.hist) < 2:
            return 0.0
        dt = self.hist[-1][0] - self.hist[0][0]
        if dt <= 0.05:
            return 0.0
        return self.path_length() / dt / self.body_length()

    def recent_speed(self, samples: int = 4) -> float:
        """Speed over the last few samples only, in body-heights per second.

        A whole-history average dilutes a sudden burst — someone who stood
        still then sprinted reads as slow. Snatch detection needs the burst.
        """
        if len(self.hist) < 2:
            return 0.0
        hist = self.hist[-max(2, samples):]
        dt = hist[-1][0] - hist[0][0]
        if dt <= 0.05:
            return 0.0
        total = 0.0
        for i in range(1, len(hist)):
            (_, b0), (_, b1) = hist[i - 1], hist[i]
            dx = ((b1[0] + b1[2]) - (b0[0] + b0[2])) / 2.0
            dy = ((b1[1] + b1[3]) - (b0[1] + b0[3])) / 2.0
            total += math.hypot(dx, dy)
        return total / dt / self.body_length()

    def net_drift(self, samples: int = 6) -> float:
        """How far the person has actually MOVED recently, as a fraction of
        body length.

        Uses net displacement, not path length: detector jitter cancels out
        because it wanders back and forth, while real movement does not.
        This is the right question for stillness — "are they staying put?" —
        whereas path-based speed accumulates noise and wrongly reports a
        motionless body as moving.
        """
        if len(self.hist) < 2:
            return 0.0
        hist = self.hist[-max(2, samples):]
        (_, b0), (_, b1) = hist[0], hist[-1]
        dx = ((b1[0] + b1[2]) - (b0[0] + b0[2])) / 2.0
        dy = ((b1[1] + b1[3]) - (b0[1] + b0[3])) / 2.0
        return math.hypot(dx, dy) / self.body_length()

    def path_ratio(self) -> float:
        """Net displacement / distance travelled.

        ~1.0 = walking in a straight line (a pedestrian going somewhere).
        Near 0 = moving hard but going nowhere (a scuffle, milling about).
        This is what separates a fight from two people walking side by side.
        """
        if len(self.hist) < 3:
            return 1.0
        path = self.path_length()
        if path < 1e-6:
            return 1.0
        pts = self._centers()
        net = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        return min(1.0, net / path)

    def direction_flips(self) -> int:
        flips, last = 0, 0.0
        for i in range(1, len(self.hist)):
            dx = (self.hist[i][1][0] + self.hist[i][1][2]) - \
                 (self.hist[i - 1][1][0] + self.hist[i - 1][1][2])
            if dx * last < 0:
                flips += 1
            if abs(dx) > 1.0:
                last = dx
        return flips


class SimpleTracker:
    def __init__(self):
        self._next = 1
        self.tracks: dict[int, Track] = {}

    def update(self, t: float, boxes, frame_w: float):
        unmatched = list(range(len(boxes)))
        # match existing tracks to nearest new box
        for tr in list(self.tracks.values()):
            cx, cy = tr.center()
            # constant-velocity prediction so fast movers stay matched
            if len(tr.hist) >= 2:
                (t0, b0), (t1, b1) = tr.hist[-2], tr.hist[-1]
                dt_h = max(1e-3, t1 - t0)
                vx = (((b1[0] + b1[2]) - (b0[0] + b0[2])) / 2.0) / dt_h
                vy = (((b1[1] + b1[3]) - (b0[1] + b0[3])) / 2.0) / dt_h
                dt = max(0.0, t - t1)
                cx, cy = cx + vx * dt, cy + vy * dt
            max_dist = max(60.0, 0.19 * frame_w, 1.8 * tr.body_length())
            best, best_d = None, max_dist
            for i in unmatched:
                x1, y1, x2, y2 = boxes[i]
                d = math.hypot((x1 + x2) / 2.0 - cx, (y1 + y2) / 2.0 - cy)
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                tr.push(t, boxes[best])
                unmatched.remove(best)
            else:
                tr.missed += 1
        # new tracks
        for i in unmatched:
            tr = Track(self._next, t, boxes[i])
            self.tracks[tr.tid] = tr
            self._next += 1
        # prune lost tracks
        for tid in [k for k, tr in self.tracks.items() if tr.missed > 10]:
            del self.tracks[tid]
        return [tr for tr in self.tracks.values() if tr.missed == 0]


# ---------------------------------------------------------------------------
# Model engine (ultralytics wrapper) — lazily imported, easily faked in tests
# ---------------------------------------------------------------------------


class UltralyticsEngine:
    """Wraps a YOLO (pose) model. infer(frame) -> list of
    {"box": (x1,y1,x2,y2), "conf": float, "kpts": [...] | None, "cls": int}

    `classes` selects which COCO classes to detect. Person (0) is always
    included; bag classes are added only when abandoned-object detection
    is enabled, so the default deployment costs nothing extra.
    """

    def __init__(self, model_path: str, conf: float, imgsz: int, classes=(0,)):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install -r requirements-yolo.txt"
            ) from exc
        # VPS/cloud CPUs often lack NNPACK support; torch then prints a
        # warning per conv op, flooding logs. Disable NNPACK through every
        # API torch exposes; harmless if unsupported (default kernels run).
        try:
            import torch
            nnp = getattr(torch.backends, "nnpack", None)
            if nnp is not None:
                for setter in ("enabled",):
                    try:
                        setattr(nnp, setter, False)
                    except Exception:
                        pass
                try:
                    nnp.set_flags(False)
                except Exception:
                    pass
        except Exception:
            pass
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.classes = list(classes) or [0]

    def infer(self, frame):
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 classes=self.classes, verbose=False)[0]
        out = []
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        kp = getattr(res, "keypoints", None)
        kp_data = None
        if kp is not None and getattr(kp, "data", None) is not None:
            try:
                kp_data = kp.data.cpu().numpy()
            except Exception:
                kp_data = None
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            kpts = None
            if kp_data is not None and i < len(kp_data):
                kpts = [(float(x), float(y), float(c)) for x, y, c in kp_data[i]]
            try:
                cls_id = int(boxes.cls[i].cpu().numpy())
            except Exception:
                cls_id = 0
            out.append({"box": tuple(xyxy), "conf": conf, "kpts": kpts, "cls": cls_id})
        return out


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


class YoloBehaviorSuite:
    """One instance per camera. detect(frame, prev_frame, now=None)
    returns a list of AICOS event dicts."""

    name = "yolo-behavior"

    def __init__(self, camera: dict = None, config: dict = None, engine=None):
        self.cfg = _merge(DEFAULTS, config or {})
        self.tracker = SimpleTracker()
        self.zones = [z for z in (Zone(s) for s in (camera or {}).get("zones") or [])
                      if z.valid()]
        self._engine = engine   # injected in tests; lazily built otherwise
        self._pair_state = {}   # (tid_a, tid_b) -> {since, samples, hot, fired_at}
        self._crowd_fired = {"warn": 0.0, "critical": 0.0}
        self._panic_fired = 0.0
        self._fw, self._fh = 1.0, 1.0   # frame size, refreshed each detect()
        self._down_sites = {}   # places where someone is lying, by position
        self._next_site = 1
        self._objects = {}      # abandoned-object candidates
        self._next_obj = 1
        self._contact_state = {}   # pairs that recently came into contact
        self._vehicles = {}        # vehicle tracks for stopped/wrong-way logic
        self._next_veh = 1
        self._congestion_fired = 0.0

    # -- engine ------------------------------------------------------------
    def _get_engine(self):
        if self._engine is None:
            classes = [0]
            if self.cfg["abandoned_object"]["enabled"]:
                classes += list(BAG_CLASSES)
            if self.cfg["vehicles"]["enabled"]:
                classes += list(VEHICLE_CLASSES)
            self._engine = UltralyticsEngine(
                self.cfg["model"], float(self.cfg["conf"]),
                int(self.cfg["imgsz"]), classes=classes)
        return self._engine

    # -- main entry --------------------------------------------------------
    def detect(self, frame, prev_frame=None, now: float = None):
        now = time.time() if now is None else now
        h, w = frame.shape[:2]
        self._fw, self._fh = float(w), float(h)
        dets = self._get_engine().infer(frame)
        persons = [d for d in dets if d.get("cls", 0) == 0]
        items = [d for d in dets if d.get("cls", 0) in BAG_CLASSES]
        vehicles = [d for d in dets if d.get("cls", 0) in VEHICLE_CLASSES]
        boxes = [d["box"] for d in persons]
        live = self.tracker.update(now, boxes, float(w))
        # attach keypoints to matched tracks by box identity
        kmap = {tuple(d["box"]): d.get("kpts") for d in persons}
        for tr in live:
            tr.push_keypoints(now, kmap.get(tuple(tr.box())))
        events = []
        if self.cfg["person_down"]["enabled"]:
            events += self._person_down(live, kmap, now)
        if self.cfg["fight"]["enabled"]:
            events += self._fight(live, now)
        if self.cfg["crowd"]["enabled"]:
            events += self._crowd(live, now)
        if self.cfg["panic"]["enabled"]:
            events += self._panic(live, now)
        if self.zones and (self.cfg["intrusion"]["enabled"] or self.cfg["loitering"]["enabled"]):
            events += self._zones(live, now, float(w), float(h))
        if self.cfg["abandoned_object"]["enabled"]:
            events += self._abandoned(items, persons, now, float(w))
        if self.cfg["snatch"]["enabled"]:
            events += self._snatch(live, now)
        if self.cfg["vehicles"]["enabled"]:
            events += self._vehicle(vehicles, now, float(w), float(h))
        return self._apply_night(events, now)

    # -- person down -------------------------------------------------------
    @staticmethod
    def _torso_horizontal(kpts) -> bool:
        def mid(a, b):
            if kpts[a][2] < KP_MIN_CONF or kpts[b][2] < KP_MIN_CONF:
                return None
            return ((kpts[a][0] + kpts[b][0]) / 2.0, (kpts[a][1] + kpts[b][1]) / 2.0)
        sh = mid(KP_L_SHOULDER, KP_R_SHOULDER)
        hp = mid(KP_L_HIP, KP_R_HIP)
        if sh is None or hp is None:
            return False
        dx, dy = abs(sh[0] - hp[0]), abs(sh[1] - hp[1])
        return dx > dy * 1.15   # torso axis closer to horizontal than vertical

    def _looks_down(self, tr: Track, kpts) -> bool:
        if kpts is not None and self._torso_horizontal(kpts):
            return True
        x1, y1, x2, y2 = tr.box()
        return (x2 - x1) > (y2 - y1) * 1.35   # wide, flat bbox fallback

    def _mark_down_site(self, tr, now: float) -> None:
        """Record that somebody is lying at this position.

        Anchored to the PLACE, not the track id. Detectors lose their lock
        on a prone body constantly — it looks nothing like the upright
        people they were trained on — and every re-acquisition creates a
        new track. Tying the timer to the ground position instead means a
        dropout no longer restarts the clock on a real medical emergency.
        """
        cx, cy = tr.center()
        radius = max(60.0, 2.0 * tr.body_length())
        key, best = None, radius
        for site_id, site in self._down_sites.items():
            dist = math.hypot(cx - site["cx"], cy - site["cy"])
            if dist < best:
                key, best = site_id, dist
        if key is None:
            key = self._next_site
            self._next_site += 1
            self._down_sites[key] = {"cx": cx, "cy": cy, "first": now,
                                     "last": now, "fired_at": 0.0}
        else:
            site = self._down_sites[key]
            site["cx"] = 0.3 * cx + 0.7 * site["cx"]
            site["cy"] = 0.3 * cy + 0.7 * site["cy"]
            site["last"] = now

    def _person_down(self, live, kmap, now):
        cfg = self.cfg["person_down"]
        need = float(cfg["seconds"])
        grace = float(cfg.get("gap_tolerance", 3.0))
        out = []

        collapse_drop = float(cfg.get("collapse_drop", 0.40))
        collapse_window = float(cfg.get("collapse_window", 6.0))
        max_drift = float(cfg.get("max_drift", 0.7))

        for tr in live:
            # Latch a collapse: this person was upright recently and is now
            # far shorter. Cleared only when they regain their height, so it
            # holds for the whole time they stay on the ground.
            drop = tr.height_collapse()
            if drop >= collapse_drop and (now - tr.peak_ts) <= collapse_window:
                tr.collapsed = True
            elif drop < collapse_drop * 0.6:
                tr.collapsed = False

            # stillness = they have not travelled anywhere, judged by net
            # drift so that detector jitter on a prone body is not mistaken
            # for movement
            posture_down = self._looks_down(tr, kmap.get(tuple(tr.box())))
            if (posture_down or tr.collapsed) and tr.net_drift() < max_drift \
                    and tr.ever_upright:
                self._mark_down_site(tr, now)

        # Once a fall has been established at a spot, someone lying
        # motionless there keeps the clock running even if the detector can
        # no longer read the posture. Models lose their lock on prone bodies
        # and re-acquire them as a NEW person with no upright history —
        # demanding fresh proof of posture every frame throws away the fact
        # that we already saw them go down.
        still_drift = float(cfg.get("still_drift", 0.35))
        upright_ratio = float(cfg.get("upright_ratio", 0.70))
        for tr in live:
            if tr.net_drift() >= still_drift:
                continue                      # moving — not lying there
            x1, y1, x2, y2 = tr.box()
            if (x2 - x1) / max(1.0, y2 - y1) < upright_ratio:
                continue                      # clearly standing — not the
                                              # person on the ground
            cx, cy = tr.center()
            reach = max(60.0, 2.0 * tr.body_length())
            for site in self._down_sites.values():
                if math.hypot(cx - site["cx"], cy - site["cy"]) < reach:
                    site["last"] = now
                    break

        max_life = float(cfg.get("site_max_seconds", 300.0))
        for site_id in list(self._down_sites):
            site = self._down_sites[site_id]
            if now - site["first"] > max_life:
                del self._down_sites[site_id]   # long resolved, or never real
                continue
            if now - site["last"] > grace:
                del self._down_sites[site_id]      # they got up, or moved off
                continue
            held = site["last"] - site["first"]
            if held >= need and (now - site["fired_at"]) > 120:
                site["fired_at"] = now
                out.append(self._px({
                    "agent": "emergency",
                    "type": "Person down / possible medical emergency",
                    "severity": 4,
                    "confidence": round(min(0.9, 0.55 + 0.04 * int(held)), 2),
                    "extra": {"site": site_id, "down_seconds": int(held),
                              "lead_seconds": round(held, 1)},
                }, site["cx"], site["cy"], self._fw, self._fh))
        return out

    # -- fight (heuristic) -------------------------------------------------
    def _fight(self, live, now):
        """Flag a pair that stays engaged while moving hard but going nowhere.

        Three signals must hold together:
          * proximity sustained over several seconds (brief breaks allowed —
            people in a scuffle separate and re-engage);
          * both moving more than detector jitter, yet with low path_ratio,
            i.e. moving without travelling anywhere;
          * the gap between them opening and closing, which grappling
            produces and a standing conversation does not.

        Pedestrians walking together fail the second test (they travel in a
        line); people talking fail the third (their gap is steady).
        """
        out = []
        fcfg = self.cfg["fight"]
        need = float(fcfg["seconds"])
        prox = float(fcfg["proximity"])
        min_speed = float(fcfg["min_speed"])
        max_ratio = float(fcfg["max_path_ratio"])
        min_gap_var = float(fcfg["min_gap_variation"])
        min_limb = float(fcfg.get("min_limb_speed", 0.7))
        secondary_ratio = float(fcfg.get("secondary_limb_ratio", 0.25))
        strike_ratio = float(fcfg.get("striking_hot_ratio", 0.35))
        max_travel = float(fcfg.get("max_travel", 0.25))
        seen_keys = set()

        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                (ax, ay), (bx, by) = a.center(), b.center()
                avg_h = max(1.0, (a.height() + b.height()) / 2.0)
                gap = math.hypot(ax - bx, ay - by) / avg_h
                key = (min(a.tid, b.tid), max(a.tid, b.tid))
                st = self._pair_state.get(key)

                if gap >= prox:
                    # brief separation is part of a scuffle; only give up on
                    # the pair once they have stayed apart
                    if st is not None:
                        st["apart"] += 1
                        if st["apart"] > 3:
                            del self._pair_state[key]
                        else:
                            seen_keys.add(key)
                            st["gaps"].append(gap)
                            if len(st["gaps"]) > 12:
                                st["gaps"].pop(0)
                    continue

                seen_keys.add(key)
                if st is None:
                    st = self._pair_state[key] = {
                        "since": now, "samples": 0, "hot": 0,
                        "fired_at": 0.0, "gaps": [], "apart": 0,
                        "striking": 0,
                    }
                st["apart"] = 0
                st["samples"] += 1
                st["gaps"].append(gap)
                if len(st["gaps"]) > 12:
                    st["gaps"].pop(0)

                # Either signature counts: bodies grappling (moving hard
                # but going nowhere), or hands flying while they stand
                # their ground. A fight can be one or the other.
                grappling = (
                    a.speed_norm() > min_speed and b.speed_norm() > min_speed
                    and a.path_ratio() < max_ratio and b.path_ratio() < max_ratio
                )
                # Not travelling: net drift, not path_ratio. Someone
                # standing perfectly still has a path_ratio of 1.0, which
                # reads as "walking in a straight line" — exactly wrong for
                # two people trading blows on the spot.
                # An assault is usually ONE-SIDED: one person strikes while
                # the other shields or backs away. Requiring both pairs of
                # hands to move fast describes a boxing match, not what
                # cameras actually see. So: one clearly striking, the other
                # reacting rather than standing still.
                fast_hands = max(a.limb_speed(), b.limb_speed())
                slow_hands = min(a.limb_speed(), b.limb_speed())
                striking = (
                    fast_hands > min_limb
                    and slow_hands > min_limb * secondary_ratio
                    and a.net_drift() < max_travel and b.net_drift() < max_travel
                )
                agitated = grappling or striking
                st["hot"] += 1 if agitated else 0
                st["striking"] = st.get("striking", 0) + (1 if striking else 0)
                ratio = st["hot"] / st["samples"]

                gaps = st["gaps"]
                gap_var = 0.0
                if len(gaps) >= 4:
                    mean = sum(gaps) / len(gaps)
                    gap_var = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5

                # Grappling makes the gap open and close; two people trading
                # blows hold their stance, so striking satisfies the
                # movement requirement on its own.
                movement_ok = gap_var >= min_gap_var or st.get("striking", 0) >= 4
                strike_rate = st.get("striking", 0) / st["samples"]
                # Grappling is continuous, so it must hold most of the window.
                # Striking comes in bursts between pauses, so it is judged on
                # its own lower hit rate.
                sustained = (ratio >= 0.6 and movement_ok) or \
                            strike_rate >= strike_ratio
                if (now - st["since"]) >= need and st["samples"] >= 5 and \
                        sustained and (now - st["fired_at"]) > 90:
                    st["fired_at"] = now
                    out.append(self._px({
                        "agent": "security",
                        "type": "Possible fight / altercation",
                        "severity": 3,
                        "confidence": round(min(0.75, 0.45 + 0.3 * ratio), 2),
                        "extra": {"tracks": list(key),
                                  "lead_seconds": round(now - st["since"], 1)},
                    }, (ax + bx) / 2.0, (ay + by) / 2.0, self._fw, self._fh))

        # crowds churn tracks constantly — drop state for pairs that are gone
        for key in [k for k in self._pair_state if k not in seen_keys]:
            del self._pair_state[key]
        return out

    # -- crowd -------------------------------------------------------------
    def _crowd(self, live, now):
        out = []
        n = len(live)
        warn = int(self.cfg["crowd"]["warn"])
        crit = int(self.cfg["crowd"]["critical"])
        if n >= crit and (now - self._crowd_fired["critical"]) > 120:
            self._crowd_fired["critical"] = now
            out.append({"agent": "security", "type": "Large crowd forming",
                        "severity": 3, "confidence": 0.8, "extra": {"count": n}})
        elif n >= warn and (now - self._crowd_fired["warn"]) > 120:
            self._crowd_fired["warn"] = now
            out.append({"agent": "security", "type": "Crowd build-up",
                        "severity": 2, "confidence": 0.7, "extra": {"count": n}})
        return out

    # -- panic (several runners) -------------------------------------------
    def _panic(self, live, now):
        thr = float(self.cfg["panic"]["run_speed"])
        runners = 0
        for tr in live:
            tr.run_hot = tr.run_hot + 1 if tr.speed_norm() > thr else 0
            if tr.run_hot >= 2:
                runners += 1
        if runners >= int(self.cfg["panic"]["min_runners"]) and \
                (now - self._panic_fired) > 90:
            self._panic_fired = now
            return [{"agent": "security", "type": "Panic movement detected",
                     "severity": 4, "confidence": 0.7, "extra": {"runners": runners}}]
        return []

    @staticmethod
    def _px(event: dict, x: float, y: float, w: float, h: float) -> dict:
        """Attach the normalized image position an event happened at.

        The edge worker turns this into real coordinates when the camera has
        a ground calibration, so the incident is reported where the person
        is rather than where the camera is.
        """
        event.setdefault("extra", {})["px"] = [
            round(max(0.0, min(1.0, x / max(1.0, w))), 4),
            round(max(0.0, min(1.0, y / max(1.0, h))), 4),
        ]
        return event

    # -- night rules -------------------------------------------------------
    def _apply_night(self, events, now):
        """Raise severity of security events during configured night hours.

        The same behavior means something different at 03:00 than at 15:00 —
        a person in a closed yard, a crowd forming outside a shut shop.
        Times use the LOCAL clock of the machine running the worker.
        """
        night = self.cfg.get("night") or {}
        hours = str(night.get("hours", "") or "")
        if not hours or not events or not _in_active_hours(hours, now):
            return events
        boost = int(night.get("severity_boost", 1))
        for ev in events:
            if ev.get("agent") == "security":
                before = int(ev.get("severity", 1))
                ev["severity"] = max(1, min(5, before + boost))
                if ev["severity"] != before:
                    ev.setdefault("extra", {})["night_escalated"] = True
        return events

    # -- vehicles ----------------------------------------------------------
    def _vehicle(self, vehicles, now, frame_w, frame_h):
        """Stopped vehicles, congestion, and wrong-way movement.

        Tracked separately from people: vehicles are matched by position and
        judged on stillness and heading, not posture.
        """
        cfg = self.cfg["vehicles"]
        need = float(cfg["stopped_seconds"])
        tol = float(cfg["move_tolerance"]) * frame_w
        wcfg = cfg.get("wrong_way") or {}
        want_dir = str(wcfg.get("direction", "") or "").lower()
        min_speed = float(wcfg.get("min_speed", 0.15)) * frame_w
        out = []
        seen = set()
        used = set()          # a track may claim at most one detection per frame
        still = 0

        for d in vehicles:
            x1, y1, x2, y2 = d["box"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            box_w = max(20.0, x2 - x1)
            # Match radius scales with the vehicle and its last known speed —
            # a car crossing the frame moves far between analyzed frames.
            # Kept below typical vehicle spacing so neighbouring cars are not
            # merged; velocity prediction covers fast movers instead.
            radius = max(tol * 3.0, 0.9 * box_w)
            key, best = None, radius
            for k, st in self._vehicles.items():
                if k in used:
                    continue
                dt_pred = max(0.0, now - st["last"])
                px = st["cx"] + st.get("vx", 0.0) * dt_pred
                py = st["cy"] + st.get("vy", 0.0) * dt_pred
                dist = math.hypot(cx - px, cy - py)
                if dist < best:
                    key, best = k, dist
            if key is None:
                key = self._next_veh
                self._next_veh += 1
                self._vehicles[key] = {"cx": cx, "cy": cy, "vx": 0.0, "vy": 0.0,
                                       "since": now, "last": now,
                                       "fired_at": 0.0, "wrong_at": 0.0}
                st = self._vehicles[key]
            else:
                st = self._vehicles[key]
                dx, dy = cx - st["cx"], cy - st["cy"]
                dt = max(1e-3, now - st["last"])
                st["vx"], st["vy"] = dx / dt, dy / dt
                moved = math.hypot(dx, dy)
                if moved > tol:
                    st["since"] = now         # it is moving — reset the stopped clock
                    if want_dir and wcfg.get("enabled") and (moved / dt) > min_speed:
                        wrong = (
                            (want_dir == "right" and dx < -tol) or
                            (want_dir == "left" and dx > tol) or
                            (want_dir == "down" and dy < -tol) or
                            (want_dir == "up" and dy > tol)
                        )
                        if wrong and (now - st["wrong_at"]) > 60:
                            st["wrong_at"] = now
                            out.append({
                                "agent": "traffic",
                                "type": "Vehicle travelling against traffic",
                                "severity": 4,
                                "confidence": 0.7,
                                "extra": {"vehicle": key, "expected_direction": want_dir},
                            })
                st["cx"], st["cy"] = cx, cy
                st["last"] = now
            seen.add(key)
            used.add(key)

            stopped_for = now - st["since"]
            if stopped_for > 5:
                still += 1
            if stopped_for >= need and (now - st["fired_at"]) > 300:
                st["fired_at"] = now
                out.append({
                    "agent": "traffic",
                    "type": "Vehicle stopped on the road",
                    "severity": 3,
                    "confidence": 0.75,
                    "extra": {"vehicle": key, "stopped_seconds": int(stopped_for)},
                })

        if still >= int(cfg["congestion_count"]) and (now - self._congestion_fired) > 300:
            self._congestion_fired = now
            out.append({
                "agent": "traffic",
                "type": "Traffic congestion building",
                "severity": 2,
                "confidence": 0.7,
                "extra": {"stationary_vehicles": still},
            })

        for k in [k for k, v in self._vehicles.items() if now - v["last"] > 15]:
            del self._vehicles[k]
        return out

    # -- abandoned / unattended object -------------------------------------
    def _abandoned(self, items, persons, now, frame_w):
        """Flag a bag or case left with nobody near it.

        A genuine security signal (unattended luggage) and one of the few
        theft-adjacent behaviors a camera can judge reliably, because it is
        about an object's stillness rather than a person's intent.
        """
        cfg = self.cfg["abandoned_object"]
        need = float(cfg["seconds"])
        tol = float(cfg["move_tolerance"]) * frame_w
        radius = float(cfg["owner_radius"]) * frame_w
        out = []
        seen = set()

        centers = [(((p["box"][0] + p["box"][2]) / 2.0),
                    ((p["box"][1] + p["box"][3]) / 2.0)) for p in persons]

        for d in items:
            x1, y1, x2, y2 = d["box"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            key, best_d = None, tol
            for k, st in self._objects.items():
                dist = math.hypot(cx - st["cx"], cy - st["cy"])
                if dist < best_d:
                    key, best_d = k, dist
            if key is None:
                key = self._next_obj
                self._next_obj += 1
                self._objects[key] = {"cx": cx, "cy": cy, "since": now,
                                      "last": now, "fired_at": 0.0}
            st = self._objects[key]
            # smooth the position so detector jitter is not read as movement
            st["cx"] = 0.3 * cx + 0.7 * st["cx"]
            st["cy"] = 0.3 * cy + 0.7 * st["cy"]
            st["last"] = now
            seen.add(key)

            attended = any(math.hypot(px - cx, py - cy) < radius for px, py in centers)
            if attended:
                st["since"] = now          # owner present — restart the clock
            elif (now - st["since"]) >= need and (now - st["fired_at"]) > 300:
                st["fired_at"] = now
                out.append(self._px({
                    "agent": "security",
                    "type": "Unattended object left behind",
                    "severity": 3,
                    "confidence": 0.7,
                    "extra": {"object": key, "unattended_seconds": int(now - st["since"])},
                }, cx, cy, frame_w, self._fh))

        for k in [k for k, v in self._objects.items() if now - v["last"] > 15]:
            del self._objects[k]
        return out

    # -- snatch and run ----------------------------------------------------
    def _snatch(self, live, now):
        """Flag the grab-and-run pattern: two people come into contact, then
        one leaves at running speed while the other does not.

        This is an INDICATOR, not proof of theft — a greeting followed by
        someone sprinting for a bus looks similar. It is severity 4 so an
        operator always reviews it before anything happens.
        """
        cfg = self.cfg["snatch"]
        contact = float(cfg["contact"])
        min_contact = int(cfg["contact_samples"])
        flee = float(cfg["flee_speed"])
        window = float(cfg["window"])
        sep = float(cfg["separation"])
        out = []

        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                (ax, ay), (bx, by) = a.center(), b.center()
                avg_h = max(1.0, (a.height() + b.height()) / 2.0)
                gap = math.hypot(ax - bx, ay - by) / avg_h
                key = (min(a.tid, b.tid), max(a.tid, b.tid))
                st = self._contact_state.get(key)

                if gap < contact:
                    if st is None:
                        st = self._contact_state[key] = {"at": now, "count": 0,
                                                         "fired_at": 0.0}
                    st["at"] = now
                    st["count"] += 1
                    continue

                # Contact must have been held, not a fly-past: someone running
                # past a stationary person is not a snatch.
                if st is None or st["count"] < min_contact \
                        or (now - st["at"]) > window or gap < sep:
                    continue

                # recent speed, not the history average: the flight is a burst
                fast, slow = (a, b) if a.recent_speed() > b.recent_speed() else (b, a)
                if fast.recent_speed() > flee and slow.recent_speed() < flee * 0.6 \
                        and (now - st["fired_at"]) > 120:
                    st["fired_at"] = now
                    fx2, fy2 = fast.center()
                    out.append(self._px({
                        "agent": "security",
                        "type": "Possible snatch theft (contact then flight)",
                        "severity": 4,
                        "confidence": 0.6,
                        "extra": {"fleeing_track": fast.tid, "other_track": slow.tid},
                    }, fx2, fy2, self._fw, self._fh))

        for k in [k for k, v in self._contact_state.items()
                  if now - v["at"] > window * 4]:
            del self._contact_state[k]
        return out

    # -- zones: intrusion + loitering --------------------------------------
    def _zones(self, live, now, w, h):
        out = []
        loiter_need = float(self.cfg["loitering"]["seconds"])
        for z in self.zones:
            if not z.active(now):
                continue
            for tr in live:
                fx, fy = tr.feet()
                inside = z.contains(fx / w, fy / h)
                if z.kind == "intrusion" and self.cfg["intrusion"]["enabled"]:
                    if inside and (now - tr.zone_fired.get(z.name, 0.0)) > 300:
                        tr.zone_fired[z.name] = now
                        out.append(self._px({
                            "agent": "security",
                            "type": "Intrusion — " + z.name,
                            "severity": 4 if z.active_hours else 3,
                            "confidence": 0.85,
                            "extra": {"zone": z.name, "track": tr.tid},
                        }, fx, fy, w, h))
                elif z.kind == "loiter" and self.cfg["loitering"]["enabled"]:
                    if inside:
                        entered = tr.zone_enter.setdefault(z.name, now)
                        if (now - entered) >= loiter_need and \
                                (now - tr.zone_fired.get(z.name, 0.0)) > 300:
                            tr.zone_fired[z.name] = now
                            out.append(self._px({
                                "agent": "security",
                                "type": "Loitering — " + z.name,
                                "severity": 2,
                                "confidence": 0.75,
                                "extra": {"zone": z.name, "track": tr.tid,
                                          "seconds": int(now - entered),
                                          "lead_seconds": 3.0},
                            }, fx, fy, w, h))
                    else:
                        tr.zone_enter.pop(z.name, None)
        return out
