"""
AICOS Trained Fall Detector
---------------------------
Runs a model that was TRAINED on fallen people, instead of inferring a fall
from box geometry.

Why this exists. The heuristics in detectors.py reason about shape and
stillness: is the box wide, has the person shrunk, have they stopped
moving. That works, but it is indirect, and it breaks in the ways we found
on real footage — a body lying along the camera's line of sight stays
narrow and tall, and a general-purpose pose model loses its lock on a prone
person entirely. A model trained on thousands of images of fallen people
does not have those blind spots, because "fallen" is a class it learned
rather than a shape we guessed at.

Default weights: melihuzunoglu/human-fall-detection (YOLOv11, three
classes — Fallen, Sitting, Standing). Downloaded once from Hugging Face,
then cached locally; a `weights:` path overrides it with your own model.

Two safeguards are deliberate and should stay:

  seconds     a single positive frame never raises an alert. Someone
              crouching is briefly "fallen" to any detector.
  severity 4  model output is a recommendation an operator confirms,
              never an autonomous dispatch trigger.

Run it ALONGSIDE the geometric detector, not instead of it: the two fail
differently, and an incident both agree on is a strong signal.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("aicos.falldet")

DEFAULTS = {
    "enabled": False,
    "repo": "melihuzunoglu/human-fall-detection",
    "filename": "",   # blank = discover it from the repo
    "weights": "",            # local path; overrides the download
    "conf": 0.35,
    "imgsz": 640,
    "seconds": 3.0,           # a fall must persist this long
    "cooldown_seconds": 90.0,
    "severity": 4,
    # class names that mean "on the ground", matched case-insensitively
    "fallen_labels": ["fallen", "fall", "falling", "fall_detected",
                      "person_fallen", "lying"],
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in (override or {}).items():
        out[key] = val
    return out


class FallModel:
    """Wraps the trained detector. Returns (score, box) for the best
    'fallen' detection in a frame, or (0.0, None)."""

    def __init__(self, weights: str, conf: float, imgsz: int, labels):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install -r "
                "requirements-yolo.txt") from exc
        try:
            import torch
            nnp = getattr(torch.backends, "nnpack", None)
            if nnp is not None:
                try:
                    nnp.enabled = False
                except Exception:
                    pass
        except Exception:
            pass
        self.model = YOLO(weights)
        self._predict_lock = threading.Lock()
        self.conf = conf
        self.imgsz = imgsz
        self.wanted = {str(x).strip().lower() for x in labels}
        names = getattr(self.model, "names", {}) or {}
        self.names = {int(k): str(v) for k, v in names.items()}
        matched = [v for v in self.names.values() if v.strip().lower() in self.wanted]
        if not matched:
            log.warning("none of the configured fallen_labels match this "
                        "model's classes %s — set fallen_labels to one of them",
                        sorted(self.names.values()))
        else:
            log.info("fall model loaded, classes %s, treating %s as a fall",
                     sorted(self.names.values()), matched)

    def best_fall(self, frame):
        with self._predict_lock:
            res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                     verbose=False)[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return 0.0, None
        best_score, best_box = 0.0, None
        for i in range(len(boxes)):
            try:
                cls_id = int(boxes.cls[i].cpu().numpy())
                score = float(boxes.conf[i].cpu().numpy())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            except Exception:
                continue
            label = self.names.get(cls_id, str(cls_id)).strip().lower()
            if label in self.wanted and score > best_score:
                best_score, best_box = score, tuple(xyxy)
        return best_score, best_box


_FALL_MODEL_CACHE = {}
_FALL_MODEL_CACHE_LOCK = threading.Lock()


def _shared_fall_model(weights: str, conf: float, imgsz: int, labels):
    key = (str(weights), float(conf), int(imgsz), tuple(sorted(str(x).lower() for x in labels)))
    with _FALL_MODEL_CACHE_LOCK:
        model = _FALL_MODEL_CACHE.get(key)
        if model is None:
            model = FallModel(weights, conf, imgsz, labels)
            _FALL_MODEL_CACHE[key] = model
        return model


class FallModelDetector:
    """One instance per camera — it holds that camera's persistence state."""

    name = "fall-model"

    def __init__(self, config: dict = None, model=None):
        self.cfg = _merge(DEFAULTS, config or {})
        self._model = model              # injected in tests
        self._since = None
        self._peak = 0.0
        self._box = None
        self._fired_at = 0.0
        self._disabled = False

    def _resolve_weights(self) -> str:
        weights = str(self.cfg.get("weights") or "").strip()
        if weights:
            return weights
        from huggingface_hub import hf_hub_download, list_repo_files
        repo = str(self.cfg["repo"])
        name = str(self.cfg.get("filename") or "").strip()
        if name:
            return hf_hub_download(repo_id=repo, filename=name)
        # Model authors name their weights whatever they like — best.pt,
        # yolov8n.pt, model.pt. Ask the repo instead of guessing.
        files = [f for f in list_repo_files(repo) if f.lower().endswith(".pt")]
        if not files:
            raise RuntimeError(f"no .pt weights found in {repo}")
        files.sort(key=lambda f: (f.count("/"), len(f)))
        log.info("weights in %s: %s — using %s", repo, files, files[0])
        return hf_hub_download(repo_id=repo, filename=files[0])

    def _get_model(self):
        if self._model is None:
            self._model = _shared_fall_model(self._resolve_weights(),
                                            float(self.cfg["conf"]),
                                            int(self.cfg["imgsz"]),
                                            self.cfg["fallen_labels"])
        return self._model

    def detect(self, frame, prev_frame=None, now: float = None):
        if self._disabled:
            return []
        now = time.time() if now is None else now
        try:
            score, box = self._get_model().best_fall(frame)
        except Exception as exc:
            self._disabled = True
            log.error("fall model detector DISABLED — could not load or run "
                      "the model: %s", exc)
            log.error("  fix the `fall_model.repo` setting, or set "
                      "`fall_model.enabled: false`. Every other detector "
                      "continues normally.")
            return []

        if score <= 0.0:
            self._since = None
            self._peak = 0.0
            self._box = None
            return []

        if self._since is None:
            self._since = now
        self._peak = max(self._peak, score)
        self._box = box or self._box

        held = now - self._since
        need = float(self.cfg["seconds"])
        cooldown = float(self.cfg["cooldown_seconds"])
        if held < need or (now - self._fired_at) < cooldown:
            return []

        self._fired_at = now
        event = {
            "agent": "emergency",
            "type": "Person down / possible medical emergency",
            "severity": int(self.cfg["severity"]),
            "confidence": round(min(0.95, self._peak), 2),
            "extra": {"detector": "trained-model",
                      "down_seconds": int(held),
                      "lead_seconds": round(held, 1)},
        }
        if self._box:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self._box
            event["extra"]["px"] = [
                round(max(0.0, min(1.0, ((x1 + x2) / 2.0) / max(1.0, w))), 4),
                round(max(0.0, min(1.0, y2 / max(1.0, h))), 4),
            ]
        self._since = now          # keep the clock running while they stay down
        self._peak = 0.0
        return [event]
