"""
AICOS Violence Model Detector
-----------------------------
Runs a trained violence/fight recognition model alongside the motion
heuristics in detectors.py, through the same Detector interface.

Why this exists separately: the heuristics read geometry (who is near
whom, moving how). A trained model reads appearance and action. They
fail differently, which is exactly why running both is stronger than
either alone.

Ships configured for YOLO-format fight models (e.g. the community
`fight_detection_yolov8` weights), because the edge worker already has
ultralytics installed — no new dependency. Any model exposing a
`.predict()` returning boxes with class ids will work.

Two safeguards are built in and should not be removed:

  persistence   a single positive frame never raises an alert. Violence
                lasts seconds; a one-frame spike is model noise.
  severity 3    model output is a recommendation an operator confirms,
                never an autonomous dispatch trigger.

See docs/VIOLENCE-MODEL-GUIDE.md for where to get weights, how to
fine-tune on your own footage, and how to validate honestly.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger("aicos.violence")

DEFAULTS = {
    "enabled": False,
    # Downloaded once from Hugging Face and cached, exactly like the fall
    # model. Set `weights` to a local path to use your own instead.
    "repo": "Musawer14/fight_detection_yolov8",
    "filename": "",   # blank = discover it from the repo
    "weights": "",                   # local path; overrides the download
    "violence_class": 1,             # class id meaning "fight/violence"
    "conf": 0.5,                     # per-frame confidence floor
    "imgsz": 640,
    "stride": 2,                     # analyze 1 of every N frames offered
    "persistence": 3,                # consecutive positives before alerting
    "cooldown_seconds": 60.0,
    "severity": 3,
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in (override or {}).items():
        out[key] = val
    return out


class YoloViolenceModel:
    """Loads a YOLO-format violence model. Kept separate from the person
    engine so the two models never share configuration."""

    def __init__(self, weights: str, conf: float, imgsz: int):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install -r requirements-yolo.txt"
            ) from exc
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
        self.conf = conf
        self.imgsz = imgsz

    def best_score(self, frame, class_id: int) -> float:
        """Highest confidence for the violence class in this frame, 0.0 if none."""
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 verbose=False)[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return 0.0
        best = 0.0
        for i in range(len(boxes)):
            try:
                cls_id = int(boxes.cls[i].cpu().numpy())
                score = float(boxes.conf[i].cpu().numpy())
            except Exception:
                continue
            if cls_id == class_id and score > best:
                best = score
        return best


class ViolenceDetector:
    """One instance per camera — it holds per-camera streak state."""

    name = "violence-model"

    def __init__(self, config: dict = None, model=None):
        self.cfg = _merge(DEFAULTS, config or {})
        self._model = model          # injected in tests; lazily loaded otherwise
        self._calls = 0
        self._streak = 0
        self._peak = 0.0
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
            path = self._resolve_weights()
            self._model = YoloViolenceModel(
                path, float(self.cfg["conf"]), int(self.cfg["imgsz"]))
            names = getattr(self._model.model, "names", {}) or {}
            log.info("violence model loaded: %s, classes %s — "
                     "set violence_class to the id that means a fight",
                     path, dict(names))
        return self._model

    def detect(self, frame, prev_frame=None, now: float = None):
        if self._disabled:
            return []
        now = time.time() if now is None else now
        stride = max(1, int(self.cfg["stride"]))
        self._calls += 1
        if self._calls % stride:
            return []

        try:
            score = self._get_model().best_score(frame, int(self.cfg["violence_class"]))
        except Exception as exc:
            self._disabled = True
            log.error("violence detector DISABLED — could not load or run the "
                      "model: %s", exc)
            log.error("  fix the `violence.repo` setting, or set "
                      "`violence.enabled: false`. Every other detector "
                      "continues normally.")
            return []

        if score <= 0.0:
            self._streak = 0
            self._peak = 0.0
            return []

        self._streak += 1
        self._peak = max(self._peak, score)

        need = max(1, int(self.cfg["persistence"]))
        cooldown = float(self.cfg["cooldown_seconds"])
        if self._streak < need or (now - self._fired_at) < cooldown:
            return []

        self._fired_at = now
        peak = self._peak
        self._streak = 0
        self._peak = 0.0
        return [{
            "agent": "security",
            "type": "Possible violence detected (model)",
            "severity": int(self.cfg["severity"]),
            "confidence": round(min(0.95, peak), 2),
            "extra": {"model": str(self.cfg["weights"]), "consecutive_frames": need},
        }]
