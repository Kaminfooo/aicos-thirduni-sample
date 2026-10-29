"""
AICOS Violence Model Probe
--------------------------
Measures how a PRETRAINED violence/crime classifier performs on YOUR
footage, before you spend a single GPU-hour training anything.

It slides a fixed-length window across a video, classifies each window,
and prints a summary: which labels fired, how often, and at what
confidence. Point it at ordinary street footage and every "Fighting"
prediction is a false positive — that count is your false-alarm rate.

Install (edge machine or any PC; a GPU is optional but much faster):
    pip install transformers torch torchvision av

Usage:
    python violence_probe.py /root/videos/street2.mp4
    python violence_probe.py clip.mp4 --model Nikeytas/videomae-crime-detector-production-v1
    python violence_probe.py clip.mp4 --threshold 0.5 --stride 8 --max-windows 40

First run downloads the model (VideoMAE-large is ~1.2 GB, base ~400 MB).
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

DEFAULT_MODEL = "OPear/videomae-large-finetuned-UCF-Crime"
NORMAL_HINTS = ("normal", "non-violent", "nonviolent", "no_fight", "nofight")


def is_normal_label(label: str) -> bool:
    """True when a label means 'nothing happening'. Model authors spell
    this differently ('Normal', 'Normal Videos', 'Non-Violent Incident'),
    so match on substrings rather than an exact list."""
    low = str(label).strip().lower()
    return any(hint in low for hint in NORMAL_HINTS)


def sample_indices(total: int, clip_len: int, stride: int) -> list:
    """Start frames for each window, so windows never run past the end.

    A video shorter than one window yields no windows — the caller
    reports that rather than silently analyzing a partial clip.
    """
    if total < clip_len or clip_len <= 0 or stride <= 0:
        return []
    return list(range(0, total - clip_len + 1, stride))


def read_window(cap, start: int, clip_len: int, size: int = 224) -> list:
    """Read clip_len consecutive frames starting at `start`, RGB, resized."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(clip_len):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, (size, size)),
                                   cv2.COLOR_BGR2RGB))
    return frames


def summarize(hits: list, windows: int, threshold: float, elapsed: float) -> None:
    print("-" * 62)
    print(f"windows analyzed: {windows}   time: {elapsed:.1f}s")
    if not windows:
        print("VERDICT: video too short for one window — use a longer clip")
        return
    flagged = [h for h in hits if not is_normal_label(h["label"])
               and h["score"] >= threshold]
    counts = {}
    for h in flagged:
        counts[h["label"]] = counts.get(h["label"], 0) + 1
    print(f"windows flagged as non-normal (>= {threshold:.2f}): "
          f"{len(flagged)} / {windows} ({100.0 * len(flagged) / windows:.1f}%)")
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        best = max(h["score"] for h in flagged if h["label"] == label)
        print(f"    {n:4d}x  {label}   (max confidence {best:.2f})")
    print("-" * 62)
    if not flagged:
        print("VERDICT: model stayed quiet on this footage.")
        print("  On ordinary street video that is the RIGHT answer.")
        print("  On fight footage it means the model missed it — check recall.")
    else:
        print("VERDICT: model raised alerts on this footage.")
        print("  On ordinary street video these are FALSE POSITIVES — the")
        print("  percentage above is your false-alarm rate at this threshold.")
        print("  On fight footage this is a hit; check WHERE it fired.")
    print("  Raise --threshold to cut false alarms (and lose real detections).")


def remap_videomae_biases(state_dict, zeros_like):
    """Repair the attention-bias layout of older VideoMAE checkpoints.

    The original VideoMAE stores `q_bias` and `v_bias` per attention layer
    and keeps the key bias fixed at zero. Newer transformers versions
    expect `query.bias`, `key.bias` and `value.bias`. Loaded as-is, every
    attention layer gets RANDOMLY INITIALIZED biases and the model's
    predictions become meaningless — while still reporting high
    confidence, which is the dangerous part.

    Returns (new_state_dict, number_of_layers_repaired).
    """
    out = {}
    repaired = 0
    for key, value in state_dict.items():
        if key.endswith("attention.attention.q_bias"):
            out[key[: -len("q_bias")] + "query.bias"] = value
            repaired += 1
        elif key.endswith("attention.attention.v_bias"):
            base = key[: -len("v_bias")]
            out[base + "value.bias"] = value
            out[base + "key.bias"] = zeros_like(value)   # zero by design
        else:
            out[key] = value
    return out, repaired


def load_model(model_id, torch, VideoMAEForVideoClassification):
    """Load the classifier, repairing the bias layout if the checkpoint
    uses the legacy naming."""
    model = VideoMAEForVideoClassification.from_pretrained(model_id)
    try:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(model_id, "model.safetensors")
            from safetensors.torch import load_file
            raw = load_file(path)
        except Exception:
            path = hf_hub_download(model_id, "pytorch_model.bin")
            raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"  (could not re-read checkpoint for bias repair: {exc})")
        return model.eval()

    if not any(k.endswith("q_bias") for k in raw):
        return model.eval()          # modern checkpoint, nothing to repair

    fixed, repaired = remap_videomae_biases(raw, torch.zeros_like)
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    print(f"  repaired legacy attention biases in {repaired} layers "
          f"({len(missing)} still missing)")
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe a pretrained violence model")
    ap.add_argument("video", help="path to a video file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--clip-len", type=int, default=16,
                    help="frames per window (16 suits VideoMAE)")
    ap.add_argument("--stride", type=int, default=16,
                    help="frames between window starts")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-windows", type=int, default=40,
                    help="stop after this many windows (CPU inference is slow)")
    ap.add_argument("--only-labels", default="",
                    help="comma-separated labels to count as alerts, e.g. "
                         "Fighting,Assault,Abuse,Robbery. Others are treated "
                         "as normal — useful when a model has classes you "
                         "do not care about (RoadAccidents, Shoplifting...)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open video: " + args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    starts = sample_indices(total, args.clip_len, args.stride)[:args.max_windows]
    print(f"video: {total} frames @ {fps:.1f} fps -> {len(starts)} window(s) "
          f"of {args.clip_len} frames")
    if not starts:
        cap.release()
        summarize([], 0, args.threshold, 0.0)
        return

    try:
        import torch
        from transformers import (VideoMAEForVideoClassification,
                                  VideoMAEImageProcessor)
    except ImportError:
        cap.release()
        raise SystemExit("missing deps — run: pip install transformers torch "
                         "torchvision av")

    print(f"loading {args.model} (first run downloads the weights)...")
    proc = VideoMAEImageProcessor.from_pretrained(args.model)
    model = load_model(args.model, torch, VideoMAEForVideoClassification)
    labels = model.config.id2label
    print(f"model labels: {', '.join(str(v) for v in labels.values())}")
    only = {x.strip() for x in args.only_labels.split(",") if x.strip()}
    if only:
        unknown = only - {str(v) for v in labels.values()}
        if unknown:
            print(f"  WARNING: not in this model: {', '.join(sorted(unknown))}")
        print(f"  counting only: {', '.join(sorted(only))}")

    hits = []
    t0 = time.time()
    for n, start in enumerate(starts, 1):
        frames = read_window(cap, start, args.clip_len)
        if len(frames) < args.clip_len:
            break                     # ran off the end of a truncated file
        inputs = proc(frames, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)[0]
        idx = int(probs.argmax())
        label, score = str(labels.get(idx, idx)), float(probs[idx])
        if only and label not in only:
            # a class the operator does not care about — report the best
            # score among the labels that DO matter instead
            best_i, best_p = None, 0.0
            for i, p in enumerate(probs):
                if str(labels.get(i, i)) in only and float(p) > best_p:
                    best_i, best_p = i, float(p)
            if best_i is not None and best_p >= args.threshold:
                label, score = str(labels.get(best_i, best_i)), best_p
            else:
                label, score = "Normal_Videos_event", 1.0 - best_p
        hits.append({"start": start, "label": label, "score": score})
        secs = start / fps if fps else 0.0
        mark = "" if is_normal_label(label) else "   <-- ALERT"
        print(f"  window {n:3d} @ {secs:6.1f}s : {label:<22} {score:.2f}{mark}")
        sys.stdout.flush()
    cap.release()
    summarize(hits, len(hits), args.threshold, time.time() - t0)


if __name__ == "__main__":
    main()
