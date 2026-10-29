#!/usr/bin/env python3
"""Label a reviewed positive clip's person-down onset without inference."""
import argparse
import json
import math
from pathlib import Path
import cv2
from eval_support import clip_metadata, frame_time, sha256, sidecar


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('clip', type=Path)
    ap.add_argument('--onset-video-seconds', type=float, required=True,
                    help='reviewed video-player timestamp at first frame of the defined person-down state')
    args = ap.parse_args()
    if args.clip.parent.name != 'positive':
        ap.error('Onset labels belong to positive clips only')
    label = sidecar(args.clip, 'labels')
    if label.exists():
        ap.error('Annotation already exists; review it explicitly rather than overwriting it')
    cap = cv2.VideoCapture(str(args.clip))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count < 1:
            ap.error('Clip has invalid video metadata')
    finally:
        cap.release()
    seconds = args.onset_video_seconds
    if not math.isfinite(seconds) or not 0 <= seconds < count / fps:
        ap.error('Onset timestamp is outside the video')
    index = min(count - 1, int(seconds * fps))
    meta = clip_metadata(args.clip, fps, count)
    labels = {'schema_version': 1, 'video_sha256': sha256(args.clip), 'onset_frame': index,
              'onset_video_seconds': seconds, 'onset_seconds': frame_time(meta, index, fps),
              'definition': 'First frame with torso supported on the floor/mat after the staged descent; human reviewed.'}
    with label.open('x', encoding='utf-8') as stream:
        json.dump(labels, stream, indent=2)
    print(f'Saved onset annotation: {label}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
