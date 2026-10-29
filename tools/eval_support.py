"""Local evaluation timing and provenance; no inference or downloads."""
import hashlib
import json
import math


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sidecar(path, kind):
    return path.with_suffix(f'.{kind}.json')


def frame_time(metadata, index, fps):
    times = metadata.get('frame_times_seconds')
    return float(times[index]) if times is not None else index / fps


def clip_metadata(path, fps, total):
    result = {'timeline_source': 'encoded_video', 'frame_times_seconds': None,
              'onset_seconds': None, 'onset_frame': None, 'confirmed_down_seconds': None,
              'onset_interval_seconds': None}
    capture_path = sidecar(path, 'capture')
    if capture_path.exists():
        capture = json.loads(capture_path.read_text(encoding='utf-8'))
        if capture.get('status') != 'valid':
            raise ValueError('capture metadata marks this clip invalid')
        if capture.get('video_sha256') != sha256(path):
            raise ValueError('capture metadata video hash mismatch')
        times = capture['frame_times_seconds']
        if len(times) != total or not times:
            raise ValueError('capture timestamp count does not match video frames')
        if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in times):
            raise ValueError('capture timestamps must be finite and nonnegative')
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('capture timestamps must be strictly increasing')
        result.update(timeline_source='capture_receipt_time', frame_times_seconds=times,
                      capture_sha256=sha256(capture_path),
                      capture_max_gap_seconds=capture.get('max_gap_seconds'))
    label_path = sidecar(path, 'labels')
    if label_path.exists():
        labels = json.loads(label_path.read_text(encoding='utf-8'))
        if labels.get('video_sha256') != sha256(path):
            raise ValueError('onset annotation video hash mismatch')
        if 'onset_interval_frames' in labels:
            interval = labels['onset_interval_frames']
            if (not isinstance(interval, list) or len(interval) != 2
                    or any(type(index) is not int for index in interval)
                    or not 0 <= interval[0] < interval[1] < total):
                raise ValueError('onset interval is outside the video or not increasing')
            times = [frame_time(result, index, fps) for index in interval]
            result.update(onset_interval_seconds=times, confirmed_down_seconds=times[1],
                          labels_sha256=sha256(label_path), annotation_method=labels.get('method'))
        else:
            index = labels['onset_frame']
            if type(index) is not int or not 0 <= index < total:
                raise ValueError('onset frame is outside the video')
            result.update(onset_frame=index, onset_seconds=frame_time(result, index, fps),
                          confirmed_down_seconds=frame_time(result, index, fps),
                          labels_sha256=sha256(label_path))
    return result


def latency(alert, onset):
    return None if alert is None or onset is None or alert < onset else alert - onset
