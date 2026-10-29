"""One explicit profile factory for offline replays and the local camera runner."""
from pathlib import Path


def build_profile(profile, camera=None):
    result = []
    definitions = (
        ('yolo', 'model', 'detectors', 'YoloBehaviorSuite'),
        ('fall', 'weights', 'falldet', 'FallModelDetector'),
        ('confirmed', 'weights', 'confirmed_person_down', 'ConfirmedPersonDownDetector'),
        ('pose_confirmed', 'model', 'pose_confirmed_person_down', 'PoseConfirmedPersonDownDetector'),
        ('site_behaviors', 'model', 'site_behaviors', 'SiteBehaviorSuite'),
        ('fire_smoke', 'weights', 'persistent_hazards', 'PersistentHazardDetector'),
        ('violence', 'weights', 'persistent_hazards', 'PersistentHazardDetector'))
    for key, weight_key, module, class_name in definitions:
        cfg = profile.get(key)
        if not cfg or not cfg.get('enabled', True):
            continue
        if not Path(cfg.get(weight_key, '')).is_file():
            raise ValueError(key + ' requires existing local weights')
        import importlib
        implementation = getattr(importlib.import_module(module), class_name)
        options = {**cfg, 'kind': key} if key in ('fire_smoke', 'violence') else cfg
        if key in ('yolo', 'site_behaviors'):
            result.append(implementation(camera=camera or {}, config=options))
        else:
            result.append(implementation(config=options))
    if not result:
        raise ValueError('Profile selects no supported detectors; no motion-demo fallback is added')
    return result
