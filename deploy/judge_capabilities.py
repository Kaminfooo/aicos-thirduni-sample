"""Public metadata for the isolated judge sandbox; no production enablement."""
CATALOG = [
    ('person_down', 'Person down', 'Reference candidate', 'Selected fall references pass; medical cause unknown.'),
    ('fire_smoke', 'Fire & smoke', 'Development candidate', 'Fire references pass; indoor smoke misses remain.'),
    ('crowd', 'Crowd occupancy', 'Site configuration', 'People count and dwell; does not infer danger or intent.'),
    ('intrusion', 'Zone entry', 'Site configuration', 'Presence inside your test rectangle; not authorization recognition.'),
    ('loitering', 'Zone dwell', 'Site configuration', 'Requires tracking continuity and your configured dwell time.'),
    ('vehicles', 'Traffic', 'Site configuration', 'Stopped vehicles, congestion and optional direction; not crash recognition.'),
    ('abandoned_object', 'Unattended object', 'Research preview', 'Stationary bag and nearby-person cues; actual ownership is not established.'),
    ('violence', 'Violence cues', 'Research preview', 'Possible violence cues for review; reliable recognition across scenes is not yet established.'),
    ('fight', 'Altercation movement', 'Research preview', 'Pose and movement cues for review; ordinary interactions can also trigger alerts.'),
    ('panic', 'Running group', 'Research preview', 'Group running pattern; does not establish panic or intent.'),
    ('snatch', 'Contact & flight', 'Research preview', 'Contact followed by rapid separation; does not establish theft or identity.'),
    ('motion', 'Image motion', 'Demonstration only', 'Pixel changes; not semantic incident recognition.'),
]
IDS = tuple(row[0] for row in CATALOG)


def catalog():
    return [dict(id=i, title=t, status=s, limitation=l) for i,t,s,l in CATALOG]
