"""Offline evidence/SOP-grounded drafts with immutable human review records.

No external model/API or dispatch is called. Demo procedures never authorize a
production recommendation. Event approval remains the existing separate action.
"""
import hashlib
import json
from pathlib import Path
import time
from licensing import module_for_event


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class TriageError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class TriageAssistant:
    def __init__(self, conn, lock, media_dir, evidence, catalog_path=None):
        self.conn, self.lock = conn, lock
        self.media_dir, self.evidence = Path(media_dir), evidence
        self.catalog_path = Path(catalog_path) if catalog_path else Path(__file__).with_name('triage_demo_sops.json')
        with self.lock:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS triage_drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL,
                created_at REAL NOT NULL, created_by TEXT NOT NULL, content TEXT NOT NULL,
                event_fingerprint TEXT NOT NULL, procedure_fingerprint TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'pending', reviewed_at REAL,
                reviewed_by TEXT, review_note TEXT DEFAULT '')''')
            self.conn.execute('CREATE INDEX IF NOT EXISTS triage_event ON triage_drafts(event_id,id)')
            self.conn.commit()

    def procedures(self):
        try:
            rows = json.loads(self.catalog_path.read_text(encoding='utf-8'))['procedures']
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TriageError(503, 'Procedure catalog is unavailable or invalid') from exc
        if not isinstance(rows, list):
            raise TriageError(503, 'Procedure catalog is invalid')
        for row in rows:
            if (not isinstance(row, dict) or row.get('status') not in ('demonstration', 'approved') or not row.get('id')
                    or not row.get('version') or not isinstance(row.get('steps'), list)):
                raise TriageError(503, 'Procedure catalog is invalid')
            if not all(isinstance(step, str) for step in row['steps']):
                raise TriageError(503, 'Procedure steps must be text')
            if row['status'] == 'approved' and not (row.get('approved_by') and row.get('approved_at')):
                raise TriageError(503, 'Approved procedure lacks approval provenance')
        return rows

    def _event(self, event_id):
        with self.lock:
            row = self.conn.execute('SELECT * FROM events WHERE id=?', (event_id,)).fetchone()
        if row is None:
            raise TriageError(404, 'Incident not found')
        event = dict(row)
        try:
            event['extra'] = json.loads(event.get('extra') or '{}')
        except (ValueError, TypeError):
            event['extra'] = {}
        return event

    def _media(self, event):
        result = {}
        for label, extension, flag in (('snapshot', 'jpg', 'has_snapshot'), ('clip', 'gif', 'has_clip')):
            path = self.media_dir / f"{int(event['id'])}.{extension}"
            exists = bool(event.get(flag)) and path.is_file()
            result[label] = {'available': exists,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if exists else None,
                             'reference': f"/api/events/{event['id']}/{label}" if exists else None}
        return result

    def _fingerprint(self, event, media):
        return digest({'event': event, 'media': media})

    def _procedure(self, event):
        module = module_for_event(event['type'], event['agent'])
        rows = [p for p in self.procedures() if p['module'] == module]
        return next((p for p in rows if p['status'] == 'approved'), rows[0] if rows else None)

    def create(self, event_id, actor):
        event = self._event(event_id)
        media = self._media(event)
        procedure = self._procedure(event)
        signed = self.evidence.latest(event_id)
        verification = self.evidence.verify(event_id) if signed else {'ok': False, 'reason': 'not configured or no signed manifest'}
        if signed and not verification.get('ok'):
            raise TriageError(409, 'Evidence verification failed; inspect the incident before drafting')
        extra = event['extra'] if isinstance(event['extra'], dict) else {}
        context = extra.get('transition_context') or {}
        if not isinstance(context, dict):
            context = {}
        transition = context.get('transition') or {}
        if not isinstance(transition, dict):
            transition = {}
        content = {
            'assistant': 'AICOS Operator Triage Assistant', 'mode': 'offline',
            'event_id': event_id, 'production_recommendations_blocked': not procedure or procedure['status'] != 'approved',
            'facts': {'source_id': event['source_id'], 'event_time': event['ts'],
                      'detector_report': event['type'], 'severity': event['severity'],
                      'model_score': event['confidence'], 'incident_status': event['status'],
                      'location_status': ('synthetic demonstration point' if extra.get('is_replay') and
                                          'synthetic' in str(extra.get('location_source', '')).lower() else
                                          'configured' if event.get('lat') or event.get('lon') else 'unconfigured'),
                      'transition_observation': transition.get('kind', 'not available'),
                      'medical_cause': 'unknown', 'consciousness': 'not established'},
            'media': media, 'signed_evidence': verification,
            'procedure': procedure,
            'recommendations': list(procedure['steps']) if procedure and procedure['status'] == 'approved' else [],
            'demonstration_steps': list(procedure['steps']) if procedure and procedure['status'] == 'demonstration' else [],
            'missing_evidence': [kind for kind, value in media.items() if not value['available']],
            'source_references': [{'id': f'event:{event_id}', 'reference': '/api/events', 'kind': 'incident_metadata'},
                                  {'id': f'evidence:{event_id}', 'reference': f'/api/events/{event_id}/evidence',
                                   'available': bool(signed), 'kind': 'signed_evidence'}],
            'limits': ['Detector reports are not independently verified facts.',
                       'No medical cause or intent is established.',
                       'Reviewing this draft does not approve an incident or dispatch anyone.']}
        if procedure:
            content['source_references'].append({'id': procedure['id'] + '@' + procedure['version'],
                                                'sha256': digest(procedure), 'kind': 'operating_procedure',
                                                'status': procedure['status']})
        now = time.time()
        with self.lock:
            cur = self.conn.execute('''INSERT INTO triage_drafts
                (event_id,created_at,created_by,content,event_fingerprint,procedure_fingerprint)
                VALUES(?,?,?,?,?,?)''', (event_id, now, actor, canonical(content),
                                        self._fingerprint(event, media), digest(procedure)))
            draft_id = cur.lastrowid
            self.conn.execute('INSERT INTO audit(event_id,ts,action,note) VALUES(?,?,?,?)',
                              (event_id, now, 'triage_drafted', f'draft {draft_id} created by {actor}'))
            self.conn.commit()
        self.evidence.sign_event(event_id)
        return self.get(draft_id)

    def get(self, draft_id):
        with self.lock:
            row = self.conn.execute('SELECT * FROM triage_drafts WHERE id=?', (draft_id,)).fetchone()
        if row is None:
            raise TriageError(404, 'Triage draft not found')
        result = dict(row)
        result['content'] = json.loads(result['content'])
        result.pop('event_fingerprint', None)
        result.pop('procedure_fingerprint', None)
        return result

    def list(self, event_id):
        self._event(event_id)
        with self.lock:
            ids = [r['id'] for r in self.conn.execute('SELECT id FROM triage_drafts WHERE event_id=? ORDER BY id DESC LIMIT 50', (event_id,))]
        return [self.get(i) for i in ids]

    def review(self, draft_id, decision, actor, note=''):
        if decision not in ('accept', 'reject'):
            raise TriageError(422, 'Review must accept or reject the draft')
        with self.lock:
            row = self.conn.execute('SELECT * FROM triage_drafts WHERE id=?', (draft_id,)).fetchone()
            if row is None:
                raise TriageError(404, 'Triage draft not found')
            if row['review_status'] != 'pending':
                raise TriageError(409, 'Draft already reviewed; create a new draft for another review')
            event = self._event(row['event_id'])
            procedure = self._procedure(event)
            if (row['event_fingerprint'] != self._fingerprint(event, self._media(event)) or
                    row['procedure_fingerprint'] != digest(procedure)):
                raise TriageError(409, 'Incident, evidence or procedure changed; generate a fresh draft')
            signed = self.evidence.latest(event['id'])
            if signed and not self.evidence.verify(event['id']).get('ok'):
                raise TriageError(409, 'Evidence verification failed')
            demo = json.loads(row['content'])['production_recommendations_blocked']
            status = 'rejected' if decision == 'reject' else 'accepted_for_demo' if demo else 'accepted'
            now = time.time()
            self.conn.execute('UPDATE triage_drafts SET review_status=?,reviewed_at=?,reviewed_by=?,review_note=? WHERE id=?',
                              (status, now, actor, note[:500], draft_id))
            self.conn.execute('INSERT INTO audit(event_id,ts,action,note) VALUES(?,?,?,?)',
                              (event['id'], now, 'triage_' + status, f'draft {draft_id} reviewed by {actor}: {note[:500]}'))
            self.conn.commit()
        self.evidence.sign_event(event['id'])
        return self.get(draft_id)
