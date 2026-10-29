"""Outbox durability limits — edge queue and integration bus.

These cover the two ways a durable queue turns into an outage of its own:
it grows without a ceiling, or it retries something that can never succeed.
"""
import os, sqlite3, sys, tempfile, time, types, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "edge"))
sys.path.insert(0, os.path.join(ROOT, "backend"))

# the edge worker imports requests; stub it so tests need no network
if "requests" not in sys.modules:
    fr = types.ModuleType("requests")
    class _S:
        def post(self, *a, **k): raise fr.RequestException("offline")
        def close(self): pass
    fr.Session = _S
    class _E(Exception): pass
    fr.RequestException = _E
    sys.modules["requests"] = fr

from rtsp_worker import BackendClient
from integrations import IntegrationBus

KEY = "device-key-abcdefghijklmnopqrstuvwxyz-1234"


def _event(i, kb=0):
    e = {"source_id": "C", "agent": "security", "type": "t", "severity": 3,
         "confidence": 0.9, "event_uuid": f"uuid-{i:06d}-aaaa-bbbb-cccc"}
    if kb:
        e["snapshot"] = "A" * (kb * 1024)
    return e


class EdgeOutboxLimits(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.environ["AICOS_OUTBOX_PATH"] = os.path.join(self.td, "o.db")

    def _client(self, rows=1000, mb=100):
        os.environ["AICOS_OUTBOX_MAX_ROWS"] = str(rows)
        os.environ["AICOS_OUTBOX_MAX_MB"] = str(mb)
        return BackendClient("http://127.0.0.1:9", KEY)

    def _count(self, c):
        with c._lock:
            r = c._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)),0) FROM outbox").fetchone()
        return int(r[0]), int(r[1])

    def test_row_cap_drops_oldest_and_keeps_newest(self):
        c = self._client(rows=10)
        c._send = lambda p: (False, "offline", False)
        for i in range(40):
            c.post_event(_event(i))
        n, _ = self._count(c)
        self.assertEqual(n, 10)
        with c._lock:
            kept = [r[0] for r in c._db.execute(
                "SELECT event_uuid FROM outbox ORDER BY id").fetchall()]
        # during an outage the NEWEST incidents are the ones still needed
        self.assertEqual(kept[-1], "uuid-000039-aaaa-bbbb-cccc")
        self.assertEqual(c.dropped_events, 30)
        c.close()

    def test_size_cap_is_respected(self):
        c = self._client(rows=99999, mb=2)
        c._send = lambda p: (False, "offline", False)
        for i in range(30):
            c.post_event(_event(i, kb=300))
        n, size = self._count(c)
        self.assertLessEqual(size, 2 * 1024 * 1024)
        self.assertEqual(n, 30)  # media is compacted before events are dropped
        self.assertGreater(c.compacted_events, 0)
        c.close()

    def test_permanent_rejection_is_quarantined_not_lost(self):
        c = self._client()
        c._send = lambda p: (False, "HTTP 401: invalid key", True)
        accepted = c.post_event(_event(1))
        self.assertFalse(accepted)
        self.assertEqual(self._count(c)[0], 0)
        with c._lock:
            dlq = c._db.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0]
        self.assertEqual(dlq, 1)
        c.close()

    def test_network_failure_is_still_queued(self):
        c = self._client()
        c._send = lambda p: (False, "connection refused", False)
        self.assertTrue(c.post_event(_event(1)))
        self.assertEqual(self._count(c)[0], 1)
        c.close()

    def test_queued_row_later_rejected_moves_to_dead_letter(self):
        c = self._client()
        c._send = lambda p: (False, "offline", False)
        c.post_event(_event(1))
        self.assertEqual(self._count(c)[0], 1)
        c._send = lambda p: (False, "HTTP 422: schema", True)
        deadline = time.time() + 8
        while time.time() < deadline and self._count(c)[0]:
            time.sleep(0.25)
        self.assertEqual(self._count(c)[0], 0)
        with c._lock:
            dlq = c._db.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0]
        self.assertEqual(dlq, 1)
        c.close()

    def test_unknown_send_shape_is_fail_safe_and_queued(self):
        c = self._client()
        c._send = lambda p: "error"
        self.assertTrue(c.post_event(_event(1)))
        self.assertEqual(self._count(c)[0], 1)
        c.close()

    def test_critical_events_survive_row_pressure(self):
        c = self._client(rows=10, mb=100)
        c._send = lambda p: (False, "offline", False)
        critical = _event(900); critical["severity"] = 5
        c.post_event(critical)
        for i in range(30):
            c.post_event(_event(i))
        with c._lock:
            uuids = [r[0] for r in c._db.execute("SELECT event_uuid FROM outbox").fetchall()]
        self.assertIn(critical["event_uuid"], uuids)
        self.assertLessEqual(len(uuids), 10)
        c.close()

    def test_dead_letter_can_be_replayed_after_configuration_fix(self):
        c = self._client()
        c._send = lambda p: (False, "HTTP 403: license", True)
        c.post_event(_event(1))
        with c._lock:
            dlq_id = c._db.execute("SELECT id FROM dead_letter").fetchone()[0]
        self.assertTrue(c.requeue_dead_letter(dlq_id))
        self.assertEqual(self._count(c)[0], 1)
        with c._lock:
            self.assertEqual(c._db.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0], 0)
        c.close()

    def test_dead_letter_compacts_media_but_keeps_critical_snapshot(self):
        c = self._client()
        c._send = lambda p: (False, "HTTP 401: invalid key", True)
        low = _event(11); low["snapshot"] = "S" * 5000; low["clip"] = "C" * 5000; low["severity"] = 2
        critical = _event(12); critical["snapshot"] = "S" * 5000; critical["clip"] = "C" * 5000; critical["severity"] = 5
        c.post_event(low); c.post_event(critical)
        with c._lock:
            rows = c._db.execute("SELECT event_uuid,payload,severity FROM dead_letter ORDER BY id").fetchall()
        import json
        low_p = json.loads(rows[0][1]); crit_p = json.loads(rows[1][1])
        self.assertNotIn("clip", low_p); self.assertNotIn("snapshot", low_p)
        self.assertNotIn("clip", crit_p); self.assertIn("snapshot", crit_p)
        c.close()

    def test_successful_retry_still_drains(self):
        c = self._client()
        c._send = lambda p: (False, "offline", False)
        c.post_event(_event(1))
        c._send = lambda p: (True, "", False)
        deadline = time.time() + 8
        while time.time() < deadline and self._count(c)[0]:
            time.sleep(0.25)
        self.assertEqual(self._count(c)[0], 0)
        c.close()


class IntegrationBusLimits(unittest.TestCase):
    def setUp(self):
        import threading
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        os.environ["AICOS_INTEGRATION_WEBHOOK_URLS"] = "https://example.invalid/hook"
        os.environ["AICOS_INTEGRATION_STAGES"] = "approved"
        self.bus = IntegrationBus(self.conn, threading.Lock())
        self.bus.close()          # no background thread during the test

    def _rows(self, where="1=1"):
        return int(self.conn.execute(
            f"SELECT COUNT(*) n FROM integration_outbox WHERE {where}").fetchone()["n"])

    def _queue(self, n=1):
        for i in range(n):
            self.bus.queue("approved", {"id": i, "type": "t", "status": "approved"})

    def test_permanent_http_status_is_abandoned_not_retried(self):
        self._queue(1)
        self.bus.flush_once(sender=lambda row: (False, "HTTP 404: no such hook"))
        self.assertEqual(self._rows("abandoned_at IS NOT NULL"), 1)
        # an abandoned row must never be selected for delivery again
        calls = []
        self.bus.flush_once(sender=lambda row: (calls.append(1), (False, "x"))[1])
        self.assertEqual(calls, [])

    def test_gives_up_after_max_attempts(self):
        self.bus.max_attempts = 3
        self._queue(1)
        for _ in range(5):
            self.conn.execute("UPDATE integration_outbox SET next_attempt_at=0")
            self.bus.flush_once(sender=lambda row: (False, "connection refused"))
        self.assertEqual(self._rows("abandoned_at IS NOT NULL"), 1)
        self.assertGreaterEqual(self.bus.abandoned, 1)

    def test_transient_failure_keeps_retrying_below_the_limit(self):
        self.bus.max_attempts = 12
        self._queue(1)
        self.bus.flush_once(sender=lambda row: (False, "connection refused"))
        self.assertEqual(self._rows("abandoned_at IS NOT NULL"), 0)
        self.assertEqual(self._rows("delivered_at IS NULL"), 1)

    def test_delivered_rows_are_pruned_not_kept_forever(self):
        self._queue(3)
        self.bus.flush_once(sender=lambda row: (True, ""))
        self.assertEqual(self._rows("delivered_at IS NOT NULL"), 3)
        # age them past the history window, then let housekeeping run
        old = time.time() - self.bus.history_seconds - 60
        self.conn.execute("UPDATE integration_outbox SET delivered_at=?", (old,))
        self.bus._last_prune = 0
        self.bus._maybe_prune()
        self.assertEqual(self._rows(), 0)

    def test_recent_history_is_kept_for_the_operator(self):
        self._queue(2)
        self.bus.flush_once(sender=lambda row: (True, ""))
        self.bus._last_prune = 0
        self.bus._maybe_prune()
        self.assertEqual(self._rows(), 2)     # delivered just now — still visible

    def test_hard_ceiling_bounds_the_table(self):
        self.bus.max_rows = 5
        self.bus.history_seconds = 10 ** 9     # history window would keep everything
        self._queue(20)
        self.bus.flush_once(sender=lambda row: (True, ""), limit=100)
        self.bus._last_prune = 0
        self.bus._maybe_prune()
        self.assertLessEqual(self._rows(), 5)

    def test_status_surfaces_abandonment(self):
        self._queue(1)
        self.bus.flush_once(sender=lambda row: (False, "HTTP 403: forbidden"))
        st = self.bus.status()
        self.assertEqual(st["abandoned"], 1)
        self.assertEqual(st["pending"], 0)
        self.assertEqual(st["retrying"], 0)    # not hidden in active counts

    def test_unknown_sender_shape_is_never_treated_as_success(self):
        self._queue(1)
        self.bus.flush_once(sender=lambda row: {"ok": True})
        st = self.bus.status()
        self.assertEqual(st["delivered"], 0)
        self.assertEqual(st["pending"], 1)
        self.assertEqual(st["retrying"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
