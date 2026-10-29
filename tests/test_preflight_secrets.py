"""Preflight must catch a deployment the backend will refuse to start.

Preflight exists so a misconfiguration is found with an explanation, rather
than as a container that will not come up.
"""
import contextlib, io, os, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "edge"))

import preflight
from preflight import Report, PASS, WARN, FAIL

STRONG_A = "device-abcdefghijklmnopqrstuvwxyz1234"
STRONG_B = "operator-abcdefghijklmnopqrstuvwxy1234"


def _run(fn, *a):
    rep = Report()
    with contextlib.redirect_stdout(io.StringIO()):
        fn(rep, *a)
    return rep


def _status(rep, part):
    return [r[0] for r in rep.rows if part in r[1]]


class Secrets(unittest.TestCase):
    def test_missing_device_key_fails(self):
        self.assertEqual(_status(_run(preflight.check_secrets, "", STRONG_B),
                                 "device key"), [FAIL])

    def test_missing_operator_key_fails_when_shared_keys_enabled(self):
        self.assertEqual(_status(_run(preflight.check_secrets, STRONG_A, "", True),
                                 "operator key"), [FAIL])

    def test_missing_operator_key_passes_in_government_rbac_mode(self):
        rep = _run(preflight.check_secrets, STRONG_A, "", False)
        self.assertEqual(_status(rep, "shared operator keys"), [PASS])
        self.assertEqual(_status(rep, "role separation"), [PASS])

    def test_known_weak_key_fails(self):
        self.assertEqual(_status(_run(preflight.check_secrets,
                                      "local-test-key-12345", STRONG_B, True),
                                 "device key"), [FAIL])

    def test_short_key_fails(self):
        self.assertEqual(_status(_run(preflight.check_secrets, "short", STRONG_B, True),
                                 "device key"), [FAIL])

    def test_identical_keys_fail_role_separation(self):
        rep = _run(preflight.check_secrets, STRONG_A, STRONG_A, True)
        self.assertEqual(_status(rep, "role separation"), [FAIL])

    def test_two_strong_distinct_keys_pass(self):
        rep = _run(preflight.check_secrets, STRONG_A, STRONG_B, True)
        self.assertEqual(_status(rep, "device key"), [PASS])
        self.assertEqual(_status(rep, "operator key"), [PASS])
        self.assertEqual(_status(rep, "role separation"), [PASS])


class OutboxBudget(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.environ["AICOS_OUTBOX_PATH"] = os.path.join(self.td, "o.db")
        os.environ["AICOS_OUTBOX_MAX_MB"] = "256"
        os.environ["AICOS_OUTBOX_MAX_ROWS"] = "5000"

    def test_reports_limits(self):
        self.assertIn(_status(_run(preflight.check_outbox), "outbox limits")[0],
                      (PASS, WARN))

    def test_cap_larger_than_free_disk_warns(self):
        os.environ["AICOS_OUTBOX_MAX_MB"] = "99999999"
        self.assertEqual(_status(_run(preflight.check_outbox), "outbox limits"), [WARN])

    def test_non_numeric_cap_fails_without_crashing(self):
        os.environ["AICOS_OUTBOX_MAX_MB"] = "not-a-number"
        self.assertEqual(_status(_run(preflight.check_outbox), "outbox limits"), [FAIL])

    def test_existing_backlog_is_reported(self):
        with open(os.environ["AICOS_OUTBOX_PATH"], "wb") as fh:
            fh.write(b"0" * 1000)
        self.assertEqual(_status(_run(preflight.check_outbox), "outbox backlog"), [PASS])


if __name__ == "__main__":
    unittest.main(verbosity=2)
