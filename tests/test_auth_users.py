import os
import sqlite3
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from auth_users import UserAuth, hash_password, verify_password


class T(unittest.TestCase):
    def test_password_and_session_lifecycle(self):
        h = hash_password('correct horse battery staple')
        self.assertTrue(verify_password('correct horse battery staple', h))
        self.assertFalse(verify_password('wrong password', h))
        c = sqlite3.connect(':memory:', check_same_thread=False); c.row_factory = sqlite3.Row
        a = UserAuth(c, threading.RLock())
        a.create_user('alice', 'a very strong password 123', 'operator')
        self.assertIsNone(a.login('alice', 'bad'))
        token, user = a.login('alice', 'a very strong password 123')
        self.assertEqual(user['role'], 'operator')
        self.assertEqual(a.resolve(token)['username'], 'alice')
        self.assertTrue(a.logout(token))
        self.assertIsNone(a.resolve(token))
        c.close()

    def test_resolve_periodically_purges_expired_and_revoked_sessions(self):
        old = os.environ.get('AICOS_SESSION_PURGE_SECONDS')
        os.environ['AICOS_SESSION_PURGE_SECONDS'] = '60'
        try:
            c = sqlite3.connect(':memory:', check_same_thread=False); c.row_factory = sqlite3.Row
            a = UserAuth(c, threading.RLock())
            a.create_user('bob', 'another very strong password 123', 'viewer')
            token, _ = a.login('bob', 'another very strong password 123')
            with a.lock:
                c.execute('UPDATE auth_sessions SET expires_at=?', (time.time() - 10,))
                c.commit()
            a._last_purge = 0
            self.assertIsNone(a.resolve(token))
            with a.lock:
                count = c.execute('SELECT COUNT(*) n FROM auth_sessions').fetchone()['n']
            self.assertEqual(count, 0)
            c.close()
        finally:
            if old is None: os.environ.pop('AICOS_SESSION_PURGE_SECONDS', None)
            else: os.environ['AICOS_SESSION_PURGE_SECONDS'] = old


if __name__ == '__main__': unittest.main()
