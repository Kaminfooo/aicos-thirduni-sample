"""Local RBAC user/session authentication for AICOS control-room operators.

Designed for sovereign/offline deployments that cannot depend on cloud identity.
Passwords are hashed with stdlib scrypt; bearer session tokens are random and
only their SHA-256 digests are stored. A later OIDC/AD adapter can coexist with
this local emergency-access path.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time

ROLES = {"viewer", "operator", "auditor", "admin"}


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    if len(password) < 12:
        raise ValueError("password must be at least 12 characters")
    salt = salt or os.urandom(16)
    n, r, p = 2**14, 8, 1
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        kind, ns, rs, ps, salt_s, hash_s = encoded.split("$", 5)
        if kind != "scrypt": return False
        expected = _unb64(hash_s)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=_unb64(salt_s), n=int(ns), r=int(rs), p=int(ps), dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


class UserAuth:
    def __init__(self, conn, lock):
        self.conn, self.lock = conn, lock
        self.session_seconds = max(900, int(os.environ.get("AICOS_SESSION_SECONDS", "43200")))
        self._last_purge = 0.0
        self._purge_interval = max(60.0, float(os.environ.get("AICOS_SESSION_PURGE_SECONDS", "300")))
        self._init_db()
        self._bootstrap()

    def _init_db(self):
        with self.lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_users(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    last_login REAL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_sessions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    revoked_at REAL,
                    FOREIGN KEY(user_id) REFERENCES auth_users(id)
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_exp ON auth_sessions(expires_at,revoked_at)")
            self.conn.commit()

    def _bootstrap(self):
        user = os.environ.get("AICOS_BOOTSTRAP_ADMIN_USER", "").strip()
        password = os.environ.get("AICOS_BOOTSTRAP_ADMIN_PASSWORD", "")
        if not user or not password:
            return
        if len(password) < 14 or password.upper().startswith("REPLACE_") or password.lower() in {"password", "changeme", "change-me"}:
            raise RuntimeError("AICOS_BOOTSTRAP_ADMIN_PASSWORD must be a non-placeholder password of at least 14 characters")
        with self.lock:
            count = int(self.conn.execute("SELECT COUNT(*) n FROM auth_users").fetchone()["n"])
        if count == 0:
            self.create_user(user, password, "admin")

    def create_user(self, username: str, password: str, role: str) -> dict:
        username = username.strip()
        role = role.strip().lower()
        if not username or len(username) > 80: raise ValueError("invalid username")
        if role not in ROLES: raise ValueError("invalid role")
        ph = hash_password(password)
        with self.lock:
            cur = self.conn.execute("INSERT INTO auth_users(username,password_hash,role,enabled,created_at) VALUES(?,?,?,?,?)",
                                    (username, ph, role, 1, time.time()))
            self.conn.commit()
            uid = cur.lastrowid
        return {"id": uid, "username": username, "role": role, "enabled": True}

    def set_enabled(self, username: str, enabled: bool) -> bool:
        with self.lock:
            cur = self.conn.execute("UPDATE auth_users SET enabled=? WHERE username=?", (1 if enabled else 0, username))
            if not enabled:
                self.conn.execute("UPDATE auth_sessions SET revoked_at=? WHERE user_id IN (SELECT id FROM auth_users WHERE username=?) AND revoked_at IS NULL", (time.time(), username))
            self.conn.commit()
            return cur.rowcount > 0

    def login(self, username: str, password: str) -> tuple[str, dict] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM auth_users WHERE username=? COLLATE NOCASE", (username.strip(),)).fetchone()
        if row is None or not int(row["enabled"]) or not verify_password(password, row["password_hash"]):
            # burn comparable scrypt work for missing users
            if row is None:
                try: hash_password(password if len(password) >= 12 else password + "0" * (12-len(password)), salt=b"AICOS-dummy-salt")
                except Exception: pass
            return None
        self._maybe_purge()
        token = secrets.token_urlsafe(40)
        th = hashlib.sha256(token.encode()).hexdigest()
        now = time.time(); exp = now + self.session_seconds
        with self.lock:
            self.conn.execute("INSERT INTO auth_sessions(user_id,token_hash,created_at,expires_at) VALUES(?,?,?,?)", (row["id"], th, now, exp))
            self.conn.execute("UPDATE auth_users SET last_login=? WHERE id=?", (now,row["id"]))
            self.conn.commit()
        return token, {"id":row["id"],"username":row["username"],"role":row["role"],"expires_at":exp}

    def resolve(self, token: str) -> dict | None:
        if not token:
            return None
        self._maybe_purge()
        th = hashlib.sha256(token.encode()).hexdigest()
        now = time.time()
        with self.lock:
            row = self.conn.execute("""
                SELECT u.id,u.username,u.role,u.enabled,s.expires_at FROM auth_sessions s
                JOIN auth_users u ON u.id=s.user_id
                WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at>? AND u.enabled=1
            """, (th, now)).fetchone()
        return dict(row) if row is not None else None

    def logout(self, token: str) -> bool:
        th = hashlib.sha256(token.encode()).hexdigest()
        with self.lock:
            cur = self.conn.execute("UPDATE auth_sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL", (time.time(), th))
            self.conn.commit(); return cur.rowcount > 0

    def count_users(self) -> int:
        with self.lock:
            return int(self.conn.execute("SELECT COUNT(*) n FROM auth_users").fetchone()["n"])

    def _maybe_purge(self) -> None:
        """Periodically remove expired/revoked sessions on active servers.

        This is throttled and request-driven: a long-running control room no
        longer depends on a restart or a fresh login for housekeeping.
        """
        now = time.time()
        if now - self._last_purge < self._purge_interval:
            return
        try:
            self.purge_expired_sessions()
            self._last_purge = now
        except Exception:
            # Authentication availability is more important than housekeeping.
            # The next request will retry after the interval.
            return

    def purge_expired_sessions(self) -> int:
        now = time.time()
        with self.lock:
            cur = self.conn.execute("DELETE FROM auth_sessions WHERE expires_at <= ? OR revoked_at IS NOT NULL", (now,))
            self.conn.commit()
            return int(cur.rowcount)

    def list_users(self) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT id,username,role,enabled,created_at,last_login FROM auth_users ORDER BY username").fetchall()
        return [{**dict(r),"enabled":bool(r["enabled"])} for r in rows]
