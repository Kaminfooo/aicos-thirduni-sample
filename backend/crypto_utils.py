"""Cryptographic helpers for AICOS Government Edition.

The public runtime only needs public verification keys. Vendor/private keys
stay offline and are used by the administrative CLI to issue licenses and
camera certificates. Evidence signing uses a deployment-scoped private key
mounted into the backend at runtime; it is never baked into the image.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64d(value: str) -> bytes:
    raw = value.encode("ascii")
    return base64.urlsafe_b64decode(raw + b"=" * (-len(raw) % 4))


def load_private_key(path: str | Path) -> Ed25519PrivateKey:
    raw = Path(path).read_bytes()
    key = serialization.load_pem_private_key(raw, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("AICOS requires an Ed25519 private key")
    return key


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    raw = Path(path).read_bytes()
    key = serialization.load_pem_public_key(raw)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("AICOS requires an Ed25519 public key")
    return key


def public_fingerprint(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:24]


def write_keypair(private_path: str | Path, public_path: str | Path) -> tuple[str, str]:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    private_pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = pub.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_path = Path(private_path)
    public_path = Path(public_path)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(private_pem)
    try:
        private_path.chmod(0o600)
    except OSError:
        pass
    public_path.write_bytes(public_pem)
    return public_fingerprint(pub), str(public_path)


def sign_envelope(schema: str, payload: Mapping[str, Any], private_key: Ed25519PrivateKey) -> dict:
    body = {"schema": str(schema), "payload": dict(payload)}
    signature = private_key.sign(canonical_json(body))
    return {
        **body,
        "algorithm": "Ed25519",
        "key_id": public_fingerprint(private_key.public_key()),
        "signature": b64e(signature),
    }


def verify_envelope(envelope: Mapping[str, Any], public_key: Ed25519PublicKey,
                    expected_schema: str | None = None) -> dict:
    if envelope.get("algorithm") != "Ed25519":
        raise ValueError("unsupported signature algorithm")
    if expected_schema and envelope.get("schema") != expected_schema:
        raise ValueError(f"expected schema {expected_schema!r}")
    advertised = str(envelope.get("key_id", ""))
    actual = public_fingerprint(public_key)
    if advertised and advertised != actual:
        raise ValueError("signature key_id does not match configured public key")
    body = {"schema": envelope.get("schema"), "payload": envelope.get("payload")}
    signature = b64d(str(envelope.get("signature", "")))
    public_key.verify(signature, canonical_json(body))
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("signed payload must be an object")
    return dict(payload)
