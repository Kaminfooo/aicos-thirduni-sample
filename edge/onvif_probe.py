#!/usr/bin/env python3
"""ONVIF interoperability probe for AICOS.

This is an integration aid, not an ONVIF conformance claim. It connects to a
known camera address, reports device information, media profiles and RTSP
stream URIs, and checks whether event services are exposed. Credentials are
never printed in output.

Optional dependency: ``pip install -r requirements-onvif.txt``.
"""
from __future__ import annotations

import argparse
import json
from typing import Callable


def _text(value):
    if value is None:
        return ""
    return str(value)


def probe(host: str, port: int, username: str, password: str, *, camera_factory: Callable | None = None) -> dict:
    if camera_factory is None:
        try:
            from onvif import ONVIFCamera
        except Exception as exc:
            raise RuntimeError("ONVIF support is optional; install edge/requirements-onvif.txt") from exc
        camera_factory = ONVIFCamera
    cam = camera_factory(host, int(port), username, password)
    result = {"host": host, "port": int(port), "device": {}, "profiles": [], "services": {}}

    dev = cam.create_devicemgmt_service()
    info = dev.GetDeviceInformation()
    for key in ("Manufacturer", "Model", "FirmwareVersion", "SerialNumber", "HardwareId"):
        result["device"][key.lower()] = _text(getattr(info, key, ""))

    media = cam.create_media_service()
    profiles = media.GetProfiles() or []
    for profile in profiles:
        token = _text(getattr(profile, "token", ""))
        name = _text(getattr(profile, "Name", ""))
        item = {"token": token, "name": name, "rtsp_uri": "", "snapshot_uri": ""}
        try:
            uri = media.GetStreamUri({"StreamSetup": {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}}, "ProfileToken": token})
            item["rtsp_uri"] = _text(getattr(uri, "Uri", ""))
        except Exception as exc:
            item["stream_error"] = str(exc)[:160]
        try:
            snap = media.GetSnapshotUri({"ProfileToken": token})
            item["snapshot_uri"] = _text(getattr(snap, "Uri", ""))
        except Exception:
            pass
        result["profiles"].append(item)

    try:
        events = cam.create_events_service()
        caps = events.GetServiceCapabilities()
        result["services"]["events"] = {"available": True, "capabilities": _safe_object(caps)}
    except Exception as exc:
        result["services"]["events"] = {"available": False, "reason": str(exc)[:160]}

    result["aicos"] = {
        "profile_t_candidate": bool(result["profiles"] and any(p.get("rtsp_uri") for p in result["profiles"])),
        "profile_m_candidate": bool(result["services"].get("events", {}).get("available")),
        "conformance_claim": False,
        "note": "Capabilities are probed at runtime; formal ONVIF conformance requires the official ONVIF process.",
    }
    return result


def _safe_object(obj):
    if obj is None:
        return {}
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_object(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_object(v) for k, v in obj.items()}
    out = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        try:
            value = getattr(obj, key)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser(description="Probe one ONVIF camera for AICOS integration")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=80)
    ap.add_argument("--username", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    result = probe(args.host, args.port, args.username, args.password)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
