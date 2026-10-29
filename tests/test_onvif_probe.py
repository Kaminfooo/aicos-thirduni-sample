import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge"))
from onvif_probe import probe

class Obj:
    def __init__(self, **kw): self.__dict__.update(kw)

class Media:
    def GetProfiles(self): return [Obj(token="P1", Name="Main")]
    def GetStreamUri(self, req): return Obj(Uri="rtsp://10.0.0.5/stream")
    def GetSnapshotUri(self, req): return Obj(Uri="http://10.0.0.5/snap.jpg")
class Dev:
    def GetDeviceInformation(self): return Obj(Manufacturer="TestCam", Model="T1", FirmwareVersion="1.2", SerialNumber="S1", HardwareId="H1")
class Events:
    def GetServiceCapabilities(self): return Obj(WSSubscriptionPolicySupport=True)
class Cam:
    def create_devicemgmt_service(self): return Dev()
    def create_media_service(self): return Media()
    def create_events_service(self): return Events()

def factory(host, port, user, password): return Cam()

class OnvifProbeTests(unittest.TestCase):
    def test_probe_normalizes_capabilities_without_credentials(self):
        out = probe("10.0.0.5", 80, "admin", "topsecret", camera_factory=factory)
        self.assertEqual(out["device"]["manufacturer"], "TestCam")
        self.assertTrue(out["aicos"]["profile_t_candidate"])
        self.assertTrue(out["aicos"]["profile_m_candidate"])
        self.assertNotIn("topsecret", str(out))
        self.assertFalse(out["aicos"]["conformance_claim"])

if __name__ == "__main__": unittest.main()
