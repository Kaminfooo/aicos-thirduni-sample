import os, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

class VersionConsistency(unittest.TestCase):
    def test_runtime_python_files_do_not_hardcode_old_release(self):
        offenders = []
        for folder in (ROOT / "backend", ROOT / "edge"):
            for path in folder.glob("*.py"):
                text = path.read_text(encoding="utf-8")
                if '"6.0.0"' in text or '"6.0.1"' in text or "v6.0.0" in text:
                    if path.name == "rtsp_worker.py" and "Upgrade v6.0/v6.0.1" in text:
                        text = text.replace("Upgrade v6.0/v6.0.1", "")
                    if '"6.0.0"' in text or '"6.0.1"' in text or "v6.0.0" in text:
                        offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_backend_version_loader_reads_version_file(self):
        sys.path.insert(0, str(ROOT / "backend"))
        import versioning
        self.assertEqual(versioning.AICOS_VERSION, EXPECTED)


if __name__ == "__main__": unittest.main()
