import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge"))
from forensic_indexer import DirectionTracker, _dominant_color


class ForensicIndexerTests(unittest.TestCase):
    def test_dominant_color_red(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 220)  # BGR red
        self.assertEqual(_dominant_color(frame, (0, 0, 100, 100), "vehicle"), "red")

    def test_direction_tracker(self):
        t = DirectionTracker()
        tid, d = t.update("person", .2, .5, 10.0)
        self.assertEqual(d, "stationary")
        tid2, d2 = t.update("person", .35, .5, 11.0)
        self.assertEqual(tid, tid2)
        self.assertEqual(d2, "right")


if __name__ == "__main__":
    unittest.main()
