"""Native timebase must be validated before timing an evaluation from AVI frames."""
import struct
import sys
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from avi_reference_clock import read_clock


def chunk(tag,data):return tag+struct.pack('<I',len(data))+data+(b'\0' if len(data)&1 else b'')
def header(rate=30000):
    body=bytearray(56);body[:4]=b'vids';struct.pack_into('<IIII',body,20,1001,rate,0,300)
    return chunk(b'LIST',b'strl'+chunk(b'strh',bytes(body)))
def avi(*headers):
    body=b'AVI '+chunk(b'LIST',b'hdrl'+b''.join(headers))
    return chunk(b'RIFF',body)


class AviReferenceClockTests(unittest.TestCase):
    def parse(self,data):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.avi';path.write_bytes(data);return read_clock(path)
    def test_exact_rational_timebase_and_declared_count(self):
        result=self.parse(avi(header()))
        self.assertEqual((result['scale'],result['rate'],result['frames']),(1001,30000,300))
    def test_multiple_video_streams_rejected(self):
        with self.assertRaisesRegex(ValueError,'exactly one'):self.parse(avi(header(),header()))
    def test_zero_rate_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unsupported AVI clock'):self.parse(avi(header(0)))
    def test_truncated_chunk_boundary_rejected(self):
        with self.assertRaises(ValueError):self.parse(avi(header())[:-3])
