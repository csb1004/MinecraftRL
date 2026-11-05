import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from minecraftrl import artifact_hash as ah


class ArtifactHashTests(unittest.TestCase):
    def test_drive_paths_are_data_arguments(self):
        self.assertEqual(ah.windows_drive_path('/mnt/d/project with space/a.jar'),r'D:\project with space\a.jar')
        for path in ['/home/user/a.jar','/mnt/network/a','/mnt/d/../secret','/mnt/d/a:b']:
            with self.assertRaises(ValueError):ah.windows_drive_path(path)
    def test_full_read_observes_changed_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'data';path.write_bytes(b'first')
            first=ah.sha256(path);path.write_bytes(b'other')
            self.assertEqual(first,hashlib.sha256(b'first').hexdigest())
            self.assertNotEqual(first,ah.sha256(path))
    def exercise(self,reply,code=0):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'data';path.write_bytes(b'data')
            with patch.object(ah,'native_bridge',return_value=('powershell.exe','D:\\project\\hash.ps1','D:\\project\\data')),patch.object(
                ah.subprocess,'run',return_value=SimpleNamespace(returncode=code,stdout=json.dumps(reply))) as run:
                result=ah.sha256(path)
                self.assertNotIn('shell',run.call_args.kwargs)
                return result
    def test_native_full_hash_is_verified(self):
        digest=hashlib.sha256(b'data').hexdigest()
        self.assertEqual(self.exercise(dict(schema=1,sha256=digest,bytes=4)),digest)
    def test_bridge_error_is_not_silently_bypassed(self):
        with self.assertRaises(ValueError):self.exercise({},code=1)
    def test_wrong_length_or_digest_fails_closed(self):
        for data in [dict(schema=1,sha256='0'*64,bytes=5),dict(schema=1,sha256='not-a-hash',bytes=4)]:
            with self.assertRaises(ValueError):self.exercise(data)
