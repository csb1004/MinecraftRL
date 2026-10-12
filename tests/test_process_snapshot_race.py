import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from minecraftrl.processes import snapshot


class ProcessSnapshotRaceTests(unittest.TestCase):
    def test_process_exit_during_scan_does_not_abort_supervisor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for pid in (123,456,789): (root/str(pid)).mkdir()
            stat='123 (worker (child)) '+' '.join(['S','42']+['0']*17+['99'])
            for pid in (123,789):(root/str(pid)/'stat').write_text(stat)
            # PID 456 disappeared after directory enumeration. Path.glob also
            # stats nested paths BEFORE the caller's protected read_text block.
            def racing_glob(pattern):
                yield root/'123/stat'
                raise ProcessLookupError(3,'No such process')
            fake=Mock();fake.glob.side_effect=racing_glob
            fake.iterdir.return_value=iter([root/'123',root/'456',root/'789',root/'self'])
            with patch('minecraftrl.processes.Path',return_value=fake):table=snapshot()
            self.assertEqual(set(table),{123,789})
            self.assertEqual(table[789],{'ppid':42,'start':99,'state':'S'})
