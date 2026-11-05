"""Bank preflight must understand queue layout before starting any process."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


path=Path(__file__).resolve().parents[1]/'scripts/probe_rm_conditioning_stop.py'
spec=importlib.util.spec_from_file_location('rm_stop_probe_under_test',path)
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class StopProbeBankTests(unittest.TestCase):
    def invoke(self, mode, nested, expected):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            bank=root/'results/pilot/goal-bank'
            manifest=(bank/'wood' if nested else bank)/'manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{}',encoding='utf-8')
            (root/'results/STOP').write_text('Preserve existing STOP',encoding='utf-8')
            argv=['probe','--root',str(root),'--batch','pilot','--mode',mode]
            with patch.object(sys,'argv',argv),patch.object(probe.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(expected[0],expected[1]):probe.main()
                launch.assert_not_called()

    def test_queue_nested_bank_is_recognized_then_stop_is_honored(self):
        self.invoke('goal-queue',True,(RuntimeError,'STOP latched'))

    def test_legacy_flat_bank_still_honors_stop(self):
        self.invoke('logo-staged',False,(RuntimeError,'STOP latched'))

    def test_queue_does_not_accept_flat_bank_without_wood_manifest(self):
        self.invoke('goal-queue',False,(ValueError,'existing project-local goal bank'))


if __name__=='__main__':unittest.main()
