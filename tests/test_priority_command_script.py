import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('pwsh'), 'PowerShell executable unavailable')
class PriorityCommandScriptTests(unittest.TestCase):
    def test_enqueue_and_reprioritize_publish_validated_runtime_commands(self):
        from minecraftrl.tasks import validate_command
        original=Path(__file__).resolve().parents[1]/'scripts/Send-TaskCommand.ps1'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); scripts=root/'scripts';scripts.mkdir()
            script=scripts/original.name;shutil.copy2(original,script)
            session=root/'results/session';session.mkdir(parents=True)
            cases=[['-Operation','enqueue','-TaskId','wood','-Goal','wood','-Count','2','-Priority','10',
                    '-DependsOn','earlier','-DeadlineSeconds','90'],
                   ['-Operation','reprioritize','-TaskId','wood','-ExpectedRevision','2','-Priority','80']]
            for args in cases:
                result=subprocess.run(['pwsh','-NoProfile','-File',str(script),'-SessionPath',str(session),*args],
                    capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)
            commands=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((session/'commands').glob('*.json'))]
            for command in commands:validate_command(command)
            self.assertEqual(commands[0]['depends_on'],['earlier'])
            self.assertEqual(commands[0]['deadline_s'],90)
            self.assertNotIn('expected_revision',commands[0])
            self.assertEqual(commands[1]['priority'],80)
