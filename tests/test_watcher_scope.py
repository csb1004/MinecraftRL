import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minecraftrl import minerl_runtime


class WatcherScopeTests(unittest.TestCase):
    def test_same_pid_pair_uses_different_episode_directories_and_preserves_stale_file(self):
        self.assertTrue(hasattr(minerl_runtime,'launch_scoped_watcher'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'logs/minerl_watchers/watcher_596-887.pid'
            old.parent.mkdir(parents=True); old.write_text('939')
            runs=[root/'run-a',root/'run-b']
            for run in runs: run.mkdir()
            with patch('subprocess.check_call') as launch:
                for run in runs: minerl_runtime.launch_scoped_watcher(run,596,887,str(root/'owned-game'))
            self.assertEqual(launch.call_count,2)
            cwds=[Path(call.kwargs['cwd']) for call in launch.call_args_list]
            self.assertNotEqual(*cwds)
            for run,cwd in zip(runs,cwds): self.assertTrue(cwd.is_relative_to(run))
            self.assertEqual(old.read_text(),'939')
            for call in launch.call_args_list:
                self.assertEqual(call.args[0][1:],['-m','minerl.utils.process_watcher','596','887','--child-dirs',str(root/'owned-game')])

    def test_launch_failure_remains_visible(self):
        self.assertTrue(hasattr(minerl_runtime,'launch_scoped_watcher'))
        import subprocess
        with tempfile.TemporaryDirectory() as tmp, patch('subprocess.check_call',side_effect=subprocess.CalledProcessError(1,['watcher'])):
            with self.assertRaises(subprocess.CalledProcessError):
                minerl_runtime.launch_scoped_watcher(Path(tmp),10,20)
