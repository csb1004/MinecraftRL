import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from minecraftrl.cli import episode


class RLFinalizationTests(unittest.TestCase):
    def test_failed_supervisor_verdict_rejects_saved_trajectory(self):
        from minecraftrl.rl_core import validate_collector_result
        with self.assertRaises(ValueError):
            validate_collector_result(dict(execution_valid=False,worker_exit_code=-15,rl_training_eligible=True))
        validate_collector_result(dict(execution_valid=True,worker_exit_code=0))

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux supervisor integration')
    def test_closed_game_gets_bounded_artifact_time(self):
        report=self.run_closed_worker(1,False)
        self.assertTrue(report['execution_valid'])
        self.assertEqual(report['worker_exit_code'],0)
        self.assertEqual(report['process_cleanup']['remaining_live_pids'],[])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux supervisor integration')
    def test_artifact_time_is_bounded(self):
        report=self.run_closed_worker(.05,False)
        self.assertFalse(report['execution_valid'])
        self.assertNotEqual(report['worker_exit_code'],0)
        self.assertEqual(report['process_cleanup']['remaining_live_pids'],[])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux supervisor integration')
    def test_manual_stop_is_not_delayed_by_artifact_grace(self):
        report=self.run_closed_worker(1,True)
        self.assertEqual(report['status'],'cancelled')
        self.assertFalse(report['execution_valid'])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux supervisor integration')
    def test_verified_cancelled_control_can_finish_artifacts_without_resuming(self):
        report=self.run_closed_worker(1,True,cancelled_archive=True)
        self.assertEqual(report['status'],'cancelled');self.assertFalse(report['execution_valid'])
        self.assertEqual(report['worker_exit_code'],0)
        self.assertEqual(report['process_cleanup']['remaining_live_pids'],[])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux supervisor integration')
    def test_cancelled_archive_grace_remains_bounded(self):
        report=self.run_closed_worker(.05,True,cancelled_archive=True)
        self.assertEqual(report['status'],'cancelled');self.assertNotEqual(report['worker_exit_code'],0)
        self.assertEqual(report['process_cleanup']['remaining_live_pids'],[])

    def run_closed_worker(self,grace,manual_stop,*,cancelled_archive=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir()
            (root/'configs/experiment.json').write_text(json.dumps(dict(startup_limit_s=3,shutdown_grace_s=.05,rl_finalization_grace_s=grace)))
            real_popen=subprocess.Popen
            def launch(argv,**kwargs):
                self.assertEqual(argv[argv.index('--seed')+1],'1')
                self.assertEqual(argv[argv.index('--rl-sampling-seed')+1],'701')
                output=Path(argv[argv.index('--output')+1])
                code=('import json,time,pathlib\n'
                      'p=pathlib.Path('+repr(str(output))+')\n'
                      '(p/"events.jsonl").write_text(json.dumps(dict(event="episode_started",monotonic_s=time.monotonic()))+"\\n")\n'
                      '(p/"rl-control-finished.json").write_text(json.dumps(dict(monotonic_s=time.monotonic(),control_closed=True,release="neutral_command_sent",cancelled_archive='+repr(cancelled_archive)+')))\n'+
                      ('pathlib.Path('+repr(str(root/'STOP'))+').touch()\n' if manual_stop else '')+
                      'time.sleep(.3)\n'
                      '(p/"summary.json").write_text(json.dumps(dict(status="time_limit",execution_valid=True,release="neutral_command_sent")))\n')
                return real_popen([sys.executable,'-c',code],**kwargs)
            args=argparse.Namespace(root=root,policy='steve1',output=root/'run',seed=1,rl_sampling_seed=701,limit_s=.05,stop_file=root/'STOP',rl_collect=True,rl_split='eval',rl_checkpoint=None,rl_embed=None,rl_world=None,rl_prepare=False)
            with patch('minecraftrl.cli.inspect',return_value={'status':'ready_for_execution_attempt'}),patch('minecraftrl.cli.command',return_value={}),patch('minecraftrl.cli.subprocess.Popen',side_effect=launch),patch('minecraftrl.cli.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','test')):
                report=episode(args,{'name':'wood'})
            return report
