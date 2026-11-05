import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from minecraftrl.cli import episode
from minecraftrl.rm_conditioning import run_conditioning
from minecraftrl.runtime import StopGate


class RMControlClosureTests(unittest.TestCase):
    def run_cancelled(self, release_fails=False, policy_close_fails=False):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            marker = output/'rm-control-finished.json'
            gate = StopGate()
            order = []
            at_environment_close = []

            def release():
                order.append('release')
                if release_fails:
                    raise RuntimeError('release failed')
                return 'neutral_command_sent'

            def close_policy():
                order.append('policy')
                if policy_close_fails:
                    raise RuntimeError('policy close failed')

            def close_environment():
                order.append('environment')
                at_environment_close.append(json.loads(marker.read_text()) if marker.exists() else None)

            def act(screen):
                gate.stop('user_stop')
                return {'forward': 1}

            env = SimpleNamespace(
                seed=lambda seed: None,
                reset=lambda: {'pov': 'frame', 'inventory': {'oak_log': 0}},
                step=lambda action: self.fail('cancelled action was dispatched'),
                release=release, close=close_environment,
                reward=SimpleNamespace(completed=False), ticks=[], initial={})
            policy = SimpleNamespace(
                environment_type=lambda env, policy, **kwargs: env,
                reset=lambda goal: None, act=act, close=close_policy,
                rows=[], switches=[], mode='logo-staged', condition_metadata={})
            report = run_conditioning(policy, env, output, seed=1, gate=gate,
                                      save_screen=lambda *args: None)
            return report, order, at_environment_close, marker.exists()

    def test_cancelled_rm_hands_off_only_after_release_and_policy_close(self):
        report, order, markers, exists = self.run_cancelled()
        self.assertEqual(report['status'], 'cancelled')
        self.assertFalse(report['execution_valid'])
        self.assertEqual(report['steps'], 0)
        self.assertEqual(order, ['release', 'policy', 'environment'])
        self.assertTrue(exists)
        self.assertTrue(markers[0]['control_closed'])
        self.assertTrue(markers[0]['cancelled_archive'])
        self.assertEqual(markers[0]['release'], 'neutral_command_sent')

    def test_failed_release_never_marks_control_closed(self):
        report, _, markers, exists = self.run_cancelled(release_fails=True)
        self.assertFalse(report['execution_valid'])
        self.assertFalse(exists)
        self.assertEqual(markers, [None])

    def test_failed_policy_close_never_marks_control_closed(self):
        report, _, markers, exists = self.run_cancelled(policy_close_fails=True)
        self.assertIn('policy_close_error', report)
        self.assertFalse(exists)
        self.assertEqual(markers, [None])


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux supervisor integration')
class RMFinalizationTests(unittest.TestCase):
    def run_worker(self, grace=1., manual_stop=False, marker=True, cancelled_archive=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'configs').mkdir()
            (root/'configs/experiment.json').write_text(json.dumps(dict(
                startup_limit_s=3, shutdown_grace_s=.05, rl_finalization_grace_s=grace)))
            real_popen=subprocess.Popen
            def launch(argv, **kwargs):
                output=Path(argv[argv.index('--output')+1])
                code=('import json,time,pathlib\n'
                      'p=pathlib.Path('+repr(str(output))+')\n'
                      '(p/"events.jsonl").write_text(json.dumps(dict(event="episode_started",monotonic_s=time.monotonic()))+"\\n")\n'
                      '(p/"summary.json").write_text(json.dumps(dict(status="time_limit",execution_valid=True,release="neutral_command_sent")))\n'+
                      ('(p/"rm-control-finished.json").write_text(json.dumps(dict(monotonic_s=time.monotonic(),control_closed=True,release="neutral_command_sent",cancelled_archive='+repr(cancelled_archive)+')))\n' if marker else '')+
                      ('pathlib.Path('+repr(str(root/'STOP'))+').touch()\n' if manual_stop else '')+
                      'time.sleep(.3)\n'+('raise SystemExit(2)\n' if cancelled_archive else ''))
                return real_popen([sys.executable,'-c',code],**kwargs)
            args=argparse.Namespace(root=root, policy='steve1', output=root/'run', seed=1,
                                    limit_s=.05, stop_file=root/'STOP',
                                    rm_conditioning='staged', rm_goal_bank=root/'results/bank')
            with patch('minecraftrl.cli.inspect',return_value={'status':'ready_for_execution_attempt'}), \
                 patch('minecraftrl.cli.command',return_value={}), \
                 patch('minecraftrl.cli.subprocess.Popen',side_effect=launch), \
                 patch('minecraftrl.cli.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','test')):
                return episode(args, {'name':'wood'})

    def test_closed_rm_control_can_finish_interpreter_teardown(self):
        report=self.run_worker()
        self.assertTrue(report['execution_valid'])
        self.assertEqual(report['worker_exit_code'],0)
        self.assertEqual(report['process_cleanup']['remaining_live_pids'],[])

    def test_marker_does_not_extend_manual_stop(self):
        report=self.run_worker(manual_stop=True)
        self.assertEqual(report['status'],'cancelled')
        self.assertFalse(report['execution_valid'])
        self.assertNotEqual(report['worker_exit_code'],0)

    def test_finalization_is_bounded(self):
        report=self.run_worker(grace=.05)
        self.assertFalse(report['execution_valid'])
        self.assertNotEqual(report['worker_exit_code'],0)

    def test_verified_cancelled_control_can_archive_without_resuming(self):
        report=self.run_worker(manual_stop=True, cancelled_archive=True)
        self.assertEqual(report['status'], 'cancelled')
        self.assertFalse(report['execution_valid'])
        self.assertEqual(report['worker_exit_code'], 2)
        self.assertFalse(report['rm_archive_timed_out'])
        self.assertEqual(report['process_cleanup']['remaining_live_pids'], [])

    def test_cancelled_archive_still_has_deadline(self):
        report=self.run_worker(grace=.05, manual_stop=True, cancelled_archive=True)
        self.assertEqual(report['status'], 'cancelled')
        self.assertTrue(report['rm_archive_timed_out'])
        self.assertNotEqual(report['worker_exit_code'], 2)

    def test_missing_marker_preserves_original_shutdown_deadline(self):
        report=self.run_worker(marker=False)
        self.assertFalse(report['execution_valid'])
        self.assertNotEqual(report['worker_exit_code'],0)


if __name__=='__main__':
    unittest.main()
