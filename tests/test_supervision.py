import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from minecraftrl.cli import episode
from minecraftrl.policies import MineRLEnvironment


class TerminalRaw:
    class action_space:
        @staticmethod
        def noop(): return {}
    def seed(self, seed): pass
    def reset(self): self.done = False; return {}
    def step(self, action):
        if self.done: raise RuntimeError('step after done')
        self.done = True
        return {}, 0, True, {}
    def close(self): pass


class SupervisorTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux supervisor integration')
    def test_planned_pause_keeps_its_reason_when_worker_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir()
            (root/'configs/experiment.json').write_text(json.dumps({'startup_limit_s':5,'shutdown_grace_s':1}))
            real_popen=subprocess.Popen
            def launch(argv,**kwargs):
                output=Path(argv[argv.index('--output')+1])
                code=('import json,pathlib,sys; p=pathlib.Path('+repr(str(output))+'); '
                      '(p/"summary.json").write_text(json.dumps({"status":"planning_paused","execution_valid":False,"steps":402,"release":"neutral_command_sent"})); '
                      'sys.exit(2)')
                return real_popen([sys.executable,'-c',code],**kwargs)
            args=argparse.Namespace(root=root,policy='steve1',output=root/'run',seed=1,limit_s=180,stop_file=root/'STOP')
            with patch('minecraftrl.cli.inspect',return_value={'status':'ready_for_execution_attempt'}), \
                 patch('minecraftrl.cli.command',return_value={}), \
                 patch('minecraftrl.cli.subprocess.Popen',side_effect=launch), \
                 patch('minecraftrl.cli.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','fixture')):
                report=episode(args,{'name':'wood'})
            self.assertEqual(report['status'],'planning_paused')
            self.assertEqual(report['worker_exit_code'],2)
            self.assertFalse(report['execution_valid'])
            self.assertEqual(report['process_cleanup']['remaining_live_pids'],[])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux supervisor integration')
    def test_prior_pause_release_is_not_shutdown_release_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir()
            (root/'configs/experiment.json').write_text(json.dumps({'startup_limit_s':2,'shutdown_grace_s':.05}))
            real_popen=subprocess.Popen
            def launch(argv, **kwargs):
                output=Path(argv[argv.index('--output')+1])
                code=('import json,time,pathlib,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                      'p=pathlib.Path('+repr(str(output))+'); '
                      'rows=[{"event":"episode_started","monotonic_s":time.monotonic()},'
                      '{"event":"input_release","controller":"cleanup","reason":"pause","outcome":"neutral_command_sent"},'
                      '{"event":"policy_dispatch_started","monotonic_s":time.monotonic()}]; '
                      '(p/"events.jsonl").write_text("\\n".join(json.dumps(x) for x in rows)+"\\n"); time.sleep(30)')
                return real_popen([sys.executable,'-c',code],**kwargs)
            args=argparse.Namespace(root=root,policy='steve1',output=root/'run',seed=1,limit_s=.1,stop_file=root/'STOP')
            with patch('minecraftrl.cli.inspect',return_value={'status':'ready_for_execution_attempt'}), \
                 patch('minecraftrl.cli.command',return_value={}), \
                 patch('minecraftrl.cli.subprocess.Popen',side_effect=launch), \
                 patch('minecraftrl.cli.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','test')):
                report=episode(args,{'name':'wood'})
            self.assertTrue(report['release'].startswith('unverified'))
            self.assertNotIn('release_evidence',report)
            self.assertFalse(report['execution_valid'])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux supervisor integration')
    def test_forced_shutdown_retains_confirmed_release_without_claiming_valid_gameplay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'configs').mkdir()
            (root/'configs/experiment.json').write_text(json.dumps({'startup_limit_s':2,'shutdown_grace_s':.05}))
            real_popen = subprocess.Popen
            def launch(argv, **kwargs):
                output = Path(argv[argv.index('--output')+1])
                code = ('import json,time,pathlib,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                        'p=pathlib.Path('+repr(str(output))+'); '
                        'rows=[{"event":"episode_started","monotonic_s":time.monotonic()},'
                        '{"event":"stop_acknowledged","monotonic_s":time.monotonic(),"steps":1},'
                        '{"event":"input_release","controller":"cleanup","outcome":"neutral_command_sent"}]; '
                        '(p/"events.jsonl").write_text("\\n".join(json.dumps(x) for x in rows)+"\\n{\\\"event\\\":"); '
                        'time.sleep(30)')
                return real_popen([sys.executable,'-c',code],**kwargs)
            args = argparse.Namespace(root=root,policy='vpt',output=root/'episode',seed=1,limit_s=.1,stop_file=root/'STOP')
            with patch('minecraftrl.cli.inspect',return_value={'status':'ready_for_execution_attempt'}), \
                 patch('minecraftrl.cli.command',return_value={}), \
                 patch('minecraftrl.cli.subprocess.Popen',side_effect=launch), \
                 patch('minecraftrl.cli.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','no GPU in test')):
                report = episode(args,{'name':'spontaneous','goal':None,'target_items':[]})
            self.assertEqual(report['release'], 'neutral_command_sent')
            self.assertEqual(report['release_evidence'], 'events.jsonl:input_release')
            self.assertFalse(report['execution_valid'])
            self.assertEqual(report['status'], 'hard_timeout')
            self.assertEqual(report['worker_exit_code'], -9)
            self.assertEqual(report['process_cleanup']['remaining_live_pids'], [])

    def test_cleanup_before_reset_does_not_step_uninitialized_environment(self):
        env = MineRLEnvironment(TerminalRaw())
        self.assertEqual(env.release(), 'environment_not_initialized_no_policy_input')

    def test_terminal_cleanup_does_not_step_finished_minerl(self):
        env = MineRLEnvironment(TerminalRaw()); env.reset(); env.step({})
        self.assertEqual(env.release(), 'environment_done_no_neutral_step')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux supervisor integration')
    def test_normal_deadlines_do_not_latch_user_stop_across_five_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'configs').mkdir()
            (root/'configs/experiment.json').write_text(json.dumps({'startup_limit_s':2,'shutdown_grace_s':1}))
            stop = root/'STOP'; real_popen = subprocess.Popen
            def launch(argv, **kwargs):
                output = Path(argv[argv.index('--output')+1])
                code = ('import json,time,pathlib; p=pathlib.Path('+repr(str(output))+'); '
                    '(p/"events.jsonl").write_text(json.dumps({"event":"episode_started","monotonic_s":time.monotonic()})+"\\n"); '
                    'time.sleep(.15); '
                    '(p/"summary.json").write_text(json.dumps({"status":"time_limit","execution_valid":True,"steps":1}))')
                return real_popen([sys.executable,'-c',code],**kwargs)
            for seed in range(5):
                args = argparse.Namespace(root=root,policy='vpt',output=root/str(seed),seed=seed,limit_s=.05,stop_file=stop)
                with patch('minecraftrl.cli.inspect',return_value={'status':'ready_for_execution_attempt'}), \
                     patch('minecraftrl.cli.command',return_value={}), \
                     patch('minecraftrl.cli.subprocess.Popen',side_effect=launch), \
                     patch('minecraftrl.cli.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','no GPU in test')):
                    report = episode(args,{'name':'spontaneous','goal':None,'target_items':[]})
                self.assertFalse(stop.exists(), 'automatic deadline must not create user STOP')
                self.assertEqual(report['status'],'time_limit')
                self.assertTrue(report['execution_valid'])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process-tree ownership')
    def test_hard_timeout_kills_setsid_descendant_preserves_unrelated_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'configs').mkdir()
            (root/'configs/experiment.json').write_text(json.dumps({'startup_limit_s':10,'shutdown_grace_s':.05}))
            pid_file = root/'child.pid'; real_popen = subprocess.Popen
            unrelated = real_popen([sys.executable,'-c','import time; time.sleep(30)'])
            child_pid = None
            def launch(argv, **kwargs):
                # Arm the deadline only after the setsid child has installed its
                # handler. WSL drvfs Python startup can exceed the old 100ms limit.
                output=Path(argv[argv.index('--output')+1])
                child_code=('import signal,time,pathlib,os; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                            'pathlib.Path('+repr(str(pid_file))+').write_text(str(os.getpid())); time.sleep(30)')
                code = ('import subprocess,sys,time,pathlib,json\n'
                        'c=subprocess.Popen([sys.executable,"-c",'+repr(child_code)+'],start_new_session=True)\n'
                        'p=pathlib.Path('+repr(str(pid_file))+')\n'
                        'while not p.exists() or not p.read_text(): time.sleep(.005)\n'
                        'pathlib.Path('+repr(str(output/'events.jsonl'))+').write_text(json.dumps({"event":"episode_started","monotonic_s":time.monotonic()})+"\\n")\n'
                        'time.sleep(30)')
                return real_popen([sys.executable,'-c',code],**kwargs)
            try:
                args = argparse.Namespace(root=root,policy='vpt',output=root/'episode',seed=1,limit_s=.05,stop_file=root/'STOP')
                with patch('minecraftrl.cli.inspect',return_value={'status':'ready_for_execution_attempt'}), \
                     patch('minecraftrl.cli.command',return_value={}), \
                     patch('minecraftrl.cli.subprocess.Popen',side_effect=launch), \
                     patch('minecraftrl.cli.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','no GPU in test')):
                    report = episode(args,{'name':'spontaneous','goal':None,'target_items':[]})
                child_pid = int(pid_file.read_text())
                stat = Path('/proc')/str(child_pid)/'stat'
                alive = stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0] != 'Z'
                self.assertFalse(alive,'MineRL-style setsid descendant escaped worker group cleanup')
                self.assertIsNone(unrelated.poll(),'unrelated pre-existing child must survive')
                self.assertFalse(report['execution_valid'])
                self.assertEqual(report['process_cleanup']['remaining_live_pids'],[])
            finally:
                if child_pid is None and pid_file.exists(): child_pid = int(pid_file.read_text())
                if child_pid:
                    try: os.kill(child_pid,9)
                    except ProcessLookupError: pass
                unrelated.kill(); unrelated.wait()

if __name__ == '__main__': unittest.main()
