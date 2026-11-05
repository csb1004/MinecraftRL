import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch

from minecraftrl.cli import parser,validate_controlled_start_args
from minecraftrl.rl_collect import RLEnvironment


class StartupIntegrationTests(unittest.TestCase):
    def make(self,controlled=True,stop=lambda:False):
        env=Mock();env.startup_control={'request_sha256':'fixed'} if controlled else None
        env.raw.action_space.noop.return_value={'neutral':True}
        count=[20]
        def step(action):
            count[0]+=1
            return {'pov':None},0,False,{'workshop':dict(server_available=1,error=0,
                server_tick=count[0]+2,client_tick=count[0],world_time=count[0]+46,server_inventory={})}
        env.step.side_effect=step
        policy=SimpleNamespace(commit=Mock())
        wrapper=RLEnvironment(env,policy,strict_ticks=True,is_stopped=stop)
        return env,policy,wrapper

    def test_missing_restore_stops_before_reward_or_policy_control(self):
        env,policy,wrapper=self.make()
        with patch('minecraftrl.rl_collect.verify_startup_receipt',side_effect=ValueError('missing')):
            with self.assertRaisesRegex(ValueError,'missing'):wrapper.reset()
        self.assertEqual(env.step.call_count,60)
        self.assertIsNone(wrapper.reward)
        policy.commit.assert_not_called()

    def test_first_step_must_verify_before_committing_experience(self):
        env,policy,wrapper=self.make()
        with patch('minecraftrl.rl_collect.verify_startup_receipt',side_effect=[{'verified':True},ValueError('wrong rule')]) as verify:
            wrapper.reset()
            with self.assertRaisesRegex(ValueError,'wrong rule'):wrapper.step({'forward':1})
            self.assertEqual(verify.call_args.kwargs,{'first_step':True})
        policy.commit.assert_not_called()
        self.assertEqual(env.step.call_count,61)

    def test_manual_stop_prevents_next_neutral_step_and_receipt_credit(self):
        polls=[0]
        def stop():
            polls[0]+=1
            return polls[0]>3
        env,policy,wrapper=self.make(stop=stop)
        with patch('minecraftrl.rl_collect.verify_startup_receipt') as verify:
            with self.assertRaises(InterruptedError):wrapper.reset()
            verify.assert_not_called()
        self.assertEqual(env.step.call_count,3)
        policy.commit.assert_not_called()

    def test_default_start_does_not_require_receipts(self):
        env,policy,wrapper=self.make(controlled=False)
        with patch('minecraftrl.rl_collect.verify_startup_receipt') as verify:
            wrapper.reset();wrapper.step({'forward':1});verify.assert_not_called()
        policy.commit.assert_called_once()

    def test_cli_requires_explicit_evaluation_and_never_final(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol=Path(temp)/'trial.json'
            args=parser().parse_args(['episode','--policy','steve1','--rl-collect','--rl-controlled-start','--rl-world',str(Path(temp)/'world.zip'),
                '--rl-navigation-trial',str(protocol)])
            for mode,split in [('evaluate','train'),('evaluate','development'),('ppo','train'),('evaluate','final')]:
                protocol.write_text(json.dumps(dict(mode=mode,split=split)))
                if mode=='evaluate' and split!='final':validate_controlled_start_args(args)
                else:
                    with self.assertRaises(SystemExit):validate_controlled_start_args(args)
            args.rl_navigation_trial=None
            with self.assertRaises(SystemExit):validate_controlled_start_args(args)


if __name__=='__main__':unittest.main()
