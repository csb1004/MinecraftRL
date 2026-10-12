import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from minecraftrl.cli import parser
from minecraftrl.policies import MineRLEnvironment


class RecoveryCollectionTests(unittest.TestCase):
    def test_sampling_seed_is_separate_from_world_and_defaults_to_legacy(self):
        from minecraftrl.cli import resolve_rl_sampling_seed
        base=['episode','--policy','steve1','--seed','104743']
        args=parser().parse_args(base+['--rl-sampling-seed','701'])
        self.assertEqual(args.seed,104743)
        self.assertEqual(resolve_rl_sampling_seed(args),701)
        self.assertEqual(resolve_rl_sampling_seed(parser().parse_args(base)),104743)

    def test_sampling_seed_rejects_outside_rng_range(self):
        for value in ('-1','4294967296'):
            with self.assertRaises(SystemExit):parser().parse_args(['episode','--policy','steve1','--rl-sampling-seed',value])

    def release(self,count):
        from minecraftrl.rl_collect import RLEnvironment
        class Raw:
            action_space=SimpleNamespace(noop=lambda:{'attack':0})
            def reset(self):return {}
            def step(self,action):
                return {},0,False,{'workshop':{'server_available':1,'error':'','held_key_count':count,'server_tick':42,'server_alive':1}}
        raw=MineRLEnvironment(Raw());raw.reset()
        with tempfile.TemporaryDirectory() as folder:
            wrapped=RLEnvironment(raw,SimpleNamespace(output=Path(folder)))
            if count!=0:
                with self.assertRaisesRegex(RuntimeError,'release'):wrapped.release()
            else:self.assertEqual(wrapped.release(),'neutral_command_sent')
            result=json.loads((Path(folder)/'input-release-state.json').read_text())
        return result

    def test_neutral_release_records_post_step_server_keys(self):
        result=self.release(0)
        self.assertTrue(result['verified']);self.assertEqual(result['held_key_count'],0)
        self.assertEqual(result['server_tick'],42)

    def test_held_or_missing_keys_cannot_claim_verified_release(self):
        self.assertFalse(self.release(1)['verified'])
        self.assertFalse(self.release(None)['verified'])
