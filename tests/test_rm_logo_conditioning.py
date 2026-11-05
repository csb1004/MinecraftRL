import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from minecraftrl.goal_contract import read_json
from minecraftrl.logo_wood import admit, acceptance_cases, prompts
from minecraftrl.rm_logo_conditioning import LogoConditioningPolicy, LogoEnvironment
from minecraftrl.rm_conditioning import validate_args
from minecraftrl.policies import Steve1Policy
import test_rm_conditioning as fixtures


class FakeGame:
    def __init__(self, servers):
        self.servers = iter(servers)
        self.raw = SimpleNamespace(action_space=SimpleNamespace(noop=lambda: {'attack':0}))
        self.actions = []
    def reset(self):
        pass
    def step(self, action):
        self.actions.append(action)
        return {'pov':None, 'inventory':{'oak_log':999}}, 99, False, {'workshop':next(self.servers)}


class LogoRuntimeTests(unittest.TestCase):
    def policy(self):
        obj = fixtures.RMConditioningTests().policy()
        obj.__class__ = LogoConditioningPolicy
        obj.mode, obj.prompts = 'logo-staged', prompts()
        obj.embeddings = {key:fixtures.FakeEmbedding(key) for key in prompts()}
        obj.bank_metadata = {'embeddings':{key:{'vector_sha256':key} for key in prompts()}}
        path = Path(__file__).resolve().parents[1]/'configs/logo-wood-validators-v1.json'
        obj.validators, obj.validator_review = admit(read_json(path))
        obj.validator_values = {}
        return obj

    def test_real_adapter_uses_generated_predicates_for_reward_and_stop(self):
        baseline = acceptance_cases()[0]['baseline']
        broke = dict(copy.deepcopy(baseline), server_tick=101, server_log_harvests=8, wood_v3_kind=2)
        acquired = dict(copy.deepcopy(broke), server_tick=102)
        acquired['server_inventory']['oak_log'] = 1
        game, policy = FakeGame([baseline, broke, acquired]), self.policy()
        env = LogoEnvironment(game, policy)
        obs = env.reset()
        self.assertEqual(obs['inventory']['oak_log'], 0)  # Not the fake client 999.
        with patch.object(Steve1Policy, 'act', fixtures.RMConditioningTests.infer):
            action = policy.act(None)
            self.assertEqual(env.step(action)[1], 0.)
            self.assertTrue(policy.rows[-1]['validators_after']['log_broken'])
            self.assertFalse(env.reward.completed)
            action = policy.act(None)
            self.assertEqual(env.step(action)[1], 1.)
        self.assertTrue(env.reward.completed)
        self.assertEqual(policy.selector.current, 'complete')
        self.assertTrue(policy.rows[-1]['terminal'])
        self.assertEqual(policy.rows[-1]['validator_tick_after'], 102)
        with patch.object(Steve1Policy, 'act') as infer:
            with self.assertRaises(InterruptedError):
                policy.act(None)
            infer.assert_not_called()
        with self.assertRaises(RuntimeError):
            env.step({'forward':1})
        self.assertEqual(len(game.actions), 3)  # Warmup plus exactly two dispatched actions.

    def test_lost_target_returns_to_search_without_clearing_recurrent_memory(self):
        policy = self.policy()
        for _ in range(3):
            policy.observe_stage('approach')
        self.assertEqual(policy.goal_key(), 'root')
        for _ in range(2):
            policy.observe_stage('search')
            self.assertEqual(policy.goal_key(), 'root')
        policy.observe_stage('search')
        self.assertEqual(policy.goal_key(), 'seek')
        self.assertEqual(policy.agent.hidden_state, 'initial')

    def test_stop_prevents_new_inference(self):
        policy = self.policy()
        policy.is_stopped = lambda: True
        with patch.object(Steve1Policy, 'act') as infer:
            with self.assertRaises(InterruptedError):
                policy.act(None)
            infer.assert_not_called()

    def test_fixed_ablation_uses_same_root_for_all_nonterminal_stages(self):
        policy = self.policy()
        policy.mode = 'logo-fixed'
        for stage in ('search','approach','mine','collect','search'):
            for _ in range(3):
                policy.observe_stage(stage)
            self.assertEqual(policy.goal_key(), 'root')
        policy.observe_stage('complete')
        with self.assertRaises(InterruptedError):
            policy.goal_key()

    def test_cli_accepts_only_isolated_new_mode(self):
        root = Path.cwd()
        args = SimpleNamespace(rm_conditioning='logo-staged', rm_goal_bank=root/'results/logo-bank',
                               policy='steve1', condition='wood', operation='episode', root=root)
        validate_args(args)
        args.rm_conditioning = 'logo-fixed'
        validate_args(args)
        args.rl_collect = True
        with self.assertRaises(ValueError):
            validate_args(args)


if __name__ == '__main__':
    unittest.main()
