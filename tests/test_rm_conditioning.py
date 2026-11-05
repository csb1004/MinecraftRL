import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from minecraftrl.rm_conditioning import ConditioningPolicy, StageSelector, validate_args, PROMPTS
from minecraftrl.policies import Steve1Policy


class FakeEmbedding:
    def __init__(self, key):
        self.key = key
    def copy(self):
        return FakeEmbedding(self.key)


class RMConditioningTests(unittest.TestCase):
    def policy(self, mode='staged'):
        obj = object.__new__(ConditioningPolicy)
        obj.mode = mode
        obj.prompts = PROMPTS
        obj.embeddings = {key:FakeEmbedding(key) for key in ('root','search','approach','mine','collect')}
        obj.bank_metadata = {'embeddings':{key:{'vector_sha256':key} for key in obj.embeddings}}
        obj.selector = StageSelector()
        obj.observe_stage('search')
        obj.agent = SimpleNamespace(hidden_state='initial')
        obj.pending = obj.pending_state = obj.last_key = None
        obj.rows, obj.switches = [], []
        obj.is_stopped = lambda: False
        return obj

    @staticmethod
    def infer(policy, screen):
        policy.agent.hidden_state = 'next-'+policy.goal_embed.key
        return {'forward':1}

    def test_stage_noise_does_not_switch_and_acceptance_is_immediate(self):
        state = StageSelector()
        self.assertEqual(state.observe('search'), 'search')
        for event in ('approach','approach','search','approach','approach'):
            self.assertEqual(state.observe(event), 'search')
        self.assertEqual(state.observe('approach'), 'approach')
        self.assertEqual(state.observe('complete'), 'complete')
        with self.assertRaises(RuntimeError):
            state.observe('search')

    def test_goal_switch_preserves_memory_until_actual_dispatch(self):
        policy = self.policy()
        with patch.object(Steve1Policy, 'act', self.infer):
            action = policy.act(None)
            self.assertEqual(policy.agent.hidden_state, 'initial')
            policy.commit(action, 0., False)
            self.assertEqual(policy.agent.hidden_state, 'next-search')
            for _ in range(3):
                policy.observe_stage('approach')
            action = policy.act(None)
            self.assertEqual(policy.agent.hidden_state, 'next-search')
            policy.commit(action, 0., False)
        self.assertEqual([x['goal_key'] for x in policy.rows], ['search','approach'])
        self.assertEqual(policy.agent.hidden_state, 'next-approach')

    def test_fixed_condition_ignores_stage_for_embedding_but_records_stage(self):
        policy = self.policy('fixed')
        for _ in range(3):
            policy.observe_stage('mine')
        with patch.object(Steve1Policy, 'act', self.infer):
            action = policy.act(None)
            policy.commit(action, 0., False)
        self.assertEqual(policy.rows[0]['goal_key'], 'root')
        self.assertEqual(policy.rows[0]['stage'], 'mine')

    def test_stop_and_completed_rm_prevent_inference(self):
        for complete in (False, True):
            policy = self.policy()
            if complete:
                policy.observe_stage('complete')
            else:
                policy.is_stopped = lambda: True
            with patch.object(Steve1Policy, 'act') as infer:
                with self.assertRaises(InterruptedError):
                    policy.act(None)
                infer.assert_not_called()

    def test_dispatch_mismatch_is_rejected_without_committing_memory(self):
        policy = self.policy()
        with patch.object(Steve1Policy, 'act', self.infer):
            policy.act(None)
        with self.assertRaises(ValueError):
            policy.commit({'forward':0}, 0., False)
        self.assertEqual(policy.agent.hidden_state, 'initial')
        self.assertEqual(policy.rows, [])

    def test_invalid_observation_cannot_select_a_goal(self):
        state = StageSelector()
        with self.assertRaises(ValueError):
            state.observe('unknown')

    def test_cli_forbids_mixing_legacy_or_persistent_controllers(self):
        root = Path.cwd()
        args = SimpleNamespace(rm_conditioning='staged', rm_goal_bank=root/'results/bank',
                               policy='steve1', condition='wood', operation='episode', root=root)
        validate_args(args)
        for attr in ('rl_collect','task_session','world_slot','craft_probe'):
            setattr(args, attr, True)
            with self.assertRaises(ValueError):
                validate_args(args)
            setattr(args, attr, False)


if __name__ == '__main__':
    unittest.main()
