import unittest
from unittest.mock import patch

from minecraftrl.rm_conditioning import StageSelector
from minecraftrl.rm_purpose_conditioning import (PurposeConditioningPolicy, candidate, prompts,
    next_phase, condition_key, checked_tokens)
from minecraftrl.policies import Steve1Policy
import test_rm_conditioning as fixtures


class PurposeTests(unittest.TestCase):
    def test_contracts_keep_target_and_purpose_in_search(self):
        self.assertIn('find a tree.', prompts()['seek'])
        for key in ('seek', 'gather'):
            self.assertIn('Purpose: collect at least one wood log.', prompts()[key])
            self.assertIn('Expected result:', prompts()[key])
        self.assertEqual(candidate('purpose-fixed')['state_bindings'],
                         dict(seeking='gather', gathering='gather'))
        self.assertFalse(candidate('purpose-switch')['capability_verified'])

    def test_noise_and_lost_target_do_not_oscillate_condition(self):
        selector, phase = StageSelector(), 'seeking'
        keys = []
        for stage in ('search','approach','search','approach','approach','approach','search','search','search'):
            phase = next_phase(phase, selector.observe(stage))
            keys.append(condition_key('purpose-switch', phase))
        self.assertEqual(keys, ['seek']*5 + ['gather']*4)
        self.assertEqual(next_phase(phase, 'complete'), 'complete')
        with self.assertRaises(InterruptedError):
            condition_key('purpose-switch', 'complete')

    def test_new_modes_commit_actual_goal_and_memory_and_honor_stop(self):
        for mode, key in (('official','root'),('purpose-fixed','gather'),('purpose-switch','seek')):
            obj = fixtures.RMConditioningTests().policy()
            obj.__class__ = PurposeConditioningPolicy
            obj.mode, obj.phase, obj.prompts = mode, 'seeking', prompts()
            obj.embeddings = {key:fixtures.FakeEmbedding(key) for key in prompts()}
            obj.bank_metadata = {'embeddings':{k:{'vector_sha256':k} for k in prompts()}}
            with patch.object(Steve1Policy, 'act', fixtures.RMConditioningTests.infer):
                action = obj.act(None)
                self.assertEqual(obj.agent.hidden_state, 'initial')
                obj.commit(action, 0, False)
            self.assertEqual(obj.rows[0]['goal_key'], key)
            self.assertEqual(obj.rows[0]['prompt'], prompts()[key])
            self.assertEqual(obj.agent.hidden_state, 'next-'+key)
            obj.observe_stage('complete')
            with patch.object(Steve1Policy, 'act') as infer:
                with self.assertRaises(InterruptedError):
                    obj.act(None)
                infer.assert_not_called()

    def test_overflow_rejected_before_truncating_encoder(self):
        tokenizer = lambda *a, **k: {'input_ids':list(range(76))}
        tokenizer.bos_token_id, tokenizer.eos_token_id = 49406, 49407
        with patch('builtins.print') as encoder:
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                checked_tokens('long', tokenizer, encoder)
            encoder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
