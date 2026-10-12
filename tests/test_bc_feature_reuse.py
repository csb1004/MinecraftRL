import copy
import unittest

import torch

from minecraftrl.bc_feature_reuse import checked_reuse
from minecraftrl.correction_bc import correction_owner


class FeatureReuseTests(unittest.TestCase):
    def fixture(self):
        sessions, rows = [], []
        for split, seed in [('train', 104743), ('development', 104761)]:
            frames = [dict(path=f'frames/{i:06d}.png', sha256='a'*64, loss_eligible=i >= 8) for i in range(10)]
            s = dict(id=split, split=split, seed=seed, context_start=0, start=0, end=10,
                eligible_for_future_masked_bc=True, expert_labels=2, frames=frames, frame_sha256=['a'*64]*10,
                handoffs=[dict(step=8, verified=True, held_key_count=0, teacher_started=True,
                               trigger=dict(step=7, event='stalled'))])
            sessions.append(s)
            for i in range(10):
                rows.append(dict(session=split, split=split, frame=i, owner=correction_owner(s, i), loss_eligible=i >= 8,
                    action={'buttons': torch.tensor([[[i]]])} if i >= 8 else None,
                    features=torch.zeros(2, 1, 9), temporal_input=torch.zeros(2, 1, 3), tail_input=torch.zeros(2, 1, 3)))
        plan = dict(schema='student_correction_v1', data_source='student_correction', masked_loss_required=True,
            goal='collect wood', frame_offset=0, rm_context='recorded', parent='parent.pt', checkpoint='parent.pt',
            reward_mode='wood-potential-v4', sessions=sessions)
        data = dict(bc_schema=2, plan=plan, rows=rows, parent_sha256='p', model_sha256='m', goal_embedding_sha256='e',
                    tail_initial={}, temporal_initial={}, temporal_initial_memory=[])
        return data, copy.deepcopy(plan)

    def check(self, data, plan):
        return checked_reuse(data, plan, parent_sha256='p', model_sha256='m', embedding_sha256='e')

    def test_complete_old_sessions_are_reused_without_new_session_labels(self):
        data, plan = self.fixture()
        new = copy.deepcopy(plan['sessions'][0]); new.update(id='new', seed=104759)
        plan['sessions'].append(new)
        groups = self.check(data, plan)
        self.assertEqual(set(groups), {'train', 'development'})
        self.assertIs(groups['train'][0], data['rows'][0])
        self.assertIsNone(groups['train'][0]['action'])

    def test_parent_model_embedding_and_goal_mismatch_are_rejected(self):
        for key in ('parent_sha256', 'model_sha256', 'goal_embedding_sha256', 'goal'):
            data, plan = self.fixture()
            if key == 'goal': plan['goal'] = 'collect dirt'
            else: data[key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(data, plan)

    def test_changed_session_receipt_cannot_reuse_stale_features(self):
        data, plan = self.fixture()
        plan['sessions'][0]['frames'][0]['sha256'] = 'b'*64
        plan['sessions'][0]['frame_sha256'][0] = 'b'*64
        with self.assertRaises(ValueError): self.check(data, plan)

    def test_incomplete_or_unmasked_cache_is_rejected(self):
        for kind in ('frame', 'target', 'cache', 'shape', 'nan'):
            data, plan = self.fixture()
            if kind == 'frame': data['rows'].pop(0)
            elif kind == 'target': data['rows'][0]['action'] = {'buttons': torch.zeros(1, 1, 1)}
            elif kind == 'cache': data.pop('temporal_initial_memory')
            elif kind == 'shape': data['rows'][0]['tail_input'] = torch.zeros(1)
            else: data['rows'][0]['tail_input'].fill_(float('nan'))
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.check(data, plan)


if __name__ == '__main__': unittest.main()
