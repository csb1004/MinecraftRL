import copy
import unittest
from collections import Counter

import torch

from minecraftrl.onset_bc import OnsetMetrics, correction_schedule


class OnsetBCTests(unittest.TestCase):
    def groups(self):
        groups = {}
        for name, first in [('a', 10), ('b', 13), ('dev', 9)]:
            groups[name] = [dict(frame=i, loss_eligible=i >= first,
                                owner='teacher' if i >= first else 'student',
                                action={'buttons': i} if i >= first else None)
                            for i in range(first+17)]
        return groups, dict(a='train', b='train', dev='development')

    def test_matched_updates_and_complete_teacher_coverage(self):
        groups, splits = self.groups()
        uniform = correction_schedule(groups, splits, 'uniform')
        onset = correction_schedule(groups, splits, 'onset')
        self.assertEqual(len(uniform), len(onset))
        for schedule in (uniform, onset):
            covered = {(u['session'], i) for u in schedule for i in range(u['start'], u['end'])
                       if groups[u['session']][i]['loss_eligible']}
            expected = {(name, i) for name, rows in groups.items() if splits[name] == 'train'
                        for i, row in enumerate(rows) if row['loss_eligible']}
            self.assertEqual(covered, expected)
            self.assertFalse(any(u['session'] == 'dev' for u in schedule))

    def test_onset_has_exactly_one_genuine_target_and_seven_context_frames(self):
        groups, splits = self.groups()
        for unit in correction_schedule(groups, splits, 'onset'):
            if unit['kind'] != 'onset':
                continue
            rows = groups[unit['session']][unit['start']:unit['end']]
            self.assertEqual(len(rows), 8)
            self.assertEqual([r['loss_eligible'] for r in rows], [False]*7+[True])
            self.assertTrue(all(r['action'] is None for r in rows[:-1]))

    def test_deterministic_balanced_and_does_not_mutate_source(self):
        groups, splits = self.groups()
        before = copy.deepcopy(groups)
        a = correction_schedule(groups, splits, 'onset')
        self.assertEqual(a, correction_schedule(groups, splits, 'onset'))
        self.assertEqual(before, groups)
        counts = Counter(u['session'] for u in a if u['kind'] == 'onset')
        self.assertLessEqual(max(counts.values())-min(counts.values()), 1)
        a[0]['start'] = -1
        self.assertGreaterEqual(correction_schedule(groups, splits, 'onset')[0]['start'], 0)

    def test_invalid_target_and_split_are_rejected(self):
        for mutation in ('student_target', 'missing_teacher', 'eval_split', 'short_prefix'):
            groups, splits = self.groups()
            if mutation == 'student_target': groups['a'][0]['action'] = {'buttons': 0}
            elif mutation == 'missing_teacher': groups['a'][-1]['loss_eligible'] = False
            elif mutation == 'eval_split': splits['dev'] = 'eval'
            else: groups['a'] = groups['a'][5:]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                correction_schedule(groups, splits, 'onset')

    def test_release_attack_camera_joint_uses_same_button_category(self):
        metric = OnsetMetrics([[0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,0]],
                              [False,False,True], [[0,0],[0,1],[0,2]],
                              ['forward','back','left','right','attack'], 1, 'cpu')
        pd = {'buttons': torch.tensor([.2,.7,.1]).log(), 'camera': torch.tensor([.1,.3,.6]).log()}
        target = {'buttons':torch.tensor(0), 'camera':torch.tensor(2)}
        result = metric.measure(pd,target)
        self.assertAlmostEqual(result['p_release_attack_and_turn'], .2*.7, places=6)
        self.assertAlmostEqual(result['p_release_attack_and_target_yaw'], .2*.6, places=6)
        self.assertAlmostEqual(result['p_target_yaw_sign'], .9*.6, places=6)


if __name__ == '__main__': unittest.main()
