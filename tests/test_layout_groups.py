"""Layout-balanced groups (mixed-layout design 2026-11-13 §4 amendment)."""
import unittest

try:
    import torch
except ImportError:   # the host lane has no torch
    torch = None


@unittest.skipIf(torch is None, 'torch required')
class LayoutGroupsTest(unittest.TestCase):
    def test_run_layout_by_seed_and_round(self):
        from minecraftrl.layout_groups import run_layout
        self.assertEqual(run_layout('s-r6/coal_ore-s271828-n7'), 'wall')
        self.assertEqual(run_layout('s-final/coal_ore-s271828-n9'), 'high')
        self.assertEqual(run_layout('l-r8/sand-s314159-n10'), 'floor')
        self.assertEqual(run_layout('l-r8/clay-s161803-n10'), 'pillar')
        with self.assertRaises(ValueError): run_layout('official/s130363-x')

    def test_regroup_splits_each_block_into_wall_and_new_rows(self):
        from minecraftrl.layout_groups import regroup
        from minecraftrl.reacquire_training import group_weights
        data = dict(runs=['r6/coal_ore-s271828-n7', 'l7/coal_ore-s314159-n9', 'r6/sand-s161803-n7'],
                    blocks=['coal_ore', 'sand'], run=torch.tensor([0, 0, 0, 1, 2, 2]), block=torch.tensor([0, 0, 0, 0, 1, 1]),
                    stage=torch.tensor([2, 2, 2, 2, 2, 2]), labelled=torch.tensor([True] * 6))
        out = regroup(data)
        self.assertEqual(out['block'].tolist(), [0, 0, 0, 1, 2, 2])
        self.assertEqual(out['blocks'], ['coal_ore', 'coal_ore@new', 'sand', 'sand@new'])
        self.assertIs(out['stage'], data['stage']); self.assertEqual(data['block'].tolist(), [0, 0, 0, 0, 1, 1])
        w = group_weights(out['labelled'], out['stage'], out['block'], torch.arange(6))
        self.assertAlmostEqual(float(w[3]), 3 * float(w[0]), places=5)    # one new coal row weighs as the three wall rows


if __name__ == '__main__':
    unittest.main()
