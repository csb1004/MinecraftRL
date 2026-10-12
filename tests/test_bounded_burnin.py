import copy
import sys
import unittest
from pathlib import Path

import torch

from minecraftrl.bounded_burnin import cache_length, forward_window_bounded
from minecraftrl.temporal_bc import forward_window


class BoundedBurninTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'external/steve1'))
        try:
            from steve1.VPT.lib.util import ResidualRecurrentBlock
            from steve1.VPT.lib.torch_util import set_default_torch_device
        except ImportError as exc:
            raise unittest.SkipTest('official VPT dependencies unavailable: '+str(exc))
        set_default_torch_device('cpu')
        cls.Block = ResidualRecurrentBlock

    def fixture(self):
        torch.manual_seed(9610)
        block = self.Block(hidsize=8, timesteps=4, recurrence_type='transformer',
                           attention_memory_size=12, attention_heads=2).eval()
        tail = torch.nn.Linear(8, 8)
        rows = [dict(temporal_input=torch.randn(2,1,8,requires_grad=True), features=torch.zeros(2,1,14))
                for _ in range(37)]
        return block, tail, rows, block.r.initial_state(2)

    def test_full_and_bounded_outputs_at_cache_boundaries(self):
        block, tail, rows, initial = self.fixture()
        self.assertEqual(cache_length(block), 8)
        for start in (0, 1, 7, 8, 9, 16, 29):
            with self.subTest(start=start), torch.no_grad():
                full = forward_window(block, tail, rows, initial, start, start+4)
                bounded = forward_window_bounded(block, tail, rows, initial, start, start+4)
                self.assertTrue(torch.equal(full, bounded))

    def test_current_weight_replay_after_parameter_change(self):
        block, tail, rows, initial = self.fixture()
        for _ in range(2):
            with torch.no_grad():
                for parameter in block.parameters(): parameter.add_(torch.randn_like(parameter)*.01)
                full = forward_window(block, tail, rows, initial, 25, 33)
                bounded = forward_window_bounded(block, tail, rows, initial, 25, 33)
            self.assertTrue(torch.equal(full, bounded))

    def test_parameter_and_input_gradients_match_without_burnin_gradients(self):
        block, tail, rows, initial = self.fixture()
        other_block, other_tail, other_rows = copy.deepcopy(block), copy.deepcopy(tail), copy.deepcopy(rows)
        forward_window(block, tail, rows, initial, 25, 33)[:, -1, :8].square().sum().backward()
        forward_window_bounded(other_block, other_tail, other_rows, initial, 25, 33)[:, -1, :8].square().sum().backward()
        for a, b in zip(list(block.parameters())+list(tail.parameters()), list(other_block.parameters())+list(other_tail.parameters())):
            if a.grad is None: self.assertIsNone(b.grad)
            else: self.assertTrue(torch.equal(a.grad, b.grad))
        for i, (a, b) in enumerate(zip(rows, other_rows)):
            x, y = a['temporal_input'].grad, b['temporal_input'].grad
            if i < 25 or i >= 33:
                self.assertIsNone(x); self.assertIsNone(y)
            else:
                self.assertTrue(torch.equal(x, y))
        self.assertGreater(float(rows[25]['temporal_input'].grad.abs().sum()), 0)

    def test_other_recurrence_and_mask_cannot_silently_use_short_history(self):
        block, tail, rows, initial = self.fixture()
        for mutation in ('type', 'mask', 'length'):
            other = copy.deepcopy(block)
            if mutation == 'type': other.recurrence_type = 'multi_layer_lstm'
            elif mutation == 'mask': other.r.mask = 'none'
            else: other.r.orc_block.cache_keep_len = 7
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                forward_window_bounded(other, tail, rows, initial, 25, 33)

    def test_stop_interrupts_burnin_without_an_output(self):
        block, tail, rows, initial = self.fixture()
        calls = []
        def stop():
            calls.append(1)
            if len(calls) == 3: raise InterruptedError('STOP')
        with self.assertRaises(InterruptedError):
            forward_window_bounded(block, tail, rows, initial, 25, 33, check_stop=stop)
        self.assertEqual(len(calls), 3)


if __name__ == '__main__': unittest.main()
