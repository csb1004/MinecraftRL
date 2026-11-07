import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # the host lane has no torch
    torch = None


class FakeHead(torch.nn.Module if torch else object):
    """Stands in for the VPT dict action head: forward(x, mask=None) -> {key: log-probs}."""

    def __init__(self, width):
        super().__init__()
        torch.manual_seed(3)
        self.buttons = torch.nn.Linear(width, 5); self.camera = torch.nn.Linear(width, 3)

    def forward(self, x, mask=None):
        return {'buttons': torch.log_softmax(self.buttons(x).float(), -1),
                'camera': torch.log_softmax(self.camera(x).float(), -1)}

    def sample(self, pd, deterministic=False): return {k: v.argmax(-1) for k, v in pd.items()}

    def logprob(self, action, pd): return sum(pd[k].gather(-1, action[k].unsqueeze(-1)).squeeze(-1) for k in pd)


@unittest.skipIf(torch is None, 'torch required')
class ReacquireHeadTest(unittest.TestCase):
    def setUp(self):
        from minecraftrl.reacquire_head import ReacquireHead
        self.Head = ReacquireHead
        self.base = FakeHead(8).requires_grad_(False)
        self.x = torch.randn(2, 1, 8, generator=torch.Generator().manual_seed(1))
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)

    def trained(self, use_stage):
        head = self.Head(self.base, 8, use_stage=use_stage)
        with torch.no_grad():
            for p in head.adapter.parameters(): p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(2)))
        return head

    def test_zero_initialized_head_returns_the_base_output_exactly(self):
        head = self.Head(self.base, 8, use_stage=True); head.stage = 2
        out, ref = head(self.x), self.base(self.x)
        for k in ref: self.assertTrue(torch.equal(out[k], ref[k]))

    def test_stage_is_ignored_without_stage_input(self):
        head = self.trained(use_stage=False)
        head.stage = 0; a = head(self.x); head.stage = 4; b = head(self.x)
        for k in a: self.assertTrue(torch.equal(a[k], b[k]))

    def test_stage_changes_output_with_stage_input(self):
        head = self.trained(use_stage=True)
        head.stage = 0; a = head(self.x); head.stage = 4; b = head(self.x)
        self.assertFalse(all(torch.equal(a[k], b[k]) for k in a))

    def test_per_row_stage_override_matches_the_scalar_stage(self):
        from minecraftrl.reacquire_head import stage_onehot
        head = self.trained(use_stage=True)
        head.stage = 3; a = head(self.x)
        onehot = torch.tensor(stage_onehot(3, True)).view(1, 1, -1).expand(2, 1, -1)
        head.stage = None; b = head(self.x, stage_onehot=onehot)
        for k in a: self.assertTrue(torch.allclose(a[k], b[k]))

    def test_last_latent_is_the_input_latent(self):
        head = self.Head(self.base, 8, use_stage=True); head(self.x)
        self.assertTrue(torch.equal(head.last_latent, self.x))

    def test_checkpoint_roundtrip_and_mismatch(self):
        from minecraftrl.reacquire_head import load_into, save_checkpoint
        trained = self.trained(use_stage=True); path = Path(self.tmp.name) / 'c.pt'
        sha = save_checkpoint(trained, path, base_weights_sha256='b', metadata={'note': 1})
        self.assertEqual(len(sha), 64)
        again = load_into(self.Head(self.base, 8, use_stage=True), path, base_weights_sha256='b')
        trained.stage = again.stage = 1
        for k, v in trained(self.x).items(): self.assertTrue(torch.equal(again(self.x)[k], v))
        with self.assertRaises(ValueError): load_into(self.Head(self.base, 8, use_stage=False), path, base_weights_sha256='b')
        with self.assertRaises(ValueError): load_into(self.Head(self.base, 8, use_stage=True), path, base_weights_sha256='c')

    def test_initialization_does_not_consume_the_global_rng(self):
        torch.manual_seed(5); expected = torch.rand(1)
        torch.manual_seed(5); self.Head(self.base, 8, use_stage=True)
        self.assertTrue(torch.equal(torch.rand(1), expected))

    def test_sample_and_logprob_go_to_the_base(self):
        head = self.Head(self.base, 8, use_stage=True); pd = head(self.x)
        action = head.sample(pd)
        self.assertTrue(torch.equal(head.logprob(action, pd), self.base.logprob(action, pd)))


if __name__ == '__main__':
    unittest.main()
