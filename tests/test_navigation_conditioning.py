import copy
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Torch unavailable')
class NavigationConditioningTests(unittest.TestCase):
    def setUp(self):
        from minecraftrl import navigation_model as model
        from minecraftrl.waypoint_actor import navigation_head
        self.model = model
        torch.set_num_threads(1)
        torch.manual_seed(17501)
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buttons = torch.nn.Linear(2054, 7)
                self.camera = torch.nn.Linear(2054, 5)
            def forward(self, x):
                return {k: getattr(self, k)(x) for k in ('buttons', 'camera')}
        self.parent = Head()
        self.prior = navigation_head(self.parent)
        self.x = torch.randn(4, 8, 2064)
        self.x[..., -8] = torch.rand(4, 8)

    def head(self, kind):
        value = self.model.refinement_head(self.prior, kind)
        with torch.no_grad():
            for module in value.modules():
                if isinstance(module, self.model.RefinedLinear):
                    last = module.residual if kind == 'linear' else module.residual[-1]
                    last.weight.normal_(0, .1)
                    if last.bias is not None:
                        last.bias.normal_(0, .1)
        return value

    def test_migration_preserves_trained_function_frozen_prior_and_roundtrip(self):
        for kind in ('linear', 'mlp'):
            old = self.head(kind)
            new = copy.deepcopy(old)
            self.model.rescale_distance_input(new, 16)
            # Linear GEMM changes its input layout; Torch 2.0 differs by one float32 ULP.
            compare = torch.equal if kind == 'mlp' else lambda a,b: torch.allclose(a,b,rtol=0,atol=1e-6)
            self.assertTrue(all(compare(v, new(self.x)[k]) for k, v in old(self.x).items()))
            for k, v in old.state_dict().items():
                if '.base.' in k:
                    self.assertTrue(torch.equal(v, new.state_dict()[k]))
            self.model.rescale_distance_input(new, 1)
            self.assertTrue(all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items()))

    def test_checkpoint_reload_preserves_scale_and_legacy_defaults(self):
        for scale in (1, 16):
            head = self.head('mlp')
            self.model.rescale_distance_input(head, scale)
            saved = dict(schema='navigation_refinement_v1', kind='mlp', actor=head.state_dict())
            if scale != 1:
                saved['schema'] = 'navigation_refinement_v2'
                saved['residual_distance_scale'] = scale
            restored, refined = self.model.load_navigation(self.parent, saved)
            self.assertTrue(refined)
            self.assertTrue(all(torch.equal(v, restored(self.x)[k]) for k, v in head(self.x).items()))
            self.assertEqual({m.distance_scale for m in restored.modules()
                              if isinstance(m, self.model.RefinedLinear)}, {scale})

    def test_scaled_checkpoint_requires_explicit_versioned_input_contract(self):
        head = self.head('mlp')
        saved = dict(schema='navigation_refinement_v1', kind='mlp', actor=head.state_dict(),
                     residual_distance_scale=16)
        with self.assertRaises(ValueError):
            self.model.load_navigation(self.parent, saved)
        saved['schema'] = 'navigation_refinement_v2'
        del saved['residual_distance_scale']
        with self.assertRaises(ValueError):
            self.model.load_navigation(self.parent, saved)

    def test_only_distance_weight_gradient_is_rescaled(self):
        old = self.head('mlp')
        new = copy.deepcopy(old)
        self.model.rescale_distance_input(new, 16)
        old.requires_grad_(True)
        new.requires_grad_(True)
        sum(v.sum() for v in old(self.x).values()).backward()
        sum(v.sum() for v in new(self.x).values()).backward()
        modules = lambda head: [m for m in head.modules() if isinstance(m, self.model.RefinedLinear)]
        for a, b in zip(modules(old), modules(new)):
            first, second = a.residual[0].weight.grad, b.residual[0].weight.grad
            self.assertTrue(torch.equal(first[:, 18] * 16, second[:, 18]))
            self.assertTrue(torch.equal(first[:, :18], second[:, :18]))
            self.assertTrue(torch.equal(first[:, 19:], second[:, 19:]))

    def test_invalid_or_inconsistent_scales_fail_before_mutation(self):
        head = self.head('mlp')
        before = copy.deepcopy(head.state_dict())
        for value in (0, -1, 65, True, float('nan'), float('inf'), '16'):
            with self.assertRaises(ValueError):
                self.model.rescale_distance_input(head, value)
        self.assertTrue(all(torch.equal(v, head.state_dict()[k]) for k, v in before.items()))
        next(m for m in head.modules() if isinstance(m, self.model.RefinedLinear)).distance_scale = 16
        with self.assertRaises(ValueError):
            self.model.rescale_distance_input(head, 1)
        self.assertTrue(all(torch.equal(v, head.state_dict()[k]) for k, v in before.items()))


if __name__ == '__main__':
    unittest.main()
