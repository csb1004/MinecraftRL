import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # the host lane has no torch
    torch = None

WIDTH = 8


class FakeBase(torch.nn.Module if torch else object):
    """VPT-like dict head: log-probs shaped [B, T, 1, n] per key."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(4)
        self.buttons = torch.nn.Linear(WIDTH, 5); self.camera = torch.nn.Linear(WIDTH, 3)
        with torch.no_grad():
            for p in self.parameters(): p.copy_(torch.randn(p.shape, generator=g) * .1)

    def forward(self, x, mask=None):
        return {'buttons': torch.log_softmax(self.buttons(x).float(), -1).unsqueeze(-2),
                'camera': torch.log_softmax(self.camera(x).float(), -1).unsqueeze(-2)}


def fake_factory(use_stage, width):
    from minecraftrl.reacquire_head import ReacquireHead
    return ReacquireHead(FakeBase().requires_grad_(False), width, use_stage=use_stage), 'base-sha'


def toy_dataset(path, *, nan_holdout=False, rows_per_run=64):
    """Two runs (the second is the holdout). Near rows (stage 2) are labelled 'button 1, camera 2'; others unlabelled."""
    g = torch.Generator().manual_seed(7); n = 2 * rows_per_run
    stage = torch.tensor([2 if i % 2 else 0 for i in range(n)], dtype=torch.int8)
    labelled = stage == 2
    data = dict(cond=torch.randn(n, WIDTH, generator=g).half(), uncond=torch.randn(n, WIDTH, generator=g).half(),
                stage=stage, labelled=labelled,
                label_buttons=torch.where(labelled, torch.tensor(1), torch.tensor(-1)).long(),
                label_camera=torch.where(labelled, torch.tensor(2), torch.tensor(-1)).long(),
                camera_on=labelled.clone(), in_view=torch.zeros(n, dtype=torch.bool),
                label_pitch=torch.zeros(n), label_yaw=torch.zeros(n), step=torch.arange(n, dtype=torch.int32),
                run=torch.tensor([0] * rows_per_run + [1] * rows_per_run), runs=['r0/a', 'r0/b'], holdout=1,
                cond_scale=6., width=WIDTH, stages=['far', 'closing', 'near', 'aimed', 'mining', 'collect', 'gui'])
    if nan_holdout: data['cond'][rows_per_run:] = float('nan')
    torch.save(data, path)
    return path


@unittest.skipIf(torch is None, 'torch required')
class GuidedLossTest(unittest.TestCase):
    def setUp(self):
        from minecraftrl import reacquire_training as rt
        self.rt = rt
        self.head, _ = fake_factory(True, WIDTH)
        g = torch.Generator().manual_seed(1)
        self.cond, self.uncond = torch.randn(6, WIDTH, generator=g), torch.randn(6, WIDTH, generator=g)
        self.stage = torch.tensor([0, 2, 2, 3, -1, 2], dtype=torch.int8)

    def test_guided_logp_is_normalized(self):
        logp = self.rt.guided_logp(self.head, self.cond, self.uncond, self.stage, 6.)
        for v in logp.values():
            self.assertEqual(v.shape[0], 6)
            self.assertTrue(torch.allclose(v.exp().sum(-1), torch.ones(v.shape[:-1]), atol=1e-5))

    def test_uniform_guided_distribution_has_log_n_nll(self):
        logp = {'buttons': torch.full((1, 4), -math.log(4)), 'camera': torch.full((1, 3), -math.log(3))}
        nll = self.rt.bc_nll(logp, torch.tensor([1]), torch.tensor([0]), torch.tensor([False]))
        self.assertAlmostEqual(nll.item(), math.log(4), places=6)
        nll = self.rt.bc_nll(logp, torch.tensor([1]), torch.tensor([0]), torch.tensor([True]))
        self.assertAlmostEqual(nll.item(), math.log(4) + math.log(3), places=6)

    def test_zero_head_has_zero_kl_to_official(self):
        logp = self.rt.guided_logp(self.head, self.cond, self.uncond, self.stage, 6.)
        official = self.rt.guided_logp(self.head, self.cond, self.uncond, self.stage, 6., official=True)
        self.assertEqual(self.rt.kl_to_official(logp, official).abs().max().item(), 0.)


@unittest.skipIf(torch is None, 'torch required')
class TrainTest(unittest.TestCase):
    def setUp(self):
        from minecraftrl.reacquire_training import train
        self.train = train
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)

    def run_train(self, **kw):
        path = toy_dataset(self.root / 'toy.pt', nan_holdout=kw.pop('nan_holdout', False))
        options = dict(use_stage=True, out_dir=self.root / 'out', epochs=4, lr=1e-2, head_factory=fake_factory, device='cpu')
        options.update(kw)
        return self.train(path, **options)

    def test_training_lowers_holdout_nll_on_a_learnable_toy_set(self):
        report = self.run_train()
        self.assertEqual(report['initial']['kl'], 0.)
        self.assertLess(report['epochs'][-1]['holdout_nll'], report['initial']['holdout_nll'])
        self.assertTrue(report['candidate']); self.assertTrue((self.root / 'out/checkpoint.pt').exists())
        self.assertEqual(report['reload_max_abs_diff'], 0.)
        self.assertEqual(json.loads((self.root / 'out/train-report.json').read_text())['checkpoint_sha256'],
                         report['checkpoint_sha256'])

    def test_epoch_over_kl_limit_is_rolled_back(self):
        report = self.run_train(lr=1., kl_limit=1e-6)
        self.assertEqual(report['rolled_back_epochs'], [1]); self.assertEqual(report['updates_kept'], 0)
        self.assertFalse(report['candidate']); self.assertTrue((self.root / 'out/checkpoint-rejected.pt').exists())

    def test_kl_limit_applies_to_unlabelled_rows_only(self):
        # Learning the labels must move the labelled rows far from the official policy; only unlabelled rows are leashed.
        from minecraftrl.reacquire_training import over_leash
        nan = float('nan')
        self.assertFalse(over_leash(dict(kl=10., kl_unlabelled=1.), 3.))
        self.assertTrue(over_leash(dict(kl=1., kl_unlabelled=4.), 3.))
        self.assertFalse(over_leash(dict(kl=10., kl_unlabelled=nan), 3.))     # no unlabelled training rows
        self.assertTrue(over_leash(dict(kl=nan, kl_unlabelled=1.), 3.))       # a broken update

    def test_holdout_rows_are_never_trained_on(self):
        report = self.run_train(nan_holdout=True, epochs=2)
        self.assertTrue(all(math.isfinite(e['train_loss']) for e in report['epochs']))
        self.assertEqual(report['rolled_back_epochs'], [])
        self.assertFalse(report['candidate'])          # the holdout NLL is not a number, so it cannot show a decrease

    def test_a_dataset_with_only_the_holdout_run_is_refused(self):
        path = toy_dataset(self.root / 'toy.pt'); data = torch.load(path); data['run'][:] = 1; torch.save(data, path)
        with self.assertRaisesRegex(ValueError, 'no training rows'):
            self.train(path, use_stage=True, out_dir=self.root / 'out', head_factory=fake_factory, device='cpu')

    def test_output_folder_is_not_overwritten(self):
        self.run_train(epochs=1)
        with self.assertRaises(FileExistsError): self.run_train(epochs=1)

    # Multi-block experiment (design 2026-11-08 §7).
    def test_group_weights_balance_block_and_stage_groups(self):
        from minecraftrl.reacquire_training import group_weights
        labelled = torch.tensor([True, True, True, True, False])
        stage = torch.tensor([2, 4, 4, 4, 4], dtype=torch.int8)
        block = torch.tensor([0, 0, 0, 0, 0])
        w = group_weights(labelled, stage, block, torch.arange(5))
        self.assertTrue(torch.allclose(w, torch.tensor([2., 2 / 3, 2 / 3, 2 / 3, 0.])))
        w = group_weights(labelled, stage, torch.tensor([0, 1, 1, 1, 0]), torch.arange(4))   # rows 0-3 only
        self.assertTrue(torch.allclose(w[:4], torch.tensor([2., 2 / 3, 2 / 3, 2 / 3])))
        self.assertEqual(w[4].item(), 0.)
        w = group_weights(labelled, stage, None, torch.arange(5))                            # no block: stage groups
        self.assertTrue(torch.allclose(w, torch.tensor([2., 2 / 3, 2 / 3, 2 / 3, 0.])))

    def test_balanced_training_runs_on_a_dataset_without_blocks(self):
        report = self.run_train(balance=True)
        self.assertTrue(report['balance']); self.assertTrue(report['candidate'])
        self.assertTrue(all(math.isfinite(e['train_loss']) for e in report['epochs']))

    def test_patience_restores_the_best_holdout_epoch(self):
        path = toy_dataset(self.root / 'toy.pt'); data = torch.load(path)
        held = data['run'] == 1
        data['label_buttons'][held] = torch.where(data['labelled'][held], torch.tensor(3), torch.tensor(-1))
        data['label_camera'][held] = torch.where(data['labelled'][held], torch.tensor(0), torch.tensor(-1))
        torch.save(data, path)     # the holdout wants other labels: its NLL rises while the training labels are learned
        report = self.train(path, use_stage=True, out_dir=self.root / 'out', epochs=30, lr=1e-2, kl_limit=1e9,
                            head_factory=fake_factory, device='cpu', patience=2)
        self.assertTrue(report['stopped_early']); self.assertGreaterEqual(report['best_epoch'], 1)
        self.assertEqual(len(report['epochs']), report['best_epoch'] + 2)
        lowest = min(e['holdout_nll'] for e in report['epochs'])
        self.assertEqual(report['epochs'][report['best_epoch'] - 1]['holdout_nll'], lowest)
        self.assertAlmostEqual(report['final']['holdout_nll'], lowest, places=5)
        self.assertEqual(report['updates_kept'], sum(e['updates'] for e in report['epochs'][:report['best_epoch']]))

    def test_patience_restores_the_lowest_holdout_epoch_of_an_improving_run(self):
        report = self.run_train(patience=2)
        self.assertFalse(report['stopped_early']); self.assertGreaterEqual(report['best_epoch'], 1)
        lowest = min(e['holdout_nll'] for e in report['epochs'] if not e.get('rolled_back'))
        self.assertAlmostEqual(report['final']['holdout_nll'], lowest, places=5)
        self.assertEqual(report['epochs'][report['best_epoch'] - 1]['holdout_nll'], lowest)
        self.assertTrue(report['candidate'])

    def test_a_dataset_with_features_trains_a_head_that_reads_them(self):
        # Jump direction input design 2026-11-14 §3: the label depends only on a feature the latent does not carry.
        path = toy_dataset(self.root / 'toy.pt'); data = torch.load(path); n = len(data['stage'])
        sign = torch.tensor([1. if (i // 2) % 2 else -1. for i in range(n)])
        data['features'] = torch.stack([sign, torch.zeros(n)], 1); data['feature_names'] = ['s', 'z']
        data['label_camera'] = torch.where(data['labelled'], torch.where(sign > 0, torch.tensor(2), torch.tensor(0)),
                                           torch.tensor(-1)).long()
        torch.save(data, path)
        seen = {}

        def factory(use_stage, width, stages=None, features=(), feature_gain=1.):
            from minecraftrl.reacquire_head import ReacquireHead
            seen['features'] = tuple(features); seen['gain'] = feature_gain
            return ReacquireHead(FakeBase().requires_grad_(False), width, use_stage=use_stage, stages=stages,
                                 features=features, feature_gain=feature_gain), 'base-sha'
        report = self.train(path, use_stage=True, out_dir=self.root / 'out', epochs=6, lr=1e-2, kl_limit=1e9,
                            head_factory=factory, device='cpu', feature_gain=10.)
        self.assertEqual((seen['features'], seen['gain']), (('s', 'z'), 10.))
        self.assertEqual((report['feature_gain'], torch.load(self.root / 'out/checkpoint.pt')['feature_gain']), (10., 10.))
        self.assertTrue(report['candidate']); self.assertEqual(report['reload_max_abs_diff'], 0.)
        self.assertEqual(torch.load(self.root / 'out/checkpoint.pt')['features'], ['s', 'z'])
        self.assertEqual(report['features'], ['s', 'z'])
        self.assertLess(report['final']['holdout_nll'], report['initial']['holdout_nll'] - .3)

    def use_dataset(self):
        """Toy set where far rows (stage 0) are labelled button 0 (no use) and near rows button 1 (use); combos 1 and 4
        press use."""
        path = toy_dataset(self.root / 'toy.pt', rows_per_run=256); data = torch.load(path)
        data['labelled'] = torch.ones(len(data['stage']), dtype=torch.bool)
        data['label_buttons'] = torch.where(data['stage'] == 2, torch.tensor(1), torch.tensor(0)).long()
        data['label_camera'] = torch.full((len(data['stage']),), 2).long(); data['camera_on'] = data['labelled'].clone()
        data['use_combos'] = torch.tensor([False, True, False, False, True])
        torch.save(data, path)
        return path

    def test_the_use_loss_lowers_use_where_the_label_has_none_and_reports_use_rates(self):
        # Building design 2026-11-15 §5: a factored loss on whether `use` is pressed.
        path = self.use_dataset()
        common = dict(use_stage=True, epochs=4, lr=1e-2, kl_limit=1e9, head_factory=fake_factory, device='cpu')
        plain = self.train(path, out_dir=self.root / 'plain', **common)
        used = self.train(path, out_dir=self.root / 'used', use_coef=5., **common)
        self.assertEqual((plain['hyper']['use_coef'], used['hyper']['use_coef']), (0., 5.))
        for r in (plain, used):
            self.assertEqual(set(r['final']['holdout_use']), {'label_no_use', 'label_use'})
        u, q = used['final']['holdout_use'], plain['final']['holdout_use']
        self.assertLess(u['label_no_use'], q['label_no_use'])
        self.assertGreater(u['label_use'] - u['label_no_use'], q['label_use'] - q['label_no_use'])

    def test_the_use_loss_needs_the_use_combinations(self):
        with self.assertRaises(ValueError):
            self.train(toy_dataset(self.root / 'toy.pt'), use_stage=True, out_dir=self.root / 'out', epochs=1,
                       head_factory=fake_factory, device='cpu', use_coef=1.)

    def test_a_list_of_holdout_runs_is_read(self):
        path = toy_dataset(self.root / 'toy.pt'); data = torch.load(path); data['holdout'] = [1]; torch.save(data, path)
        report = self.train(path, use_stage=True, out_dir=self.root / 'out', epochs=2, lr=1e-2, head_factory=fake_factory,
                            device='cpu')
        self.assertEqual(report['holdout'], ['r0/b']); self.assertEqual(report['rows']['holdout'], 64)


if __name__ == '__main__':
    unittest.main()
