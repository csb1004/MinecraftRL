import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minecraftrl.pipeline.stage_replay import STAGES
from minecraftrl.reacquire_data import FORBIDDEN_SEEDS, TRAIN_SEEDS, build_dataset, run_examples
from test_pipeline_stage_replay import row, timeline

FIXTURE = {'rotate_with_facing': True, 'blocks': [{'dx': 0, 'dy': 1, 'dz': 8, 'block': 'iron_ore'}]}
STEVE1 = importlib.util.find_spec('torch') is not None and (Path(__file__).resolve().parents[1] / 'external/steve1').exists()
NEAR = dict(pos=(0.5, 64., 5.5), yaw=180.)     # 3 blocks from the iron, facing away from it


def mapping():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'external/steve1'))
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.actions import ActionTransformer
    return CameraHierarchicalMapping(n_camera_bins=11), ActionTransformer(**ACTION_TRANSFORMER_KWARGS)


def make_run(folder, *, seed=271828, policy=((2, {}), (3, {})), recorded=('far', 'far'), latents=None,
             complete=True, valid=True, tamper=False, block='iron_ore', item='iron_ore', features=None, feature_names=None):
    """A recorded probe run: one prepare row, then policy rows (step, row kwargs), with its latent record. `block`
    replaces the fixture's iron ore and names the step's block (multi-block runs, design 2026-11-08). `features`: values
    logged with the latents by a head with the mining direction input (design 2026-11-14), under `feature_names`
    (default MINING_FEATURES)."""
    folder = Path(folder); (folder / 'reacquire').mkdir(parents=True, exist_ok=True)
    fixture = folder.parent / ('fixture.json' if block == 'iron_ore' else 'fixture-%s.json' % block)
    if not fixture.exists():
        fixture.write_text(json.dumps(dict(FIXTURE, blocks=[dict(b, block=block) for b in FIXTURE['blocks']])), encoding='utf-8')
    skill = 'mine_iron_reacquire' if block == 'iron_ore' else 'mine_%s_reacquire' % block
    rows = [row(1, controller='prepare')] + [row(step, **kw) for step, kw in policy]
    end = policy[-1][0] + 1
    (folder / 'request.json').write_text(json.dumps(dict(mode='probe', seed=seed, sampling_seed=1, fixture=str(fixture),
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())), encoding='utf-8')
    (folder / 'summary.json').write_text(json.dumps(dict(execution_valid=valid)), encoding='utf-8')
    tl = timeline(skill=skill, target=item, block=block, start=2, end=end)
    for name, items in (('timeline.jsonl', tl), ('telemetry.jsonl', rows)):
        (folder / name).write_text(''.join(json.dumps(x) + '\n' for x in items), encoding='utf-8')
    steps = [s for s, _ in policy] if latents is None else list(latents)
    n = len(steps)
    stage = [STAGES.index(x) if x else -1 for x in recorded][:n] + [-1] * max(0, n - len(recorded))
    path = folder / 'reacquire/latents-000.npz'
    extra = {} if features is None else dict(features=np.asarray(features, np.float32))
    np.savez(path, step=np.array(steps, np.int32), cond=np.ones((n, 4), np.float16) * np.arange(n)[:, None],
             uncond=np.zeros((n, 4), np.float16), stage=np.array(stage, np.int8), **extra)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if tamper: digest = '0' * 64
    manifest = dict(schema='reacquire_latents_v1', width=4, cond_scale=6., stages=list(STAGES), rows=n,
                    chunks=[dict(file=path.name, rows=n, sha256=digest)], complete=complete,
                    meta=dict(skill=skill, use_stage=True))
    if features is not None:
        from minecraftrl.reacquire_teacher import MINING_FEATURES
        manifest['features'] = list(feature_names or MINING_FEATURES)
    (folder / 'reacquire/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return folder


class SeedAndRecordChecksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)

    def test_seed_lists_are_disjoint(self):
        self.assertFalse(set(TRAIN_SEEDS) & set(FORBIDDEN_SEEDS))

    def test_forbidden_and_unknown_seeds_are_refused(self):
        for seed in (130363, 104729, 577215, 12345):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'training seed'):
                run_examples(make_run(self.root / ('s%d' % seed), seed=seed), None, None, root=self.root)

    def test_invalid_execution_and_incomplete_or_tampered_records_are_refused(self):
        for name, kw in (('invalid', dict(valid=False)), ('incomplete', dict(complete=False)), ('tampered', dict(tamper=True))):
            with self.subTest(name), self.assertRaises(ValueError):
                run_examples(make_run(self.root / name, **kw), None, None, root=self.root)


@unittest.skipUnless(STEVE1, 'isolated WSL Torch/STEVE-1 environment required')
class ExamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.mapper, self.transformer = mapping()

    def examples(self, folder):
        return run_examples(folder, self.mapper, self.transformer, root=self.root)

    def test_label_uses_the_state_before_the_action(self):
        ex = self.examples(make_run(self.root / 'a', policy=((2, {}), (3, NEAR)), recorded=('far', 'far')))
        self.assertEqual(ex['stage'].tolist(), [STAGES.index('far')] * 2)    # step 3 chose its action seeing row 2
        self.assertEqual(ex['labelled'].tolist(), [False, False])

    def test_near_row_gets_a_turn_label_for_the_next_latent(self):
        ex = self.examples(make_run(self.root / 'a', policy=((2, NEAR), (3, NEAR)), recorded=('far', 'near')))
        self.assertEqual(ex['labelled'].tolist(), [False, True])
        self.assertTrue(ex['camera_on'][1]); self.assertFalse(ex['in_view'][1])      # iron behind: turn, do not attack
        self.assertEqual(ex['label_buttons'][0], -1); self.assertGreaterEqual(ex['label_buttons'][1], 0)
        self.assertAlmostEqual(float(ex['label_yaw'][1]), 10., delta=1.)
        self.assertEqual(ex['step'].tolist(), [2, 3]); self.assertEqual(ex['cond_scale'], 6.)

    def test_hold_teacher_labels_an_aimed_state_without_camera(self):
        aimed = dict(pos=(0.5, 64., 5.5), yaw=0., hit='iron_ore', hit_pos=(0, 65, 8))
        folder = make_run(self.root / 'a', policy=((2, aimed), (3, aimed)), recorded=('far', 'aimed'))
        centre = run_examples(folder, self.mapper, self.transformer, root=self.root)
        hold = run_examples(folder, self.mapper, self.transformer, root=self.root, teacher='hold')
        self.assertTrue(hold['labelled'][1]); self.assertFalse(hold['camera_on'][1])
        self.assertEqual(hold['teacher'], 'hold'); self.assertEqual(centre['teacher'], 'centre')

    def test_sight_teacher_walks_round_a_block_that_hides_the_target(self):
        # Line-of-sight design 2026-11-10: the builder hands the fixture blocks of the moment to the teacher.
        blocks = FIXTURE['blocks'] + [{'dx': 1, 'dy': 1, 'dz': 8, 'block': 'stone'}, {'dx': -1, 'dy': 1, 'dz': 8, 'block': 'stone'},
                                      {'dx': 0, 'dy': 0, 'dz': 10, 'block': 'stone'}]      # a wall row; no room behind it
        (self.root / 'fixture.json').write_text(json.dumps(dict(FIXTURE, blocks=blocks)), encoding='utf-8')
        beside = dict(pos=(-1.5, 64., 8.5), yaw=-100., hit='stone', hit_pos=(-1, 65, 8))   # the stone at x = -1 hides the iron
        folder = make_run(self.root / 'a', policy=((2, beside), (3, beside)), recorded=('far', 'near'))
        press = run_examples(folder, self.mapper, self.transformer, root=self.root, teacher='press')
        sight = run_examples(folder, self.mapper, self.transformer, root=self.root, teacher='sight')

        def keys(ex): return self.mapper.BUTTONS_IDX_TO_COMBINATION[int(ex['label_buttons'][1])]
        self.assertNotIn('forward', keys(press)); self.assertAlmostEqual(float(press['label_yaw'][1]), 10., delta=1.)
        self.assertIn('forward', keys(sight)); self.assertAlmostEqual(float(sight['label_yaw'][1]), -10., delta=1.)

    def test_latent_count_must_match_policy_rows(self):
        with self.assertRaisesRegex(ValueError, 'policy rows'):
            self.examples(make_run(self.root / 'a', latents=[2]))

    def test_recorded_stage_must_match_recomputed_stage(self):
        with self.assertRaisesRegex(ValueError, 'live stage differs'):
            self.examples(make_run(self.root / 'a', recorded=('near', 'far')))

    def test_holdout_run_is_kept_apart(self):
        import torch
        a = make_run(self.root / 'r0/a', policy=((2, NEAR), (3, NEAR)), recorded=('far', 'near'))
        b = make_run(self.root / 'r0/b', policy=((2, NEAR), (3, NEAR), (4, NEAR)), recorded=('far', 'near', 'near'))
        out = self.root / 'data/r0.pt'
        summary = build_dataset([a, b], out, holdout='b', mapper=self.mapper, transformer=self.transformer, root=self.root)
        data = torch.load(out)
        self.assertEqual(summary['holdout'], 'b'); self.assertEqual(data['runs'], ['r0/a', 'r0/b'])
        self.assertEqual(data['run'].tolist(), [0, 0, 1, 1, 1]); self.assertEqual(data['holdout'], 1)
        self.assertEqual(summary['labelled'], 3)
        self.assertEqual(json.loads(out.with_suffix('.summary.json').read_text())['dataset_sha256'],
                         hashlib.sha256(out.read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            build_dataset([a, b], self.root / 'data/x.pt', holdout='missing', mapper=self.mapper,
                          transformer=self.transformer, root=self.root)

    # Multi-block experiment (design 2026-11-08 §5, §7).
    def test_rows_record_their_block_and_several_runs_can_be_held_out(self):
        import torch
        near = dict(policy=((2, NEAR), (3, NEAR)), recorded=('far', 'near'))
        a = make_run(self.root / 'r0/coal_ore-s271828-n1', block='coal_ore', item='coal', **near)
        b = make_run(self.root / 'r0/coal_ore-s161803-n1', block='coal_ore', item='coal', **near)
        c = make_run(self.root / 'r0/sand-s161803-n1', block='sand', item='sand', **near)
        out = self.root / 'data/r0.pt'
        summary = build_dataset([a, b, c], out, holdout=['coal_ore-s161803-n1', 'sand-s161803-n1'], mapper=self.mapper,
                                transformer=self.transformer, root=self.root, forbidden_blocks=('iron_ore',))
        data = torch.load(out)
        self.assertEqual(data['blocks'], ['coal_ore', 'sand'])
        self.assertEqual(data['block'].tolist(), [0, 0, 0, 0, 1, 1])
        self.assertEqual(data['holdout'], [1, 2])
        self.assertEqual(summary['holdout'], ['coal_ore-s161803-n1', 'sand-s161803-n1'])
        self.assertEqual(summary['labelled_by_block'], dict(coal_ore=2, sand=1))
        self.assertEqual([r['block'] for r in summary['runs']], ['coal_ore', 'coal_ore', 'sand'])

    def test_direction_features_are_recomputed_and_checked_against_the_record(self):
        # Mining direction input design 2026-11-14 §1-§2: earlier records get the features from telemetry; a record that
        # logged them must agree with the replay.
        import torch
        from minecraftrl.reacquire_teacher import MINING_FEATURES
        near = dict(policy=((2, NEAR), (3, NEAR)), recorded=('far', 'near'))
        old = make_run(self.root / 'r0/coal_ore-s271828-n1', block='coal_ore', item='coal', **near)
        self.assertNotIn('features', self.examples(old))
        ex = run_examples(old, self.mapper, self.transformer, root=self.root, teacher='sight', features=True)
        self.assertEqual(ex['features'].shape, (2, len(MINING_FEATURES)))
        self.assertGreater(ex['features'][0][1], .9)               # start row: the target is ahead
        self.assertLess(ex['features'][1][1], -.9)                 # near row facing away: the target is behind
        logged = make_run(self.root / 'r1/coal_ore-s271828-n1', block='coal_ore', item='coal', features=ex['features'], **near)
        np.testing.assert_allclose(run_examples(logged, self.mapper, self.transformer, root=self.root, features=True)['features'],
                                   ex['features'], atol=1e-6)
        wrong = ex['features'].copy(); wrong[1, 2] += .5
        bad = make_run(self.root / 'r2/coal_ore-s271828-n1', block='coal_ore', item='coal', features=wrong, **near)
        with self.assertRaisesRegex(ValueError, 'features differ'):
            run_examples(bad, self.mapper, self.transformer, root=self.root, features=True)
        out = self.root / 'data/f.pt'
        other = make_run(self.root / 'r0/sand-s161803-n1', block='sand', item='sand', **near)
        summary = build_dataset([old, other], out, holdout='sand-s161803-n1', mapper=self.mapper,
                                transformer=self.transformer, root=self.root, teacher='sight', features=True)
        data = torch.load(out)
        self.assertEqual((data['feature_names'], tuple(data['features'].shape)), (list(MINING_FEATURES), (4, 4)))
        self.assertEqual(summary['features'], list(MINING_FEATURES))

    def test_aim_features_are_output_while_records_are_checked_by_their_own_list(self):
        # Mining aim point design 2026-11-14 §1, §5: the output list is chosen; a logged record replays its own list.
        import torch
        from minecraftrl.reacquire_teacher import AIM_FEATURES, MINING_FEATURES
        near = dict(policy=((2, NEAR), (3, NEAR)), recorded=('far', 'near'))
        old = make_run(self.root / 'r0/coal_ore-s271828-n1', block='coal_ore', item='coal', **near)
        centre = run_examples(old, self.mapper, self.transformer, root=self.root, teacher='sight', features=True)
        aim = run_examples(old, self.mapper, self.transformer, root=self.root, teacher='sight', features=AIM_FEATURES)
        self.assertEqual((aim['feature_names'], aim['features'].shape), (AIM_FEATURES, (2, 4)))
        self.assertEqual(centre['feature_names'], MINING_FEATURES)
        np.testing.assert_allclose(aim['features'], centre['features'], atol=1e-6)   # open fixture: the centre is in sight
        v2 = make_run(self.root / 'r1/coal_ore-s271828-n1', block='coal_ore', item='coal', features=centre['features'], **near)
        run_examples(v2, self.mapper, self.transformer, root=self.root, features=AIM_FEATURES)          # checked as v2
        wrong = centre['features'].copy(); wrong[1, 2] += .5
        bad = make_run(self.root / 'r2/coal_ore-s271828-n1', block='coal_ore', item='coal', features=wrong, **near)
        with self.assertRaisesRegex(ValueError, 'features differ'):
            run_examples(bad, self.mapper, self.transformer, root=self.root, features=AIM_FEATURES)
        v3 = make_run(self.root / 'r3/coal_ore-s271828-n1', block='coal_ore', item='coal', features=aim['features'],
                      feature_names=AIM_FEATURES, **near)
        run_examples(v3, self.mapper, self.transformer, root=self.root, features=True)                  # checked as v3
        odd = make_run(self.root / 'r4/coal_ore-s271828-n1', block='coal_ore', item='coal', features=aim['features'],
                       feature_names=('a', 'b', 'c', 'd'), **near)
        with self.assertRaisesRegex(ValueError, 'another feature list'):
            run_examples(odd, self.mapper, self.transformer, root=self.root, features=AIM_FEATURES)
        with self.assertRaisesRegex(ValueError, 'unknown mining feature list'):
            run_examples(old, self.mapper, self.transformer, root=self.root, features=('a', 'b'))
        other = make_run(self.root / 'r0/sand-s161803-n1', block='sand', item='sand', **near)
        out = self.root / 'data/aim.pt'
        summary = build_dataset([old, other], out, holdout='sand-s161803-n1', mapper=self.mapper,
                                transformer=self.transformer, root=self.root, teacher='sight', features=AIM_FEATURES)
        data = torch.load(out)
        self.assertEqual((data['feature_names'], tuple(data['features'].shape)), (list(AIM_FEATURES), (4, 4)))
        self.assertEqual(summary['features'], list(AIM_FEATURES))
        self.assertIn('aim point', summary['privileged'])

    def test_a_held_out_block_is_refused(self):
        near = dict(policy=((2, NEAR), (3, NEAR)), recorded=('far', 'near'))
        a = make_run(self.root / 'r0/coal_ore-s271828-n1', block='coal_ore', item='coal', **near)
        iron = make_run(self.root / 'r0/s271828-n1', **near)
        with self.assertRaisesRegex(ValueError, 'held-out block iron_ore'):
            build_dataset([a, iron], self.root / 'data/x.pt', holdout='coal_ore-s271828-n1', mapper=self.mapper,
                          transformer=self.transformer, root=self.root, forbidden_blocks=('iron_ore',))
        self.assertFalse((self.root / 'data/x.pt').exists())


if __name__ == '__main__':
    unittest.main()
