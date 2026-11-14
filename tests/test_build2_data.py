"""Recovering building dataset (design 2026-11-15 §4): round 1 runs relabelled, recovering runs checked, use combos."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minecraftrl.build_recover import BUILD2_FEATURES, BUILD2_STAGES, build2_fixture
from test_build_data import TASK, make_build_run
from test_reacquire_data import STEVE1, mapping


def as_recovering(folder, recorded=('aim', 'place'), features=None):
    """Turn a round 1 test run into one recorded with the recovering feed (fixture, stage list, stages, features)."""
    folder = Path(folder)
    path = folder.parent / 'task.json'; path.write_text(json.dumps(build2_fixture(TASK)), encoding='utf-8')
    request = json.loads((folder / 'request.json').read_text()); request['fixture_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / 'request.json').write_text(json.dumps(request), encoding='utf-8')
    latents = folder / 'reacquire/latents-000.npz'
    with np.load(latents) as d: arrays = {k: d[k] for k in d.files}
    arrays['stage'] = np.array([BUILD2_STAGES.index(s) for s in recorded], np.int8)
    arrays.pop('features', None)
    if features is not None: arrays['features'] = np.asarray(features, np.float32)
    np.savez(latents, **arrays)
    manifest = json.loads((folder / 'reacquire/manifest.json').read_text())
    manifest.update(stages=list(BUILD2_STAGES))
    manifest['chunks'][0]['sha256'] = hashlib.sha256(latents.read_bytes()).hexdigest()
    if features is not None: manifest['features'] = list(BUILD2_FEATURES)
    else: manifest.pop('features', None)
    (folder / 'reacquire/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return folder


@unittest.skipUnless(STEVE1, 'torch and external/steve1 required')
class Build2ExamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.mapper, self.transformer = mapping()

    def examples(self, folder):
        from minecraftrl.build2_data import build2_run_examples
        return build2_run_examples(folder, self.mapper, self.transformer, root=Path(self.tmp.name))

    def keys(self, ex, i): return self.mapper.BUTTONS_IDX_TO_COMBINATION[int(ex['label_buttons'][i])]

    def test_a_round_1_run_is_relabelled_with_the_recovering_stages_and_features(self):
        ex = self.examples(make_build_run(Path(self.tmp.name) / 'a/r1', recorded=('reach', 'reach')))  # not compared
        self.assertEqual(ex['stage'].tolist(), [BUILD2_STAGES.index('aim'), BUILD2_STAGES.index('place')])
        self.assertNotIn('use', self.keys(ex, 0)); self.assertIn('use', self.keys(ex, 1))
        self.assertEqual(ex['features'].shape, (2, len(BUILD2_FEATURES))); self.assertTrue(ex['relabelled'])

    def test_a_recovering_run_must_match_its_replay(self):
        ok = as_recovering(make_build_run(Path(self.tmp.name) / 'b/r1'))
        ex = self.examples(ok); self.assertFalse(ex['relabelled'])
        logged = as_recovering(make_build_run(Path(self.tmp.name) / 'c/r1'), features=ex['features'])
        np.testing.assert_allclose(self.examples(logged)['features'], ex['features'], atol=1e-6)
        with self.assertRaises(ValueError):
            self.examples(as_recovering(make_build_run(Path(self.tmp.name) / 'd/r1'), recorded=('aim', 'aim')))
        wrong = ex['features'].copy(); wrong[0, 0] += .5
        with self.assertRaises(ValueError):
            self.examples(as_recovering(make_build_run(Path(self.tmp.name) / 'e/r1'), features=wrong))

    def test_dataset_names_the_recovering_lists_and_the_use_combinations(self):
        import torch
        from minecraftrl.build2_data import build2_dataset
        runs = [make_build_run(Path(self.tmp.name) / 'f' / name) for name in ('r1', 'r2')]
        out = Path(self.tmp.name) / 'ds.pt'
        summary = build2_dataset(runs, out, holdout='r2', mapper=self.mapper, transformer=self.transformer,
                                 root=Path(self.tmp.name))
        data = torch.load(out)
        self.assertEqual((data['stages'], data['feature_names']), (list(BUILD2_STAGES), list(BUILD2_FEATURES)))
        self.assertEqual(tuple(data['features'].shape), (4, len(BUILD2_FEATURES)))
        combos = data['use_combos']
        self.assertEqual(len(combos), len(self.mapper.BUTTONS_IDX_TO_COMBINATION))
        self.assertEqual(combos[data['label_buttons'][data['labelled']]].tolist(), [False, True, False, True])
        self.assertEqual(summary['labelled_by_stage']['place'], 2); self.assertTrue(summary['runs'][0]['relabelled'])


if __name__ == '__main__':
    unittest.main()
