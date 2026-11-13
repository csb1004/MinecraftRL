"""Building dataset (design 2026-11-14 §4): labels on the state before each action, replayed stages and features."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minecraftrl.build_task import BUILD_FEATURES, BUILD_STAGES, build_fixture
from test_pipeline_stage_replay import row, timeline
from test_reacquire_data import STEVE1, mapping

TASK = dict(schema='build_task_v1', seed=5, shape='pillar2', cells=[[-1, 0, 4], [-1, 1, 4]], block='cobblestone')
SKILL = 'build'
NONE = dict(place_available=1, place_count=0, place_last_x=-999, place_last_y=-999, place_last_z=-999)


def prow(step, *, hit=None, face=1, controller='policy', z=1.5):
    # Start block (0, 64, 0) facing south: the first cell (-1, 0, 4) is the world cell (1, 64, 4) on the floor (1, 63, 4).
    r = row(step, controller=controller, hit='air' if hit is None else 'stone_bricks', hit_pos=hit)
    r.update(pitch=30., client_x=0.5, client_y=64., client_z=z, client_on_ground=1, hit_face=face, place=dict(NONE))
    return r


def make_build_run(folder, *, seed=271828, recorded=('aim', 'place'), features=None):
    """Prepare row looking at nothing, then policy rows on the floor top under the first cell: decision 2 sees the
    prepare row (aim), decision 3 sees row 2 (place)."""
    folder = Path(folder); (folder / 'reacquire').mkdir(parents=True, exist_ok=True)
    path = folder.parent / 'task.json'
    path.write_text(json.dumps(build_fixture(TASK)), encoding='utf-8')
    rows = [prow(1, controller='prepare'), prow(2, hit=(1, 63, 4)), prow(3, hit=(1, 63, 4))]
    (folder / 'request.json').write_text(json.dumps(dict(mode='probe', seed=seed, sampling_seed=1, fixture=str(path),
        fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest())), encoding='utf-8')
    (folder / 'summary.json').write_text(json.dumps(dict(execution_valid=True, all_steps_completed=False)), encoding='utf-8')
    tl = timeline(skill=SKILL, target='build_done', block=None, start=2, end=4)
    for name, items in (('timeline.jsonl', tl), ('telemetry.jsonl', rows)):
        (folder / name).write_text(''.join(json.dumps(x) + '\n' for x in items), encoding='utf-8')
    stage = [BUILD_STAGES.index(x) if x else -1 for x in recorded]
    latents = folder / 'reacquire/latents-000.npz'
    extra = {} if features is None else dict(features=np.asarray(features, np.float32))
    np.savez(latents, step=np.array([2, 3], np.int32), cond=np.ones((2, 4), np.float16), uncond=np.zeros((2, 4), np.float16),
             stage=np.array(stage, np.int8), **extra)
    manifest = dict(schema='reacquire_latents_v1', width=4, cond_scale=6., stages=list(BUILD_STAGES), rows=2,
                    chunks=[dict(file=latents.name, rows=2, sha256=hashlib.sha256(latents.read_bytes()).hexdigest())],
                    complete=True, meta=dict(skill=SKILL, use_stage=True))
    if features is not None: manifest['features'] = list(BUILD_FEATURES)
    (folder / 'reacquire/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return folder


@unittest.skipUnless(STEVE1, 'torch and external/steve1 required')
class BuildExamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.mapper, self.transformer = mapping()

    def examples(self, folder, **kw):
        from minecraftrl.build_data import build_run_examples
        return build_run_examples(folder, self.mapper, self.transformer, root=Path(self.tmp.name), **kw)

    def keys(self, ex, i): return self.mapper.BUTTONS_IDX_TO_COMBINATION[int(ex['label_buttons'][i])]

    def test_labels_come_from_the_state_before_each_action_and_features_are_checked(self):
        ex = self.examples(make_build_run(Path(self.tmp.name) / 'a/r1'), features=True)
        self.assertEqual(ex['stage'].tolist(), [BUILD_STAGES.index('aim'), BUILD_STAGES.index('place')])
        self.assertEqual(ex['labelled'].tolist(), [True, True])
        self.assertNotIn('use', self.keys(ex, 0)); self.assertIn('use', self.keys(ex, 1))
        self.assertEqual(ex['features'].shape, (2, len(BUILD_FEATURES)))
        logged = make_build_run(Path(self.tmp.name) / 'b/r1', features=ex['features'])
        np.testing.assert_allclose(self.examples(logged, features=True)['features'], ex['features'], atol=1e-6)
        wrong = ex['features'].copy(); wrong[0, 0] += .5
        with self.assertRaises(ValueError): self.examples(make_build_run(Path(self.tmp.name) / 'c/r1', features=wrong))

    def test_refusals(self):
        with self.assertRaises(ValueError): self.examples(make_build_run(Path(self.tmp.name) / 'd/r1', recorded=('aim', 'aim')))
        with self.assertRaises(ValueError): self.examples(make_build_run(Path(self.tmp.name) / 'e/r1', seed=130363))

    def test_dataset_names_the_build_stages_one_group_and_the_features(self):
        import torch
        from minecraftrl.build_data import build_build_dataset
        runs = [make_build_run(Path(self.tmp.name) / 'f' / name) for name in ('r1', 'r2')]
        out = Path(self.tmp.name) / 'ds.pt'
        summary = build_build_dataset(runs, out, holdout='r2', mapper=self.mapper, transformer=self.transformer,
                                      root=Path(self.tmp.name), features=True)
        data = torch.load(out)
        self.assertEqual((data['stages'], data['blocks'], data['holdout']), (list(BUILD_STAGES), ['build'], 1))
        self.assertEqual((data['feature_names'], tuple(data['features'].shape)), (list(BUILD_FEATURES), (4, 8)))
        self.assertEqual(summary['labelled_by_stage']['place'], 2)
        self.assertIn('client pose', summary['privileged'])


if __name__ == '__main__':
    unittest.main()
