"""Bridging dataset (design 2026-11-14 §5): labels on the state before each action, replayed stages and features."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minecraftrl.bridge_task import BRIDGE_FEATURES, BRIDGE_STAGES, bridge_fixture
from test_pipeline_stage_replay import row, timeline
from test_reacquire_data import STEVE1, mapping

TASK = dict(schema='bridge_task_v1', seed=5, gap=3, goal_dx=0, block='cobblestone')
SKILL = 'bridge'
NONE = dict(place_available=1, place_count=0, place_last_x=-999, place_last_y=-999, place_last_z=-999)


def brow(step, *, z=0.5, yaw=180., pitch=82., hit=None, face=1, controller='policy'):
    # Start block (0, 64, 0) facing south (the prepare row's yaw 0): the first bridge cell is (0, 63, 3).
    r = row(step, controller=controller, hit='air' if hit is None else 'stone_bricks', hit_pos=hit, yaw=yaw, pos=(0.5, 64., z))
    r.update(pitch=pitch, client_x=0.5, client_y=64., client_z=z, client_on_ground=1, server_on_ground=1, hit_face=face,
             place=dict(NONE))
    return r


def make_bridge_run(folder, *, seed=271828, recorded=('turn', 'place'), features=None):
    """A prepare row facing the gap at the start, then policy rows at the edge with the crosshair on the edge face:
    decision 2 sees the prepare row (turn), decision 3 sees row 2 (place)."""
    folder = Path(folder); (folder / 'reacquire').mkdir(parents=True, exist_ok=True)
    path = folder.parent / 'task.json'
    path.write_text(json.dumps(bridge_fixture(TASK)), encoding='utf-8')
    rows = [brow(1, controller='prepare', yaw=0., pitch=0.), brow(2, z=3.27, hit=(0, 63, 2), face=3),
            brow(3, z=3.27, hit=(0, 63, 2), face=3)]
    (folder / 'request.json').write_text(json.dumps(dict(mode='probe', seed=seed, sampling_seed=1, fixture=str(path),
        fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest())), encoding='utf-8')
    (folder / 'summary.json').write_text(json.dumps(dict(execution_valid=True, all_steps_completed=False)), encoding='utf-8')
    tl = timeline(skill=SKILL, target='bridge_goal', block=None, start=2, end=4)
    for name, items in (('timeline.jsonl', tl), ('telemetry.jsonl', rows)):
        (folder / name).write_text(''.join(json.dumps(x) + '\n' for x in items), encoding='utf-8')
    stage = [BRIDGE_STAGES.index(x) if x else -1 for x in recorded]
    latents = folder / 'reacquire/latents-000.npz'
    extra = {} if features is None else dict(features=np.asarray(features, np.float32))
    np.savez(latents, step=np.array([2, 3], np.int32), cond=np.ones((2, 4), np.float16), uncond=np.zeros((2, 4), np.float16),
             stage=np.array(stage, np.int8), **extra)
    manifest = dict(schema='reacquire_latents_v1', width=4, cond_scale=6., stages=list(BRIDGE_STAGES), rows=2,
                    chunks=[dict(file=latents.name, rows=2, sha256=hashlib.sha256(latents.read_bytes()).hexdigest())],
                    complete=True, meta=dict(skill=SKILL, use_stage=True))
    if features is not None: manifest['features'] = list(BRIDGE_FEATURES)
    (folder / 'reacquire/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return folder


@unittest.skipUnless(STEVE1, 'torch and external/steve1 required')
class BridgeExamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.mapper, self.transformer = mapping()

    def examples(self, folder, **kw):
        from minecraftrl.bridge_data import bridge_run_examples
        return bridge_run_examples(folder, self.mapper, self.transformer, root=Path(self.tmp.name), **kw)

    def keys(self, ex, i): return self.mapper.BUTTONS_IDX_TO_COMBINATION[int(ex['label_buttons'][i])]

    def test_labels_come_from_the_state_before_each_action_and_features_are_checked(self):
        ex = self.examples(make_bridge_run(Path(self.tmp.name) / 'a/r1'), features=True)
        self.assertEqual(ex['stage'].tolist(), [BRIDGE_STAGES.index('turn'), BRIDGE_STAGES.index('place')])
        self.assertEqual(ex['labelled'].tolist(), [True, True])
        self.assertIn('sneak', self.keys(ex, 0)); self.assertNotIn('use', self.keys(ex, 0))
        self.assertIn('use', self.keys(ex, 1))
        self.assertEqual(ex['features'].shape, (2, len(BRIDGE_FEATURES)))
        logged = make_bridge_run(Path(self.tmp.name) / 'b/r1', features=ex['features'])
        np.testing.assert_allclose(self.examples(logged, features=True)['features'], ex['features'], atol=1e-6)
        wrong = ex['features'].copy(); wrong[0, 3] += .5
        with self.assertRaises(ValueError): self.examples(make_bridge_run(Path(self.tmp.name) / 'c/r1', features=wrong))

    def test_refusals(self):
        with self.assertRaises(ValueError): self.examples(make_bridge_run(Path(self.tmp.name) / 'd/r1', recorded=('turn', 'aim')))
        with self.assertRaises(ValueError): self.examples(make_bridge_run(Path(self.tmp.name) / 'e/r1', seed=130363))

    def test_dataset_names_the_bridge_stages_one_group_and_the_features(self):
        import torch
        from minecraftrl.bridge_data import build_bridge_dataset
        runs = [make_bridge_run(Path(self.tmp.name) / 'f' / name) for name in ('r1', 'r2')]
        out = Path(self.tmp.name) / 'ds.pt'
        summary = build_bridge_dataset(runs, out, holdout='r2', mapper=self.mapper, transformer=self.transformer,
                                       root=Path(self.tmp.name), features=True)
        data = torch.load(out)
        self.assertEqual((data['stages'], data['blocks'], data['holdout'], data['teacher']), (list(BRIDGE_STAGES), ['bridge'], 1, 'bridge'))
        self.assertEqual((data['feature_names'], tuple(data['features'].shape)), (list(BRIDGE_FEATURES), (4, 5)))
        self.assertEqual(summary['labelled_by_stage']['place'], 2)
        self.assertIn('client pose', summary['privileged'])
        self.assertEqual(summary['runs'][0]['task'], 5)


if __name__ == '__main__':
    unittest.main()
