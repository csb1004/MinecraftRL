"""Jump course dataset (design 2026-11-13 §5): labels on the state before each action, replayed stages, refusals."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minecraftrl.jump_course import JUMP_STAGES, course_fixture
from test_pipeline_stage_replay import row, timeline
from test_reacquire_data import STEVE1, mapping

COURSE = dict(schema='jump_course_v1', seed=5, platforms=[
    dict(x0=-1, x1=1, z0=-1, z1=1, top=0, gap=0, block='stone_bricks'),
    dict(x0=-1, x1=1, z0=3, z1=5, top=0, gap=1, block='gold_block')])
SKILL = 'jump_course'


def jrow(step, pos, ground, controller='policy', yaw=0.):
    r = row(step, pos=pos, controller=controller, yaw=yaw)
    r['server_on_ground'] = int(ground); r['pitch'] = 20.
    return r


def make_jump_run(folder, *, seed=271828, recorded=('run', 'edge'), fixture=None):
    """Prepare row on the start, a policy row near the edge, a policy row in the air."""
    folder = Path(folder); (folder / 'reacquire').mkdir(parents=True, exist_ok=True)
    path = folder.parent / 'course.json'
    path.write_text(json.dumps(fixture or course_fixture(COURSE)), encoding='utf-8')
    rows = [jrow(1, (0.5, 64., 0.5), True, controller='prepare'), jrow(2, (0.5, 64., 1.9), True),
            jrow(3, (0.5, 64.5, 2.4), False)]
    (folder / 'request.json').write_text(json.dumps(dict(mode='probe', seed=seed, sampling_seed=1, fixture=str(path),
        fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest())), encoding='utf-8')
    (folder / 'summary.json').write_text(json.dumps(dict(execution_valid=True, all_steps_completed=False)), encoding='utf-8')
    tl = timeline(skill=SKILL, target='course_goal', block=None, start=2, end=4)
    for name, items in (('timeline.jsonl', tl), ('telemetry.jsonl', rows)):
        (folder / name).write_text(''.join(json.dumps(x) + '\n' for x in items), encoding='utf-8')
    stage = [JUMP_STAGES.index(x) if x else -1 for x in recorded]
    latents = folder / 'reacquire/latents-000.npz'
    np.savez(latents, step=np.array([2, 3], np.int32), cond=np.ones((2, 4), np.float16), uncond=np.zeros((2, 4), np.float16),
             stage=np.array(stage, np.int8))
    manifest = dict(schema='reacquire_latents_v1', width=4, cond_scale=6., stages=list(JUMP_STAGES), rows=2,
                    chunks=[dict(file=latents.name, rows=2, sha256=hashlib.sha256(latents.read_bytes()).hexdigest())],
                    complete=True, meta=dict(skill=SKILL, use_stage=True))
    (folder / 'reacquire/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return folder


@unittest.skipUnless(STEVE1, 'torch and external/steve1 required')
class JumpExamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.mapper, self.transformer = mapping()

    def examples(self, folder):
        from minecraftrl.jump_data import jump_run_examples
        return jump_run_examples(folder, self.mapper, self.transformer, root=Path(self.tmp.name))

    def keys(self, ex, i): return self.mapper.BUTTONS_IDX_TO_COMBINATION[int(ex['label_buttons'][i])]

    def test_labels_come_from_the_state_before_each_action(self):
        ex = self.examples(make_jump_run(Path(self.tmp.name) / 'a/r1'))
        self.assertEqual(ex['stage'].tolist(), [JUMP_STAGES.index('run'), JUMP_STAGES.index('edge')])
        self.assertEqual(ex['labelled'].tolist(), [True, True])
        self.assertIn('forward', self.keys(ex, 0)); self.assertNotIn('jump', self.keys(ex, 0))
        self.assertIn('jump', self.keys(ex, 1))
        self.assertEqual(ex['course'], 5)

    def test_refusals(self):
        with self.assertRaises(ValueError): self.examples(make_jump_run(Path(self.tmp.name) / 'b/r1', recorded=('run', 'run')))
        with self.assertRaises(ValueError): self.examples(make_jump_run(Path(self.tmp.name) / 'c/r1', seed=130363))
        plain = dict(course_fixture(COURSE)); plain.pop('course')
        with self.assertRaises(ValueError): self.examples(make_jump_run(Path(self.tmp.name) / 'd/r1', fixture=plain))

    def test_dataset_names_the_jump_stages_and_one_group(self):
        import torch
        from minecraftrl.jump_data import build_jump_dataset
        runs = [make_jump_run(Path(self.tmp.name) / 'e' / name) for name in ('r1', 'r2')]
        out = Path(self.tmp.name) / 'ds.pt'
        summary = build_jump_dataset(runs, out, holdout='r2', mapper=self.mapper, transformer=self.transformer,
                                     root=Path(self.tmp.name))
        data = torch.load(out)
        self.assertEqual((data['stages'], data['blocks'], data['holdout']), (list(JUMP_STAGES), ['jump_course'], 1))
        self.assertEqual(summary['labelled_by_stage']['edge'], 2)
        self.assertEqual(data['block'].tolist(), [0, 0, 0, 0])

    def test_summary_names_the_pose_each_privileged_input_uses(self):
        """Stages and labels follow the client pose since 2026-11-13 09:20; the note must not say server pose."""
        from minecraftrl.jump_data import build_jump_dataset
        runs = [make_jump_run(Path(self.tmp.name) / 'f' / name) for name in ('r1', 'r2')]
        summary = build_jump_dataset(runs, Path(self.tmp.name) / 'ds2.pt', holdout='r2', mapper=self.mapper,
                                     transformer=self.transformer, root=Path(self.tmp.name))
        self.assertIn('client pose', summary['privileged'])
        self.assertIn('server pose', summary['privileged'])
        self.assertNotIn('use the course and the server pose', summary['privileged'])


if __name__ == '__main__':
    unittest.main()
