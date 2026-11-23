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


def as_sensor(folder, recorded=None, features=None, log=True):
    """Turn a round 1 test run into one recorded with the sensor feed (plan 2026-11-22): the sensor fixture, a sensor log
    with one raw frame per row (floor seen under the start, the first cell in reach), and the rows' `sensor_row`."""
    from minecraftrl.build_sensor import SENSOR_BUILD_FEATURES, sensor_fixture
    from minecraftrl.pipeline.sensor import SensorLog
    from test_pipeline_sensor import raw_sensor
    folder = Path(folder)
    if recorded is not None: as_recovering(folder, recorded=recorded, features=features)   # recorded live with this feed
    path = folder.parent / 'task.json'; path.write_text(json.dumps(sensor_fixture(TASK)), encoding='utf-8')
    request = json.loads((folder / 'request.json').read_text()); request['fixture_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / 'request.json').write_text(json.dumps(request), encoding='utf-8')
    rows = [json.loads(line) for line in (folder / 'telemetry.jsonl').read_text().splitlines()]
    if log:
        sensor = SensorLog(folder / 'sensor')
        for r in rows:
            r['sensor_row'] = sensor.add(r['step'], raw_sensor(sensor_client_tick=r['step'], sensor_foot_x=0, sensor_foot_z=0))
        sensor.close()
    (folder / 'telemetry.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    if features is not None:
        manifest = json.loads((folder / 'reacquire/manifest.json').read_text())
        manifest['features'] = list(SENSOR_BUILD_FEATURES)
        (folder / 'reacquire/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return folder


@unittest.skipUnless(STEVE1, 'torch and external/steve1 required')
class SensorExamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.mapper, self.transformer = mapping()

    def examples(self, folder):
        from minecraftrl.build2_data import build2_run_examples
        from minecraftrl.build_sensor import SENSOR
        return build2_run_examples(folder, self.mapper, self.transformer, root=Path(self.tmp.name), teacher=SENSOR)

    def test_a_sensor_run_replays_its_frames_and_must_match_the_live_stages_and_features(self):
        from minecraftrl.build_sensor import SENSOR_BUILD_FEATURES
        replay = self.examples(as_sensor(make_build_run(Path(self.tmp.name) / 'a/r1')))   # round 1 lists: relabelled
        self.assertEqual(replay['features'].shape, (2, len(SENSOR_BUILD_FEATURES)))
        self.assertTrue(replay['relabelled']); self.assertTrue(replay['labelled'].all())
        names = [BUILD2_STAGES[i] for i in replay['stage'].tolist()]
        own = self.examples(as_sensor(make_build_run(Path(self.tmp.name) / 'b/r1'), recorded=names, features=replay['features']))
        self.assertFalse(own['relabelled'])
        np.testing.assert_allclose(own['features'], replay['features'], atol=1e-6)
        wrong = replay['features'].copy(); wrong[0, 3] += .5
        with self.assertRaises(ValueError):
            self.examples(as_sensor(make_build_run(Path(self.tmp.name) / 'c/r1'), recorded=names, features=wrong))
        other = [('place' if n != 'place' else 'aim') for n in names]
        with self.assertRaises(ValueError):
            self.examples(as_sensor(make_build_run(Path(self.tmp.name) / 'd/r1'), recorded=other, features=replay['features']))

    def test_a_run_without_a_sensor_log_cannot_feed_the_sensor_head(self):
        # Design §5.2: earlier records have no rays; they are not sensor training data.
        with self.assertRaises(ValueError):
            self.examples(as_sensor(make_build_run(Path(self.tmp.name) / 'f/r1'), log=False))
        with self.assertRaises(ValueError):
            self.examples(make_build_run(Path(self.tmp.name) / 'e/r1'))


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

    def test_a_run_recorded_with_the_recovering_teacher_is_relabelled_by_the_careful_one(self):
        # Round 4 (design 2026-11-15 careful place): runs of another teacher are relabelled, not compared.
        from minecraftrl.build2_data import build2_run_examples
        for name, recorded in (('g/r1', ('aim', 'place')), ('h/r1', ('aim', 'aim'))):
            run = as_recovering(make_build_run(Path(self.tmp.name) / name), recorded=recorded)
            ex = build2_run_examples(run, self.mapper, self.transformer, root=Path(self.tmp.name), teacher='careful')
            self.assertTrue(ex['relabelled']); self.assertEqual(ex['features'].shape, (2, len(BUILD2_FEATURES)))
        with self.assertRaises(ValueError):
            build2_run_examples(run, self.mapper, self.transformer, root=Path(self.tmp.name), teacher='other')

    def test_a_careful_dataset_names_its_teacher(self):
        import torch
        from minecraftrl.build2_data import build2_dataset
        runs = [make_build_run(Path(self.tmp.name) / 'i' / name) for name in ('r1', 'r2')]
        out = Path(self.tmp.name) / 'careful.pt'
        summary = build2_dataset(runs, out, holdout='r2', mapper=self.mapper, transformer=self.transformer,
                                 root=Path(self.tmp.name), teacher='careful')
        self.assertEqual((torch.load(out)['teacher'], summary['teacher']), ('build_careful', 'build_careful'))

    def test_the_repair_teacher_relabels_other_runs_with_its_six_stages_and_twelve_features(self):
        # Round 5 (design 2026-11-15 build-repair): the dig stage and the pickaxe-in-hand feature.
        import torch
        from minecraftrl.build2_data import build2_dataset, build2_run_examples
        from minecraftrl.build_repair import REPAIR_FEATURES, REPAIR_STAGES
        def holding(folder, item):
            path = Path(folder) / 'telemetry.jsonl'
            rows = [dict(json.loads(line), server_selected_item=item) for line in path.read_text().splitlines()]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
            return folder

        run = holding(as_recovering(make_build_run(Path(self.tmp.name) / 'j/r1')), 'cobblestone')
        ex = build2_run_examples(run, self.mapper, self.transformer, root=Path(self.tmp.name), teacher='repair')
        self.assertTrue(ex['relabelled']); self.assertEqual(ex['features'].shape, (2, len(REPAIR_FEATURES)))
        # No stray and the block in hand: the careful teacher's stages, labels and features, plus 'held' = 0.
        careful = build2_run_examples(run, self.mapper, self.transformer, root=Path(self.tmp.name), teacher='careful')
        self.assertEqual(ex['stage'].tolist(), [REPAIR_STAGES.index(BUILD2_STAGES[s]) for s in careful['stage']])
        np.testing.assert_array_equal(ex['label_buttons'], careful['label_buttons'])
        np.testing.assert_allclose(ex['features'][:, :-1], careful['features']); self.assertEqual(ex['features'][:, -1].tolist(), [0., 0.])
        # The pickaxe in hand: the same stages, 'held' = 1, and every label switches back to the block (slot 1).
        pick = holding(as_recovering(make_build_run(Path(self.tmp.name) / 'j2/r1')), 'stone_pickaxe')
        ex = build2_run_examples(pick, self.mapper, self.transformer, root=Path(self.tmp.name), teacher='repair')
        self.assertEqual(ex['features'][:, -1].tolist(), [1., 1.])
        self.assertTrue(all('hotbar.1' in self.keys(ex, i) and 'use' not in self.keys(ex, i) for i in range(2)))
        runs = [make_build_run(Path(self.tmp.name) / 'k' / name) for name in ('r1', 'r2')]
        out = Path(self.tmp.name) / 'repair.pt'
        summary = build2_dataset(runs, out, holdout='r2', mapper=self.mapper, transformer=self.transformer,
                                 root=Path(self.tmp.name), teacher='repair')
        data = torch.load(out)
        self.assertEqual((data['stages'], data['feature_names'], data['teacher']),
                         (list(REPAIR_STAGES), list(REPAIR_FEATURES), 'build_repair'))
        self.assertEqual((summary['labelled_by_stage']['dig'], summary['features']), (0, list(REPAIR_FEATURES)))

    def test_a_repair_run_must_match_its_replay(self):
        from minecraftrl.build2_data import build2_run_examples
        from minecraftrl.build_repair import REPAIR_STAGES, repair_fixture

        def as_repair(folder, recorded):
            folder = as_recovering(folder)
            path = folder.parent / 'task.json'; path.write_text(json.dumps(repair_fixture(TASK)), encoding='utf-8')
            request = json.loads((folder / 'request.json').read_text())
            request['fixture_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            (folder / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            latents = folder / 'reacquire/latents-000.npz'
            with np.load(latents) as d: arrays = {k: d[k] for k in d.files}
            arrays['stage'] = np.array([REPAIR_STAGES.index(s) for s in recorded], np.int8); np.savez(latents, **arrays)
            manifest = json.loads((folder / 'reacquire/manifest.json').read_text())
            manifest.update(stages=list(REPAIR_STAGES))
            manifest['chunks'][0]['sha256'] = hashlib.sha256(latents.read_bytes()).hexdigest()
            (folder / 'reacquire/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            return folder

        run = as_repair(make_build_run(Path(self.tmp.name) / 'l/r1'), ('aim', 'aim'))     # the careful-place stages
        ex = build2_run_examples(run, self.mapper, self.transformer, root=Path(self.tmp.name), teacher='repair')
        self.assertFalse(ex['relabelled'])
        with self.assertRaises(ValueError):
            build2_run_examples(as_repair(make_build_run(Path(self.tmp.name) / 'm/r1'), ('aim', 'dig')), self.mapper,
                                self.transformer, root=Path(self.tmp.name), teacher='repair')
        with self.assertRaises(ValueError):                  # a repair record is not relabelled by the careful teacher
            build2_run_examples(run, self.mapper, self.transformer, root=Path(self.tmp.name), teacher='careful')

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


class DigFeatureListTests(unittest.TestCase):
    """The dig head's crosshair inputs (user decision 2026-11-24): a dataset may extend the d_ list of earlier runs."""

    def test_only_the_dig_teacher_takes_the_extended_list(self):
        from minecraftrl.build2_data import feature_list_for
        from minecraftrl.build_sensor_repair import DIG2_FEATURES, DIG_FEATURES
        self.assertEqual(feature_list_for('sensor_dig', None), DIG_FEATURES)
        self.assertEqual(feature_list_for('sensor_dig', DIG2_FEATURES), DIG2_FEATURES)
        with self.assertRaises(ValueError): feature_list_for('sensor_dig', ('d_x',))
        with self.assertRaises(ValueError): feature_list_for('sensor_recover', DIG2_FEATURES)

    def test_an_extended_list_compares_the_logged_columns_and_another_list_is_refused(self):
        from minecraftrl.build2_data import compare_logged_features
        replayed = np.array([[1., 2., 0., 1.]], np.float32); names = ('a', 'b', 'c', 'd')
        compare_logged_features(True, ['a', 'b'], names, np.array([[1., 2.]], np.float32), replayed, 'r')
        compare_logged_features(True, list(names), names, replayed.copy(), replayed, 'r')
        with self.assertRaises(ValueError):
            compare_logged_features(True, ['a', 'b'], names, np.array([[1., 2.5]], np.float32), replayed, 'r')
        with self.assertRaises(ValueError): compare_logged_features(True, ['a', 'x'], names, None, replayed, 'r')
        compare_logged_features(False, ['a', 'x'], names, np.array([[9., 9.]], np.float32), replayed, 'r')   # relabelled


if __name__ == '__main__':
    unittest.main()
