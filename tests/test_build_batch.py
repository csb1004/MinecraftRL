"""Building batch runner and report (design 2026-11-14 §4): modes, files, the teacher check, the input record."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / ('scripts/%s.py' % name))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class BuildBatchTest(unittest.TestCase):
    def setUp(self):
        self.m = load_script('run_build_batch'); self.registry = self.m.load_registry(ROOT)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.batch = ROOT / 'results' / ('build-batch-test-' + Path(self.tmp.name).name)
        self.addCleanup(lambda: shutil.rmtree(self.batch, ignore_errors=True))
        self.launched = []

    def launch(self, command, run_dir):
        self.launched.append((command, run_dir))
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / 'summary.json').write_text(json.dumps(dict(
            execution_valid=True, status='completed', all_steps_completed=True, release='neutral_command_sent',
            stage_results=[dict(status='completed', filled=3, wrong=1, unknown_placements=0, steps=180)],
            process_cleanup=dict(remaining_live_pids=[]))), encoding='utf-8')

    def collect(self, **kw):
        options = dict(batch=self.batch, arm='teacher', checkpoint=None, use_stage=True, mode='train', tasks=[101],
                       seeds=[271828], sampling=[1], teacher_check=True, root=ROOT, launch=self.launch, gpu_free=lambda: 3000.)
        options.update(kw)
        return self.m.collect(**options)

    def test_task_splits_are_disjoint_and_modes_check_seeds_and_tasks(self):
        t = self.registry['tasks']
        self.assertFalse(set(t['train']) & set(t['dev']) or set(t['train']) & set(t['final']) or set(t['dev']) & set(t['final']))
        self.m.check('train', [101], [271828], self.registry); self.m.check('final', [301], [662607], self.registry)
        for mode, task, seed in (('train', 201, 271828), ('train', 101, 130363), ('dev', 101, 130363),
                                 ('final', 301, 577215), ('dev', 201, 662607)):
            with self.subTest(mode=mode, task=task, seed=seed), self.assertRaises(ValueError):
                self.m.check(mode, [task], [seed], self.registry)

    def test_policy_files_carry_the_task_the_build_head_the_monitor_and_a_build_step(self):
        from minecraftrl.build_task import build_fixture, generate_build
        config, fixture, plan = self.m.prepare(self.batch, 'official', ROOT, 103, None, True)
        self.assertEqual(json.loads(fixture.read_text()), build_fixture(generate_build(103)))
        self.assertEqual(fixture.name, 'task-103.json')
        value = json.loads(config.read_text())
        self.assertIs(value['place_monitor'], True)
        skills = value['skills']
        self.assertEqual([n for n, s in skills.items() if s.get('reacquire')], ['build'])
        self.assertEqual(skills['build']['reacquire'], dict(checkpoint=None, use_stage=True, record_latents=True, stages='build'))
        self.assertEqual(json.loads(plan.read_text()), [dict(id='build', skill='build', kind='policy', target='build_done',
                                                              count=1, params=dict(until='build'))])

    def test_the_recovering_teacher_is_named_in_the_fixture_and_its_module_hashed(self):
        # Design 2026-11-15: the same task and world, with the teacher named for the feed.
        from minecraftrl.build_recover import build2_fixture
        from minecraftrl.build_task import generate_build
        _, fixture, _ = self.m.prepare(self.batch, 'official', ROOT, 103, None, True, teacher='recover')
        self.assertEqual(json.loads(fixture.read_text()), build2_fixture(generate_build(103)))
        self.assertEqual(self.collect(teacher='recover'), 0)
        frozen = json.loads((self.batch / 'frozen-teacher-c101.json').read_text())
        self.assertTrue(any(k.replace('\\', '/').endswith('src/minecraftrl/build_recover.py') for k in frozen['hashes']))
        self.assertEqual(frozen['teacher'], 'recover')
        with self.assertRaises(ValueError): self.m.prepare(self.batch, 'official', ROOT, 103, None, True, teacher='other')

    def test_the_sensor_teacher_turns_on_the_sensor_monitor_writes_the_blueprint_and_hashes_its_modules(self):
        # Plan 2026-11-22 sensor build: the recovering teacher on sensors, a blueprint instruction file, a first-use final.
        from minecraftrl.build_sensor import blueprint_of, sensor_fixture
        from minecraftrl.build_task import generate_build
        config, fixture, _ = self.m.prepare(self.batch, 'official', ROOT, 103, None, True, teacher='sensor')
        self.assertEqual(json.loads(fixture.read_text()), sensor_fixture(generate_build(103)))
        self.assertEqual(json.loads(fixture.read_text())['build']['teacher'], 'sensor_recover')
        value = json.loads(config.read_text())
        self.assertIs(value['sensor_monitor'], True); self.assertIs(value['place_monitor'], True)
        self.assertEqual(json.loads((self.batch / 'blueprints' / 'task-103.json').read_text()), blueprint_of(generate_build(103)))
        self.assertEqual(self.collect(teacher='sensor'), 0)
        names = [k.replace('\\', '/') for k in json.loads((self.batch / 'frozen-teacher-c101.json').read_text())['hashes']]
        for module in ('build_sensor.py', 'SensorObservation.java', 'pipeline/sensor.py', 'pipeline/seen_map.py',
                       'pipeline/sensor_guard.py', 'build_recover.py'):
            self.assertTrue(any(n.endswith(module) for n in names), module)
        self.m.check('sensor_final', [1401, 1408], [618034], self.registry)
        for bad in (([301], [618034]), ([1401], [662607])):
            with self.assertRaises(ValueError): self.m.check('sensor_final', *bad, self.registry)
        self.m.prepare(self.batch, 'official', ROOT, 401, None, True, teacher='sensor', occluded=True)   # a recovering teacher
        plain = json.loads(self.m.prepare(self.batch, 'other', ROOT, 103, None, True, teacher='recover')[0].read_text())
        self.assertNotIn('sensor_monitor', plain)

    def test_the_beacon_mode_runs_the_pyramid_on_its_own_seed_with_a_hold_after_success(self):
        # Beacon demo (user request 2026-11-23): tasks 1501-1504 on 316227 (first use), the sensor teacher only.
        from minecraftrl.build_beacon import beacon_build
        from minecraftrl.build_sensor import blueprint_of, sensor_fixture
        self.assertEqual(self.registry['tasks']['beacon'], [1501, 1502, 1503, 1504])
        self.assertEqual(self.m.MODES['beacon'], (316227,))
        self.m.check('beacon', [1501], [316227], self.registry)
        for task, seed in ((1501, 618034), (1401, 316227)):
            with self.subTest(task=task, seed=seed), self.assertRaises(ValueError):
                self.m.check('beacon', [task], [seed], self.registry)
        _, fixture, plan = self.m.prepare(self.batch, 'official', ROOT, 1502, None, True, teacher='sensor', beacon=True)
        self.assertEqual(json.loads(fixture.read_text()), sensor_fixture(beacon_build(1502)))
        self.assertEqual(json.loads((self.batch / 'blueprints' / 'task-1502.json').read_text()), blueprint_of(beacon_build(1502)))
        self.assertEqual(json.loads(plan.read_text())[0]['params'], dict(until='build', hold_after_steps=self.m.HOLD_AFTER_STEPS))
        with self.assertRaises(ValueError): self.m.prepare(self.batch, 'official', ROOT, 1502, None, True, beacon=True)
        with self.assertRaises(ValueError): self.collect(mode='beacon', tasks=[1501], seeds=[316227], teacher='careful')
        self.assertEqual(self.collect(mode='beacon', tasks=[1501], seeds=[316227], teacher='sensor'), 0)
        teacher_plan = json.loads((self.batch / 'plans' / 'teacher.json').read_text())[0]
        self.assertEqual(teacher_plan['params']['hold_after_steps'], self.m.HOLD_AFTER_STEPS)
        frozen = json.loads((self.batch / 'frozen-teacher-c1501.json').read_text())
        self.assertTrue(any(k.replace('\\', '/').endswith('src/minecraftrl/build_beacon.py') for k in frozen['hashes']))
        self.assertIn('rule_select', frozen['scope'])
        _, _, plain = self.m.prepare(self.batch, 'official', ROOT, 103, None, True, teacher='sensor')
        self.assertEqual(json.loads(plain.read_text())[0]['params'], dict(until='build'))     # other modes keep their plan

    def test_the_occluded_mode_has_its_own_seed_and_tasks_and_sets_the_block(self):
        # Occlusion check (design 2026-11-15): recovering teacher only, a block in front of the first cell.
        from minecraftrl.build_recover import build2_fixture, occluded_build
        t = self.registry['tasks']
        self.assertFalse(set(t['occluded']) & (set(t['train']) | set(t['dev']) | set(t['final'])))
        self.assertTrue(all(occluded_build(k) is not None for k in t['occluded']))
        self.m.check('occluded', [401], [707106], self.registry)
        for mode, task, seed in (('occluded', 401, 130363), ('occluded', 301, 707106), ('dev', 401, 130363)):
            with self.subTest(mode=mode, task=task, seed=seed), self.assertRaises(ValueError):
                self.m.check(mode, [task], [seed], self.registry)
        _, fixture, _ = self.m.prepare(self.batch, 'official', ROOT, 401, None, True, teacher='recover', occluded=True)
        self.assertEqual(json.loads(fixture.read_text()), build2_fixture(occluded_build(401)))
        with self.assertRaises(ValueError): self.m.prepare(self.batch, 'official', ROOT, 401, None, True, occluded=True)
        with self.assertRaises(ValueError): self.collect(mode='occluded', tasks=[401], seeds=[707106])
        self.assertEqual(self.collect(mode='occluded', tasks=[401], seeds=[707106], teacher='recover'), 0)
        written = json.loads((self.batch / 'fixtures' / 'task-401.json').read_text())
        self.assertEqual(written['blocks'], [dict(dx=-2, dy=0, dz=3, block='cobblestone')])
        self.assertEqual(json.loads((self.batch / 'frozen-teacher-c401.json').read_text())['mode'], 'occluded')

    def test_round_4_modes_have_their_own_seeds_and_tasks_and_the_careful_teacher(self):
        # Round 4 (design 2026-11-15 careful place): occluded training tasks, dev4 and final4 with occluded halves.
        from minecraftrl.build_recover import build2_fixture, occluded_build
        from minecraftrl.build_task import generate_build
        t = self.registry['tasks']
        ids = [i for k in t for i in t[k]]
        self.assertEqual(len(ids), len(set(ids)))                                   # no task id in two modes
        for mode in self.m.OCCLUDED_MODES:
            self.assertTrue(all(occluded_build(k) is not None for k in t[mode]), mode)
        for mode, task, seed in (('occluded_train', 504, 271828), ('dev4', 601, 302585), ('dev4_occluded', 705, 302585),
                                 ('final4', 801, 866025), ('final4_occluded', 901, 866025)):
            self.m.check(mode, [task], [seed], self.registry)
        for mode, task, seed in (('occluded_train', 504, 130363), ('dev4', 601, 866025), ('final4_occluded', 705, 866025),
                                 ('train', 601, 271828)):
            with self.subTest(mode=mode, task=task, seed=seed), self.assertRaises(ValueError):
                self.m.check(mode, [task], [seed], self.registry)
        _, fixture, _ = self.m.prepare(self.batch, 'official', ROOT, 103, None, True, teacher='careful')
        self.assertEqual(json.loads(fixture.read_text()), build2_fixture(generate_build(103), teacher='careful'))
        self.assertEqual(self.collect(mode='dev4_occluded', tasks=[705], seeds=[302585], teacher='careful'), 0)
        written = json.loads((self.batch / 'fixtures' / 'task-705.json').read_text())
        self.assertEqual((written['build']['teacher'], len(written['blocks'])), ('careful', 1))
        self.assertEqual(self.collect(mode='dev4', tasks=[601], seeds=[302585], teacher='careful'), 0)
        self.assertEqual(json.loads((self.batch / 'fixtures' / 'task-601.json').read_text())['blocks'], [])
        with self.assertRaises(ValueError): self.collect(mode='occluded_train', tasks=[504], seeds=[271828])

    def test_the_large_mode_has_its_own_seed_and_tasks_and_the_larger_blueprints(self):
        # Size check (design 2026-11-15 build-large-check): tasks 1001-1008 on 693147, a recovering teacher only.
        from minecraftrl.build_large import large_build
        from minecraftrl.build_recover import build2_fixture
        self.assertEqual(self.registry['tasks']['large'], list(range(1001, 1009)))
        self.assertEqual(self.m.MODES['large'], (693147,))
        self.m.check('large', [1001], [693147], self.registry)
        for mode, task, seed in (('large', 1001, 866025), ('large', 801, 693147), ('final4', 1001, 866025)):
            with self.subTest(mode=mode, task=task, seed=seed), self.assertRaises(ValueError):
                self.m.check(mode, [task], [seed], self.registry)
        _, fixture, _ = self.m.prepare(self.batch, 'official', ROOT, 1004, None, True, teacher='careful', large=True)
        self.assertEqual(json.loads(fixture.read_text()), build2_fixture(large_build(1004), teacher='careful'))
        with self.assertRaises(ValueError): self.m.prepare(self.batch, 'official', ROOT, 1004, None, True, large=True)
        with self.assertRaises(ValueError): self.collect(mode='large', tasks=[1001], seeds=[693147])
        self.assertEqual(self.collect(mode='large', tasks=[1001], seeds=[693147], teacher='careful'), 0)
        written = json.loads((self.batch / 'fixtures' / 'task-1001.json').read_text())
        self.assertEqual(written['build']['cells'], large_build(1001)['cells'])
        frozen = json.loads((self.batch / 'frozen-teacher-c1001.json').read_text())
        self.assertEqual(frozen['mode'], 'large')
        self.assertTrue(any(k.replace('\\', '/').endswith('src/minecraftrl/build_large.py') for k in frozen['hashes']))

    def test_the_repair_modes_use_their_seeds_tasks_fixtures_and_open_slots_1_and_2(self):
        # Round 5 (design 2026-11-15 build-repair): repair tasks with strays; the repair teacher, or the careful one as a
        # reference; the policy may switch between the block and the pickaxe.
        from minecraftrl.build_repair import repair_build, repair_fixture
        t = self.registry['tasks']
        self.assertEqual((t['repair_train'], t['repair_dev'], t['repair_final']),
                         (list(range(1101, 1113)), list(range(1201, 1209)), list(range(1301, 1309))))
        self.assertEqual((self.m.MODES['repair_train'], self.m.MODES['repair_dev'], self.m.MODES['repair_final']),
                         (self.m.TRAIN_SEEDS, (414213,), (236067,)))
        for mode, task, seed in (('repair_dev', 1201, 236067), ('repair_final', 1201, 236067), ('repair_train', 1101, 414213)):
            with self.subTest(mode=mode, task=task, seed=seed), self.assertRaises(ValueError):
                self.m.check(mode, [task], [seed], self.registry)
        config, fixture, _ = self.m.prepare(self.batch, 'official', ROOT, 1203, None, True, teacher='repair', repair=True)
        self.assertEqual(json.loads(fixture.read_text()), repair_fixture(repair_build(1203)))
        value = json.loads(config.read_text())
        self.assertEqual(value['policy_tool_allowed'], ['hotbar.1', 'hotbar.2']); self.assertIs(value['policy_tool_lock'], True)
        _, fixture, _ = self.m.prepare(self.batch, 'c4', ROOT, 1203, None, True, teacher='careful', repair=True)
        self.assertEqual(json.loads(fixture.read_text()), repair_fixture(repair_build(1203), teacher='careful'))
        plain, _, _ = self.m.prepare(self.batch, 'plain', ROOT, 101, None, True, teacher='careful')
        self.assertNotIn('policy_tool_allowed', json.loads(plain.read_text()))
        for teacher, repair in (('recover', True), ('repair', False)):
            with self.subTest(teacher=teacher), self.assertRaises(ValueError):
                self.m.prepare(self.batch, 'x', ROOT, 1203, None, True, teacher=teacher, repair=repair)
        with self.assertRaises(ValueError): self.collect(mode='repair_train', tasks=[1101], seeds=[271828], teacher='careful2')
        with self.assertRaises(ValueError): self.collect(mode='repair_train', tasks=[1101], seeds=[271828])
        self.assertEqual(self.collect(mode='repair_train', tasks=[1101], seeds=[271828], teacher='repair'), 0)
        frozen = json.loads((self.batch / 'frozen-teacher-c1101.json').read_text())
        self.assertEqual((frozen['mode'], frozen['teacher']), ('repair_train', 'repair'))
        self.assertIn('no stray left', frozen['scope'])
        self.assertTrue(any(k.replace('\\', '/').endswith('src/minecraftrl/build_repair.py') for k in frozen['hashes']))

    def test_teacher_check_has_no_head_and_a_scripted_step(self):
        config, _, plan = self.m.prepare(self.batch, 'teacher', ROOT, 101, None, True, teacher_check=True)
        value = json.loads(config.read_text())
        self.assertIs(value['place_monitor'], True)
        self.assertFalse(any(s.get('reacquire') for s in value['skills'].values()))
        step = json.loads(plan.read_text())[0]
        self.assertEqual((step['kind'], step['params']['driver'], step['params']['limit_steps']), ('scripted', 'build_teacher', 1200))

    def test_collect_names_runs_records_inputs_and_skips_finished(self):
        self.assertEqual(self.collect(tasks=[101, 102], seeds=[271828, 314159]), 0)
        self.assertEqual([r.name for _, r in self.launched],
                         ['c101-s271828-n1', 'c101-s314159-n1', 'c102-s271828-n1', 'c102-s314159-n1'])
        self.assertTrue((self.batch / 'frozen-teacher-c101.json').exists())
        line = json.loads((self.batch / 'results.jsonl').read_text().splitlines()[0])
        self.assertEqual((line['task'], line['success'], line['filled'], line['wrong']), (101, True, 3, 1))
        command = self.launched[0][0]
        self.assertEqual(command[command.index('--limit-s') + 1], '60')
        self.assertEqual(self.collect(tasks=[101, 102], seeds=[271828, 314159]), 0)
        self.assertEqual(len(self.launched), 4)

    def test_policy_arm_needs_the_prompt_embedding_and_the_teacher_check_no_checkpoint(self):
        if self.registry['sha256']: self.skipTest('the embedding is made')
        with self.assertRaises(ValueError): self.collect(arm='official', teacher_check=False)
        with self.assertRaises(ValueError): self.collect(checkpoint=ROOT / 'configs/build-v1.json')
        self.assertEqual(self.launched, [])

    def test_report_counts_successes_cells_and_time(self):
        from minecraftrl.build_task import generate_build
        report = load_script('build_report')
        arm = Path(self.tmp.name) / 'arm'
        for name, done, stage in (('c101-s271828-n1', True, dict(status='completed', steps=200, filled=3, wrong=0)),
                                  ('c101-s271828-n2', False, dict(status='time_limit', filled=1, wrong=2)),
                                  ('c102-s271828-n1', False, dict(status='time_limit', filled=0, wrong=0))):
            (arm / name).mkdir(parents=True)
            (arm / name / 'summary.json').write_text(json.dumps(dict(execution_valid=True, all_steps_completed=done,
                                                                     stage_results=[stage])), encoding='utf-8')
        r = report.arm_report(arm)
        cells = sum(len(generate_build(t)['cells']) for t in (101, 101, 102))
        self.assertEqual((r['runs'], r['success'], r['other'], r['filled'], r['cells'], r['wrong'], r['median_s']),
                         (3, 1, {'time_limit': 2}, 4, cells, 2, 10.))
        self.assertEqual(sum(v[0] for v in r['by_shape'].values()), 3)
        self.assertIn('| arm | 1/3 |', report.markdown(Path('b'), dict(arm=r)))

if __name__ == '__main__':
    unittest.main()
