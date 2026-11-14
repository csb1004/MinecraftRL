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
