"""Jump course batch runner (design 2026-11-13 §4-§5): modes, files, the teacher check and the input record."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location('run_jump_batch', ROOT / 'scripts/run_jump_batch.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class JumpBatchTest(unittest.TestCase):
    def setUp(self):
        self.m = load_script(); self.registry = self.m.load_registry(ROOT)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.batch = ROOT / 'results' / ('jump-batch-test-' + Path(self.tmp.name).name)
        self.addCleanup(lambda: shutil.rmtree(self.batch, ignore_errors=True))
        self.launched = []

    def launch(self, command, run_dir):
        self.launched.append((command, run_dir))
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / 'summary.json').write_text(json.dumps(dict(
            execution_valid=True, status='completed', all_steps_completed=True, release='neutral_command_sent',
            stage_results=[dict(status='completed', best_platform=4, steps=300)],
            process_cleanup=dict(remaining_live_pids=[]))), encoding='utf-8')

    def collect(self, **kw):
        options = dict(batch=self.batch, arm='teacher', checkpoint=None, use_stage=True, mode='train', courses=[101],
                       seeds=[271828], sampling=[1], teacher_check=True, root=ROOT, launch=self.launch, gpu_free=lambda: 3000.)
        options.update(kw)
        return self.m.collect(**options)

    def test_course_splits_are_disjoint_and_modes_check_seeds_and_courses(self):
        c = self.registry['courses']
        self.assertFalse(set(c['train']) & set(c['dev']) or set(c['train']) & set(c['final']) or set(c['dev']) & set(c['final']))
        self.m.check('train', [101], [271828], self.registry); self.m.check('final', [301], [244948], self.registry)
        for mode, course, seed in (('train', 201, 271828), ('train', 101, 130363), ('dev', 101, 130363),
                                   ('final', 301, 130363), ('dev', 201, 244948)):
            with self.subTest(mode=mode, course=course, seed=seed), self.assertRaises(ValueError):
                self.m.check(mode, [course], [seed], self.registry)

    def test_policy_files_carry_the_course_the_jump_head_and_a_course_step(self):
        from minecraftrl.jump_course import course_fixture, generate_course
        config, fixture, plan = self.m.prepare(self.batch, 'official', ROOT, 103, None, True)
        self.assertEqual(json.loads(fixture.read_text()), course_fixture(generate_course(103)))
        self.assertEqual(fixture.name, 'course-103.json')
        skills = json.loads(config.read_text())['skills']
        self.assertEqual([n for n, s in skills.items() if s.get('reacquire')], ['jump_course'])
        self.assertEqual(skills['jump_course']['reacquire'], dict(checkpoint=None, use_stage=True, record_latents=True, stages='jump'))
        self.assertEqual(json.loads(plan.read_text()), [dict(id='course', skill='jump_course', kind='policy', target='course_goal',
                                                              count=1, params=dict(until='course'))])

    def test_teacher_check_has_no_head_and_a_scripted_step(self):
        config, _, plan = self.m.prepare(self.batch, 'teacher', ROOT, 101, None, True, teacher_check=True)
        skills = json.loads(config.read_text())['skills']
        self.assertFalse(any(s.get('reacquire') for s in skills.values())); self.assertNotIn('jump_course', skills)
        step = json.loads(plan.read_text())[0]
        self.assertEqual((step['kind'], step['params']['driver']), ('scripted', 'course_teacher'))

    def test_collect_names_runs_records_inputs_per_course_and_skips_finished(self):
        self.assertEqual(self.collect(courses=[101, 102], seeds=[271828, 314159]), 0)
        self.assertEqual([r.name for _, r in self.launched],
                         ['c101-s271828-n1', 'c101-s314159-n1', 'c102-s271828-n1', 'c102-s314159-n1'])
        self.assertTrue((self.batch / 'frozen-teacher-c101.json').exists())
        line = json.loads((self.batch / 'results.jsonl').read_text().splitlines()[0])
        self.assertEqual((line['course'], line['success'], line['best_platform']), (101, True, 4))
        command = self.launched[0][0]
        self.assertEqual(command[command.index('--limit-s') + 1], '90')
        self.assertEqual(self.collect(courses=[101, 102], seeds=[271828, 314159]), 0)
        self.assertEqual(len(self.launched), 4)

    def test_policy_arm_needs_the_prompt_embedding_and_the_teacher_check_no_checkpoint(self):
        if self.registry['sha256']: self.skipTest('the embedding is made')
        with self.assertRaises(ValueError): self.collect(arm='official', teacher_check=False)
        with self.assertRaises(ValueError): self.collect(checkpoint=ROOT / 'configs/jump-v1.json')
        self.assertEqual(self.launched, [])

    def test_report_counts_success_falls_and_jumps(self):
        spec = importlib.util.spec_from_file_location('jump_report', ROOT / 'scripts/jump_report.py')
        report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
        from minecraftrl.jump_course import generate_course
        arm = Path(self.tmp.name) / 'arm'
        for name, valid, done, stage in (('c101-s271828-n1', True, True, dict(status='completed', steps=400)),
                                         ('c101-s271828-n2', True, False, dict(status='course_fell', best_platform=1)),
                                         ('c102-s271828-n1', True, False, dict(status='time_limit', best_platform=0))):
            (arm / name).mkdir(parents=True)
            (arm / name / 'summary.json').write_text(json.dumps(dict(execution_valid=valid, all_steps_completed=done,
                                                                     stage_results=[stage])), encoding='utf-8')
        r = report.arm_report(arm)
        self.assertEqual((r['runs'], r['success'], r['fell'], r['other'], r['median_s']), (3, 1, 1, {'time_limit': 1}, 20.))
        self.assertEqual(r['best'], {'0': 1, '1': 1, '3': 1})
        p = generate_course(101)['platforms']
        first = 'gap %d, %+d' % (p[1]['gap'], p[1]['top'] - p[0]['top'])
        tried = sum(t for t, _ in r['jumps'].values()); made = sum(m for _, m in r['jumps'].values())
        self.assertEqual((tried, made), (3 + 2 + 1, 3 + 1))      # success: 3 jumps; fall after 1: 2 tried; time-out: 1
        self.assertGreaterEqual(r['jumps'][first][0], 2)


if __name__ == '__main__':
    unittest.main()
