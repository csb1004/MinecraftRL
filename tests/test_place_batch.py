"""Block placing batch runner (design 2026-11-14 §5): modes, files, the teacher check, the input record, the report."""
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


class PlaceBatchTest(unittest.TestCase):
    def setUp(self):
        self.m = load_script('run_place_batch'); self.registry = self.m.load_registry(ROOT)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.batch = ROOT / 'results' / ('place-batch-test-' + Path(self.tmp.name).name)
        self.addCleanup(lambda: shutil.rmtree(self.batch, ignore_errors=True))
        self.launched = []

    def launch(self, command, run_dir):
        self.launched.append((command, run_dir))
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / 'summary.json').write_text(json.dumps(dict(
            execution_valid=True, status='completed', all_steps_completed=True, release='neutral_command_sent',
            stage_results=[dict(status='completed', wrong=1, steps=80)], process_cleanup=dict(remaining_live_pids=[]))),
            encoding='utf-8')

    def collect(self, **kw):
        options = dict(batch=self.batch, arm='teacher', checkpoint=None, use_stage=True, mode='train', targets=[101],
                       seeds=[271828], sampling=[1], teacher_check=True, root=ROOT, launch=self.launch, gpu_free=lambda: 3000.)
        options.update(kw)
        return self.m.collect(**options)

    def test_target_splits_are_disjoint_and_modes_check_seeds_and_targets(self):
        t = self.registry['targets']
        self.assertFalse(set(t['train']) & set(t['dev']) or set(t['train']) & set(t['final']) or set(t['dev']) & set(t['final']))
        self.m.check('train', [101], [271828], self.registry); self.m.check('final', [301], [357913], self.registry)
        for mode, target, seed in (('train', 201, 271828), ('train', 101, 130363), ('dev', 101, 130363),
                                   ('final', 301, 130363), ('dev', 201, 357913)):
            with self.subTest(mode=mode, target=target, seed=seed), self.assertRaises(ValueError):
                self.m.check(mode, [target], [seed], self.registry)

    def test_policy_files_carry_the_task_the_place_head_the_monitor_and_a_place_step(self):
        from minecraftrl.place_task import generate_target, place_fixture
        config, fixture, plan = self.m.prepare(self.batch, 'official', ROOT, 103, None, True)
        self.assertEqual(json.loads(fixture.read_text()), place_fixture(generate_target(103)))
        self.assertEqual(fixture.name, 'target-103.json')
        value = json.loads(config.read_text())
        self.assertIs(value['place_monitor'], True)
        skills = value['skills']
        self.assertEqual([n for n, s in skills.items() if s.get('reacquire')], ['place_block'])
        self.assertEqual(skills['place_block']['reacquire'], dict(checkpoint=None, use_stage=True, record_latents=True, stages='place'))
        self.assertEqual(json.loads(plan.read_text()), [dict(id='place', skill='place_block', kind='policy', target='place_target',
                                                              count=1, params=dict(until='place'))])

    def test_teacher_check_has_no_head_and_a_scripted_step(self):
        config, _, plan = self.m.prepare(self.batch, 'teacher', ROOT, 101, None, True, teacher_check=True)
        value = json.loads(config.read_text())
        self.assertIs(value['place_monitor'], True)
        self.assertFalse(any(s.get('reacquire') for s in value['skills'].values()))
        step = json.loads(plan.read_text())[0]
        self.assertEqual((step['kind'], step['params']['driver']), ('scripted', 'place_teacher'))

    def test_collect_names_runs_records_inputs_and_skips_finished(self):
        self.assertEqual(self.collect(targets=[101, 102], seeds=[271828, 314159]), 0)
        self.assertEqual([r.name for _, r in self.launched],
                         ['t101-s271828-n1', 't101-s314159-n1', 't102-s271828-n1', 't102-s314159-n1'])
        self.assertTrue((self.batch / 'frozen-teacher-t101.json').exists())
        line = json.loads((self.batch / 'results.jsonl').read_text().splitlines()[0])
        self.assertEqual((line['target'], line['success'], line['wrong']), (101, True, 1))
        command = self.launched[0][0]
        self.assertEqual(command[command.index('--limit-s') + 1], '45')
        self.assertEqual(self.collect(targets=[101, 102], seeds=[271828, 314159]), 0)
        self.assertEqual(len(self.launched), 4)

    def test_policy_arm_needs_the_prompt_embedding_and_the_teacher_check_no_checkpoint(self):
        if self.registry['sha256']: self.skipTest('the embedding is made')
        with self.assertRaises(ValueError): self.collect(arm='official', teacher_check=False)
        with self.assertRaises(ValueError): self.collect(checkpoint=ROOT / 'configs/place-v1.json')
        self.assertEqual(self.launched, [])

    def test_report_counts_successes_wrong_placements_and_time(self):
        report = load_script('place_report')
        arm = Path(self.tmp.name) / 'arm'
        for name, done, stage in (('t101-s271828-n1', True, dict(status='completed', steps=40, wrong=0)),
                                  ('t101-s271828-n2', False, dict(status='time_limit', wrong=2)),
                                  ('t102-s271828-n1', True, dict(status='completed', steps=60, wrong=1))):
            (arm / name).mkdir(parents=True)
            (arm / name / 'summary.json').write_text(json.dumps(dict(execution_valid=True, all_steps_completed=done,
                                                                     stage_results=[stage])), encoding='utf-8')
        r = report.arm_report(arm)
        self.assertEqual((r['runs'], r['success'], r['wrong'], r['other'], r['median_s']), (3, 2, 3, {'time_limit': 1}, 2.5))
        from minecraftrl.place_task import generate_target
        want = {}
        for target, ok in ((101, True), (101, False), (102, True)):
            dy = str(generate_target(target)['target'][1]); tried, made = want.get(dy, (0, 0)); want[dy] = (tried + 1, made + ok)
        self.assertEqual({k: tuple(v) for k, v in r['by_height'].items()}, want)


if __name__ == '__main__':
    unittest.main()
