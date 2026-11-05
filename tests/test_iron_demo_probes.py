"""Probe batch: an invalid launch is set aside and halts the batch instead of counting as a policy loss."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'scripts/run_iron_demo_probes.py'
spec = importlib.util.spec_from_file_location('iron_demo_probes_under_test', path)
probes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probes)


class ProbeBatchTests(unittest.TestCase):
    def run_batch(self, summaries, config_fixture=None):
        calls = []; self.fixtures = []

        def fake_run(argv, **_):
            output = Path(argv[argv.index('--output') + 1]); output.mkdir(parents=True)
            (output / 'summary.json').write_text(json.dumps(summaries[len(calls)]))
            self.fixtures.append(json.loads(Path(argv[argv.index('--fixture') + 1]).read_text()))
            calls.append(output)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'configs').mkdir()
            config = {'development_seed': 130363}
            if config_fixture: config['fixture'] = config_fixture
            (root / 'configs/iron-demo.json').write_text(json.dumps(config))
            (root / 'configs/iron-demo-fixture.json').write_text(json.dumps({'description': 'arena.'}))
            (root / 'configs/arena-v2.json').write_text(json.dumps({'description': 'bounded arena.'}))
            output = root / 'results/batch'
            argv = ['probes', '--root', str(root), '--output', str(output), '--only', 'log:wood_visual']
            with patch.object(sys, 'argv', argv), patch.object(probes, 'gpu_free_mib', return_value=3000.), \
                    patch.object(probes.subprocess, 'run', side_effect=fake_run):
                code = probes.main()
            lines = [json.loads(l) for l in (output / 'probe-results.jsonl').read_text().splitlines()] \
                if (output / 'probe-results.jsonl').exists() else []
            trials = sorted(p.name for p in (output / 'log-wood_visual').iterdir()) if (output / 'log-wood_visual').exists() else []
            excluded = sorted(str(p.relative_to(output / 'excluded')) for p in (output / 'excluded').glob('*/*')) \
                if (output / 'excluded').exists() else []
            return code, len(calls), lines, trials, excluded

    def test_invalid_launch_is_excluded_and_halts(self):
        code, calls, lines, trials, excluded = self.run_batch(
            [dict(status='worker_failed', execution_valid=False, error='worker exit -6')])
        self.assertEqual(code, 4)
        self.assertEqual(calls, 1)
        self.assertEqual(trials, [])
        self.assertEqual(len(excluded), 1)
        self.assertTrue(excluded[0].startswith('log-wood_visual'))
        self.assertEqual(lines[0]['excluded'], True)
        self.assertEqual(lines[0]['success'], False)

    def test_config_fixture_is_the_base_layout(self):
        done = dict(status='completed', execution_valid=True, all_steps_completed=True, stage_results=[])
        self.run_batch([done, done, done], config_fixture='configs/arena-v2.json')
        self.assertTrue(all(f['description'].startswith('bounded arena.') for f in self.fixtures))

    def test_valid_policy_failure_counts_and_continues(self):
        failed = dict(status='time_limit', execution_valid=True, all_steps_completed=False, stage_results=[])
        done = dict(status='completed', execution_valid=True, all_steps_completed=True, stage_results=[])
        code, calls, lines, trials, excluded = self.run_batch([failed, done, done])
        self.assertEqual(code, 0)
        self.assertEqual(calls, 3)
        self.assertEqual(trials, ['trial-1', 'trial-2', 'trial-3'])
        self.assertEqual(excluded, [])
        self.assertEqual([l['success'] for l in lines], [False, True, True])


if __name__ == '__main__':
    unittest.main()
