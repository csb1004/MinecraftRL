import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location('run_reacquire_batch', ROOT / 'scripts/run_reacquire_batch.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.m = load_script()
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.batch = ROOT / 'results' / ('reacquire-batch-test-' + Path(self.tmp.name).name)
        self.addCleanup(lambda: __import__('shutil').rmtree(self.batch, ignore_errors=True))
        self.launched = []

    def launch(self, valid=True):
        def run(command, run_dir):
            self.launched.append(run_dir)
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / 'summary.json').write_text(json.dumps(dict(
                execution_valid=valid, status='completed', all_steps_completed=valid, steps=10,
                release='neutral_command_sent', process_cleanup=dict(remaining_live_pids=[]))), encoding='utf-8')
        return run

    def collect(self, **kw):
        options = dict(batch=self.batch, arm='official', checkpoint=None, use_stage=True, seeds=[271828], sampling=[1, 2],
                       mode='train', root=ROOT, launch=self.launch(), gpu_free=lambda: 3000.)
        options.update(kw)
        return self.m.collect(**options)

    def test_collect_refuses_seeds_outside_the_mode(self):
        for mode, seed in (('train', 130363), ('train', 577215), ('dev', 271828), ('dev', 577215), ('final', 130363)):
            with self.subTest(mode=mode, seed=seed), self.assertRaises(ValueError): self.m.check_seeds(mode, [seed])
        self.m.check_seeds('dev', [130363]); self.m.check_seeds('final', [577215]); self.m.check_seeds('train', [314159])

    def test_arm_config_only_changes_the_reacquire_block(self):
        base = json.loads((ROOT / 'configs/reacquire-v1.json').read_text(encoding='utf-8'))
        config = self.m.arm_config(base, 'results/x/checkpoint.pt', False)
        block = config['skills']['mine_iron_reacquire'].pop('reacquire')
        base['skills']['mine_iron_reacquire'].pop('reacquire')
        self.assertEqual(config, base)
        self.assertEqual(block, dict(checkpoint='results/x/checkpoint.pt', use_stage=False, record_latents=True))

    def test_finished_trials_are_skipped_and_stop_halts(self):
        self.assertEqual(self.collect(), 0); self.assertEqual(len(self.launched), 2)
        self.assertEqual(self.collect(), 0); self.assertEqual(len(self.launched), 2)
        (self.batch / 'STOP').touch()
        self.assertEqual(self.collect(sampling=[3]), 1); self.assertEqual(len(self.launched), 2)
        lines = (self.batch / 'results.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 2); self.assertTrue(json.loads(lines[0])['success'])

    def test_invalid_launch_is_moved_aside_and_halts(self):
        self.assertEqual(self.collect(launch=self.launch(valid=False)), 4)
        self.assertEqual(len(self.launched), 1)
        self.assertTrue(any((self.batch / 'excluded').rglob('summary.json')))
        self.assertFalse((self.batch / 'official/s271828-n1').exists())

    def test_changed_inputs_halt_the_batch(self):
        self.assertEqual(self.collect(sampling=[1]), 0)
        frozen = self.batch / 'frozen-official.json'
        record = json.loads(frozen.read_text()); record['hashes'][next(iter(record['hashes']))] = '0' * 64
        frozen.write_text(json.dumps(record))
        self.assertEqual(self.collect(sampling=[2]), 5); self.assertEqual(len(self.launched), 1)

    def test_busy_gpu_halts_without_launching(self):
        self.assertEqual(self.collect(gpu_free=lambda: 100., gpu_wait=(1, 0.)), 3); self.assertEqual(self.launched, [])

    def test_probe_files_match_the_2026_11_04_iron_probe(self):
        self.collect(sampling=[1])
        plan = json.loads((self.batch / 'plans/iron.json').read_text())
        self.assertEqual(plan, [dict(id='iron', skill='mine_iron_reacquire', kind='policy', target='iron_ore', count=1,
                                     block='iron_ore')])
        fixture = json.loads((self.batch / 'fixtures/iron-probe.json').read_text())
        self.assertEqual(fixture['give'], [dict(item='stone_pickaxe', count=1)])
        base = json.loads((ROOT / 'configs/iron-demo-fixture-v2.json').read_text())
        self.assertEqual({k: v for k, v in fixture.items() if k not in ('give', 'description')},
                         {k: v for k, v in base.items() if k not in ('give', 'description')})

    # Multi-block experiment (design 2026-11-08 §3): the iron ores of fixture v2 swapped for a training block.
    def test_iron_files_are_unchanged_by_the_block_option(self):
        default = self.m.prepare(self.batch / 'a', 'official', ROOT, None, True)
        explicit = self.m.prepare(self.batch / 'b', 'official', ROOT, None, True, block='iron_ore')
        for x, y in zip(default, explicit):
            self.assertEqual(x.name, y.name); self.assertEqual(x.read_bytes(), y.read_bytes())
        base = json.loads((ROOT / 'configs/reacquire-v1.json').read_text(encoding='utf-8'))
        self.assertEqual(json.loads(default[0].read_text()), self.m.arm_config(base, None, True))

    def test_block_fixture_replaces_only_the_iron_ores(self):
        base = json.loads((ROOT / 'configs/iron-demo-fixture-v2.json').read_text())
        _, fixture, _ = self.m.prepare(self.batch, 'official', ROOT, None, True, block='coal_ore')
        self.assertEqual(fixture.name, 'coal_ore-probe.json')
        got = json.loads(fixture.read_text())
        self.assertEqual(got['give'], [dict(item='stone_pickaxe', count=1)])
        self.assertEqual(got['fill'], base['fill'])
        want = [dict(b, block='coal_ore') if b['block'] == 'iron_ore' else b for b in base['blocks']]
        self.assertEqual(got['blocks'], want)
        self.assertEqual(sum(b['block'] == 'coal_ore' for b in got['blocks']), 2)

    def test_block_config_has_one_reacquire_skill(self):
        registry = json.loads((ROOT / 'configs/multi-block-v1.json').read_text(encoding='utf-8'))
        config_path, _, _ = self.m.prepare(self.batch, 'r1', ROOT, '/x/checkpoint.pt', True, block='coal_ore')
        self.assertEqual(config_path.name, 'r1-coal_ore.json')
        config = json.loads(config_path.read_text())
        carrying = [name for name, spec in config['skills'].items() if spec.get('reacquire')]
        self.assertEqual(carrying, ['mine_coal_ore_reacquire'])
        spec = config['skills']['mine_coal_ore_reacquire']
        entry = registry['blocks']['coal_ore']
        self.assertEqual(spec, dict(embedding=entry['embedding'], sha256=entry['sha256'], cond_scale=6.0,
                                    source='text_prompt:mine coal ore', target_block='coal_ore', target_items=['coal'],
                                    reacquire=dict(checkpoint='/x/checkpoint.pt', use_stage=True, record_latents=True)))

    def test_block_plan_run_names_and_frozen_file(self):
        self.assertEqual(self.collect(sampling=[1], block='coal_ore'), 0)
        plan = json.loads((self.batch / 'plans/coal_ore.json').read_text())
        self.assertEqual(plan, [dict(id='coal_ore', skill='mine_coal_ore_reacquire', kind='policy', target='coal', count=1,
                                     block='coal_ore')])
        self.assertEqual(self.launched, [self.batch / 'official/coal_ore-s271828-n1'])
        self.assertTrue((self.batch / 'frozen-official-coal_ore.json').exists())
        self.assertFalse((self.batch / 'frozen-official.json').exists())
        line = json.loads((self.batch / 'results.jsonl').read_text().splitlines()[0])
        self.assertEqual((line['block'], line['run']), ('coal_ore', 'coal_ore-s271828-n1'))

    def test_two_blocks_share_an_arm_without_tripping_the_input_check(self):
        self.assertEqual(self.collect(sampling=[1], block='coal_ore'), 0)
        self.assertEqual(self.collect(sampling=[1], block='sand'), 0)
        self.assertEqual(len(self.launched), 2)

    def test_unknown_or_held_out_block_names_are_refused(self):
        for block in ('diamond_ore', 'stone'):
            with self.subTest(block=block), self.assertRaises(ValueError):
                self.m.prepare(self.batch, 'official', ROOT, None, True, block=block)


if __name__ == '__main__':
    unittest.main()
