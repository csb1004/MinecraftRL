import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.monitor_replay import load_run
from minecraftrl.pipeline.monitor_signals import calibrated_spec, flag_signals, signal_rates
from minecraftrl.pipeline.monitor_spec import parse_spec
from test_pipeline_monitor_replay import finished, moving, row, started, stationary, write_run

RAW = {'version': 'sig-test', 'author': 'llm:x', 'margin': .2, 'constraints': [],
       'subgoals': {'climb_out_text': {'ending': 'bounded', 'unit': 'block', 'anomaly': ['stationary_repetition'],
                                       'waste': ['looking_down', 'moving_but_stuck']},
                    'mine_iron_text': {'ending': 'until_found', 'unit': 'item', 'found': 'iron_ore',
                                       'anomaly': ['repeated_cycle', 'harvest_without_drop'], 'waste': ['looking_down']}}}
SPEC = parse_spec(RAW)


def one_step(root, name, skill, rows, status='completed', events=(), target='height', block=None):
    first, last = rows[0], rows[-1]
    timeline = [started('s', skill, first['t'], first['step'], target=target, block=block), *events,
                finished('s', skill, last['t'] + .05, last['step'] + 1, status, target=target, block=block)]
    return load_run(write_run(Path(root) / name, timeline, rows))


class SignalRatesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def climb(self, name, status='completed'):
        """Pillaring: looking down and jumping in place for 450 ticks (no movement keys)."""
        rows = [row(k, keys=('jump',), pitch=80.) for k in range(1, 451, 2)]
        return one_step(self.root, name, 'climb_out_text', rows, status)

    def test_rates_use_completed_steps_only(self):
        runs = [self.climb('c1'), self.climb('c2'), self.climb('t1', 'time_limit')]
        rates = signal_rates(runs, SPEC)['climb_out_text']
        self.assertEqual(rates['completed_steps'], 2)
        self.assertAlmostEqual(rates['waste']['looking_down'], 1.)
        self.assertEqual(rates['anomaly']['stationary_repetition'], {'rate': 1., 'source': 'replayed_stall_detector'})

    def test_recorded_monitor_events_count_for_other_anomalies(self):
        event = {'t': 1., 'step': 20, 'kind': 'monitor_event', 'step_id': 's', 'event': {'kind': 'harvest_without_drop'}}
        runs = [one_step(self.root, 'i1', 'mine_iron_text', moving(200, hit_from=100), events=[event],
                         target='iron_ore', block='iron_ore'),
                one_step(self.root, 'i2', 'mine_iron_text', moving(200, hit_from=100), target='iron_ore', block='iron_ore')]
        rates = signal_rates(runs, SPEC)['mine_iron_text']
        self.assertEqual(rates['anomaly']['harvest_without_drop'], {'rate': .5, 'source': 'recorded_events'})
        self.assertEqual(rates['anomaly']['repeated_cycle']['rate'], 0.)

    def test_flags_and_calibrated_spec(self):
        rates = signal_rates([self.climb('c1'), self.climb('c2')], SPEC)
        flags = flag_signals(rates, waste_max=.5, anomaly_max=.2)
        self.assertEqual({(f['key'], f['role'], f['signal']) for f in flags},
                         {('climb_out_text', 'waste', 'looking_down'), ('climb_out_text', 'anomaly', 'stationary_repetition')})
        raw = calibrated_spec(RAW, flags, version='sig-test-cal')
        spec = parse_spec(raw)
        climb = spec.for_step('climb_out_text', 'height')
        self.assertEqual((climb.anomaly, climb.waste), ((), ('moving_but_stuck',)))
        self.assertEqual(spec.version, 'sig-test-cal')
        self.assertEqual(RAW['subgoals']['climb_out_text']['waste'], ['looking_down', 'moving_but_stuck'])  # input kept

    def test_subgoal_without_completed_steps_is_not_flagged(self):
        rates = signal_rates([self.climb('t1', 'time_limit')], SPEC)
        self.assertEqual(rates['climb_out_text']['completed_steps'], 0)
        self.assertEqual(flag_signals(rates, waste_max=.5, anomaly_max=.2), [])


def load_script():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'check_monitor_signals.py'
    spec = importlib.util.spec_from_file_location('check_monitor_signals_under_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class ScriptTest(unittest.TestCase):
    def test_writes_rates_flags_and_calibrated_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = root / 'spec.json'; spec_path.write_text(json.dumps(RAW), encoding='utf-8')
            rows = [row(k, keys=('jump',), pitch=80.) for k in range(1, 451, 2)]
            runs = [one_step(root, f'c{i}', 'climb_out_text', rows).path for i in range(2)]
            out = root / 'out'
            args = ['--spec', str(spec_path), '--fit', *map(str, runs), '--out', str(out), '--version', 'cal-1']
            self.assertEqual(load_script().main(args), 0)
            report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['flags']), 2)
            self.assertEqual((report['limits']['waste_max'], report['limits']['anomaly_max']), (.5, .2))
            self.assertEqual(parse_spec(json.loads((out / 'spec.json').read_text(encoding='utf-8'))).version, 'cal-1')
            self.assertEqual(load_script().main(args), 2)   # output folder exists


if __name__ == '__main__':
    unittest.main()
