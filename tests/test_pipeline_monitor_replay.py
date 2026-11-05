import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.monitor_replay import (TimeEntry, fit_times, found_time, load_run, policy_rows, replay_step,
                                                 sample_from_row, unit_count)
from minecraftrl.pipeline.monitor_spec import parse_spec
from minecraftrl.pipeline.rm import sample_from


def started(sid, skill, t, start_step, *, target='iron_ore', count=1, block=None):
    return {'t': t, 'step': start_step, 'kind': 'step_started', 'id': sid, 'skill': skill, 'target': target,
            'count': count, 'block': block, 'status': 'active', 'start_step': start_step}


def finished(sid, skill, t, step, status='completed', *, target='iron_ore', count=1, block=None):
    return {'t': t, 'step': step, 'kind': 'step_finished', 'id': sid, 'skill': skill, 'target': target,
            'count': count, 'block': block, 'status': status}


def row(step, *, t=None, tick=None, pos=(0.5, 64., 0.5), yaw=0., pitch=0., keys=(), hit='air', inventory=None,
        controller='policy', available=1.):
    """One telemetry row in the recorded format; 20 ticks per second unless `t`/`tick` say otherwise."""
    tick = step if tick is None else tick
    action = {k: 1 for k in keys}; action['camera'] = [0., 0.]
    return {'t': step / 20. if t is None else t, 'step': step, 'world_time': float(tick), 'server_tick': float(tick),
            'server_x': pos[0], 'server_y': pos[1], 'server_z': pos[2], 'yaw': yaw, 'pitch': pitch, 'screen': 0.,
            'server_selected_item': 'air', 'server_health': 20., 'server_available': available,
            'hit_x': 3. if hit != 'air' else -999., 'hit_y': 64., 'hit_z': 1., 'held_key_count': float(len(keys)),
            'controller': controller, 'action': action, 'inventory': dict(inventory or {}),
            'demo': {'demo_available': 1., 'demo_destroys': 0., 'demo_last_destroyed_block': 'air',
                     'demo_last_destroy_harvestable': 0., 'demo_mining_block': 'air', 'demo_hit_block': hit}}


def write_run(directory, timeline, rows, seed=130363):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    (directory / 'request.json').write_text(json.dumps({'mode': 'probe', 'seed': seed}), encoding='utf-8')
    for name, items in (('timeline.jsonl', timeline), ('telemetry.jsonl', rows)):
        (directory / name).write_text(''.join(json.dumps(x) + '\n' for x in items), encoding='utf-8')
    return directory


class LoadRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_steps_from_timeline(self):
        timeline = [{'t': 0.5, 'step': 0, 'kind': 'session_started'},
                    started('logs', 'wood_visual', 1., 10, target='log', count=4, block='log'),
                    {'t': 2., 'step': 20, 'kind': 'monitor_event', 'step_id': 'logs', 'event': {'kind': 'no_progress'}},
                    finished('logs', 'wood_visual', 5., 50, target='log', count=4, block='log'),
                    started('craft-stick', 'craft', 5.1, 50, target='stick'),
                    finished('craft-stick', 'craft', 6., 60, 'time_limit', target='stick')]
        run = load_run(write_run(self.root / 'A1', timeline, []))
        self.assertEqual(run.name, 'A1')
        self.assertEqual(len(run.steps), 2)
        logs, craft = run.steps
        self.assertEqual((logs.id, logs.skill, logs.target, logs.block, logs.count), ('logs', 'wood_visual', 'log', 'log', 4))
        self.assertEqual((logs.start_t, logs.end_t, logs.start_step, logs.end_step, logs.status), (1., 5., 10, 50, 'completed'))
        self.assertEqual((craft.start_step, craft.end_step, craft.status, craft.run), (50, 60, 'time_limit', 'A1'))

    def test_restarted_step_id_is_new_step(self):
        timeline = [started('iron', 'mine_iron_text', 1., 10), finished('iron', 'mine_iron_text', 2., 20, 'anomaly_hold'),
                    started('iron', 'mine_iron_text', 3., 20), finished('iron', 'mine_iron_text', 4., 40)]
        run = load_run(write_run(self.root / 'A2', timeline, []))
        self.assertEqual([(s.id, s.status, s.start_step) for s in run.steps],
                         [('iron', 'anomaly_hold', 10), ('iron', 'completed', 20)])

    def test_unfinished_step_kept_separate(self):
        timeline = [started('iron', 'mine_iron_text', 1., 10),
                    {'t': 9., 'step': 90, 'kind': 'stop_acknowledged'}]
        step, = load_run(write_run(self.root / 'A3', timeline, [])).steps
        self.assertEqual((step.status, step.end_t, step.end_step), ('unfinished', 9., None))

    def test_seed_read_from_request(self):
        self.assertEqual(load_run(write_run(self.root / 'E1', [], [], seed=104729)).seed, 104729)

    def test_sample_from_row_matches_live_adapter(self):
        r = row(30, pos=(1.5, 70., -2.5), yaw=45., pitch=10., keys=('forward', 'attack'), hit='iron_ore',
                inventory={'stone_pickaxe': 1, 'dirt': 0})
        live = sample_from({'workshop': {**r, 'server_inventory': r['inventory']}, 'demo': r['demo']}, r['action'], r['t'])
        replayed = sample_from_row(r)
        for field in ('position', 'inventory', 'hit_block', 'valid', 'action_keys', 'tick', 't'):
            self.assertEqual(getattr(replayed, field), getattr(live, field), field)
        self.assertEqual(replayed.inventory, {'stone_pickaxe': 1})
        self.assertFalse(sample_from_row(row(31, available=0.)).valid)

    def test_policy_rows_exclude_rule_and_other_steps(self):
        timeline = [started('iron', 'mine_iron_text', 0.5, 10), finished('iron', 'mine_iron_text', 1., 20),
                    started('craft', 'craft', 1., 20), finished('craft', 'craft', 1.5, 30)]
        rows = [row(s, controller='rule' if s in (12, 25) else 'policy') for s in range(5, 35)]
        run = load_run(write_run(self.root / 'A4', timeline, rows))
        self.assertEqual([r['step'] for r in policy_rows(run, run.steps[0])], [s for s in range(10, 20) if s != 12])
        unfinished = load_run(write_run(self.root / 'A5', [started('iron', 'mine_iron_text', 0.5, 10)], rows))
        self.assertEqual([r['step'] for r in policy_rows(unfinished, unfinished.steps[0])][-1], 34)


SPEC = parse_spec({
    'version': 'replay-test', 'author': 'human', 'margin': 0.2,
    'subgoals': {
        'craft': {'ending': 'bounded', 'unit': 'step'},
        'climb_out_text': {'ending': 'bounded', 'unit': 'block'},
        'mine_iron_text': {'ending': 'until_found', 'unit': 'item', 'found': 'iron_ore',
                           'anomaly': ['no_progress', 'repeated_cycle', 'stationary_repetition'],
                           'waste': ['moving_but_stuck', 'looking_down', 'revisiting_area']},
    },
    'constraints': [],
})
AFTER_FOUND = {'mine_iron_text:after_found': TimeEntry(3, 2., 5.)}


def steps_run(root, name, items, seed=130363):
    """Sequential steps without telemetry: items are (skill, duration_s, status, count, target)."""
    timeline, t = [], 0.
    for i, (skill, duration, status, count, target) in enumerate(items):
        sid = f'{skill}-{i}'
        timeline += [started(sid, skill, t, 10 * i, target=target, count=count),
                     finished(sid, skill, t + duration, 10 * i + 10, status, target=target, count=count)]
        t += duration
    return load_run(write_run(Path(root) / name, timeline, [], seed))


def search_run(root, name, rows, status='time_limit', seed=130363):
    """One mine_iron_text step spanning the rows; it ends one tick after the last row."""
    first, last = rows[0], rows[-1]
    timeline = [started('iron', 'mine_iron_text', first['t'], first['step'], block='iron_ore'),
                finished('iron', 'mine_iron_text', last['t'] + .05, last['step'] + 1, status, block='iron_ore')]
    return load_run(write_run(Path(root) / name, timeline, rows, seed))


def moving(ticks, *, hit_from=None, stride=2):
    """Walking east half a block per tick: a new 2-block cell every few ticks, so the detector sees progress."""
    return [row(k, pos=(0.5 + 0.5 * k, 64., 0.5), keys=('forward',),
                hit='iron_ore' if hit_from is not None and k >= hit_from else 'air') for k in range(1, ticks, stride)]


def stationary(ticks, *, start=1, stride=2):
    return [row(k, keys=('forward',)) for k in range(start, start + ticks, stride)]


def replay(run, margin=0.2, table=None):
    step = run.steps[0]
    return replay_step(run, step, SPEC.for_step(step.skill, step.target), AFTER_FOUND if table is None else table, margin)


class FitAndReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def crafts(self, name, durations, status='completed'):
        return steps_run(self.root, name, [('craft', d, status, 1, 'stick') for d in durations])

    def test_bounded_check_fires_after_worst_plus_margin(self):
        table, insufficient = fit_times([self.crafts('dev', [10., 12., 20.])], SPEC)
        self.assertEqual(table['craft'], TimeEntry(3, 10., 20.))
        self.assertNotIn('craft', insufficient)
        late = replay(self.crafts('slow', [30.], 'time_limit'), table=table)
        self.assertAlmostEqual(late['check_s'], 24.)
        self.assertAlmostEqual(late['saved_s'], 6.)
        self.assertEqual((late['check_reason'], late['ending'], late['note']), ('over_check_time', 'bounded', None))
        within = replay(self.crafts('ok', [23.]), table=table)
        self.assertIsNone(within['check_s']); self.assertIsNone(within['saved_s'])

    def test_per_unit_normalisation(self):
        dev = steps_run(self.root, 'dev', [('climb_out_text', 9., 'completed', 3, 'height'),
                                           ('climb_out_text', 12., 'completed', 6, 'height'),
                                           ('climb_out_text', 8., 'completed', 2, 'height')])
        table, _ = fit_times([dev], SPEC)
        self.assertEqual(table['climb_out_text'], TimeEntry(3, 2., 4.))
        climb = steps_run(self.root, 'eval', [('climb_out_text', 30., 'time_limit', 5, 'height')])
        self.assertEqual(unit_count(climb.steps[0], SPEC.for_step('climb_out_text', 'height')), 5)
        self.assertAlmostEqual(replay(climb, table=table)['check_s'], 24.)

    def test_fit_needs_min_records(self):
        table, insufficient = fit_times([self.crafts('dev', [10., 12.])], SPEC)
        self.assertNotIn('craft', table)
        self.assertEqual(insufficient['craft'], 2)
        self.assertEqual(insufficient['mine_iron_text:after_found'], 0)
        self.assertEqual(replay(self.crafts('eval', [99.]), table=table)['note'], 'no_records')

    def test_fit_ignores_non_completed(self):
        dev = steps_run(self.root, 'dev', [('craft', d, s, 1, 'stick') for d, s in
                                           ((10., 'completed'), (100., 'time_limit'), (12., 'completed'),
                                            (50., 'anomaly_hold'), (20., 'completed'))])
        self.assertEqual(fit_times([dev], SPEC)[0]['craft'], TimeEntry(3, 10., 20.))

    def test_fit_after_found_uses_time_after_target_seen(self):
        runs = [search_run(self.root, f'dev{i}', moving(tail, hit_from=600), 'completed') for i, tail in
                enumerate((640, 660, 700))]
        self.assertAlmostEqual(found_time(policy_rows(runs[0], runs[0].steps[0]), runs[0].steps[0]), 30.05)
        table, _ = fit_times(runs, SPEC)
        self.assertNotIn('mine_iron_text', table)
        entry = table['mine_iron_text:after_found']
        self.assertEqual(entry.n, 3)
        # Last rows at ticks 639/659/699 end the steps at 32.0/33.0/35.0 s; the target was first seen at 30.05 s.
        self.assertAlmostEqual(entry.best, 1.95)
        self.assertAlmostEqual(entry.worst, 4.95)

    def test_until_found_has_no_time_check_before_found(self):
        result = replay(search_run(self.root, 'eval', moving(12000)))
        self.assertGreater(result['duration_s'], 599.)
        self.assertIsNone(result['found_s'])
        self.assertIsNone(result['check_s'])

    def test_until_found_stall_check(self):
        result = replay(search_run(self.root, 'eval', stationary(1200)))
        self.assertEqual(result['check_reason'], 'stationary_repetition')
        self.assertAlmostEqual(result['check_s'], 20.)

    def test_found_switches_to_bounded(self):
        result = replay(search_run(self.root, 'eval', moving(1200, hit_from=600)))
        self.assertAlmostEqual(result['found_s'], 30.)
        self.assertAlmostEqual(result['check_s'], 36.)
        self.assertEqual(result['check_reason'], 'over_check_time')

    def test_observation_gap_resets_stall(self):
        jump = stationary(300) + stationary(300, start=351)
        unavailable = (stationary(300) + [row(k, keys=('forward',), available=0.) for k in range(303, 351, 2)]
                       + stationary(300, start=351))
        for name, rows in (('jump', jump), ('unavailable', unavailable)):
            with self.subTest(name=name):
                self.assertIsNone(replay(search_run(self.root, name, rows))['check_s'])

    def test_waste_fractions(self):
        rows = [row(k, keys=('forward',), pitch=70. if k >= 120 else 0.) for k in range(160)]
        waste = replay(search_run(self.root, 'stuck', rows))['waste']
        self.assertEqual(waste, {'moving_but_stuck': 0.75, 'looking_down': 0.25, 'revisiting_area': 0.})
        walk = [row(k, pos=(0.1 * k, 64., 0.5), keys=('forward',)) for k in range(80)]
        self.assertEqual(replay(search_run(self.root, 'walk', walk))['waste']['moving_but_stuck'], 0.)
        # Out to x=5.85 through cells 0, 1, 2 and back: rows 53-79 walk into cells 1 and 0 again.
        back = [row(k, pos=(0.15 * (k if k < 40 else 79 - k), 64., 0.5), keys=('forward',)) for k in range(80)]
        self.assertAlmostEqual(replay(search_run(self.root, 'back', back))['waste']['revisiting_area'], 27 / 80)

    def test_step_without_spec_reported(self):
        run = steps_run(self.root, 'eval', [('teleport', 5., 'completed', 1, 'home')])
        result = replay_step(run, run.steps[0], SPEC.for_step('teleport', 'home'), AFTER_FOUND, 0.2)
        self.assertEqual((result['note'], result['check_s'], result['skill']), ('no_spec', None, 'teleport'))


if __name__ == '__main__':
    unittest.main()
