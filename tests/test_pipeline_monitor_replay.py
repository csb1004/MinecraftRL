import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.monitor_replay import (ReplayOptions, TimeEntry, engagements, evaluate, fit_times, found_time,
                                                 load_run, policy_rows, replay_step, sample_from_row, stall_fidelity,
                                                 unit_count)
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
        timeline = [started('iron', 'mine_iron_text', 0.5, 10), finished('iron', 'mine_iron_text', 1.01, 20),
                    started('craft', 'craft', 1.01, 20), finished('craft', 'craft', 1.51, 30)]
        rows = [row(s, controller='rule' if s in (12, 25) else 'policy') for s in range(5, 35)]
        run = load_run(write_run(self.root / 'A4', timeline, rows))
        # Row 20 is the observation that finished the first step (recorded at 1.0 s, before the next step started).
        self.assertEqual([r['step'] for r in policy_rows(run, run.steps[0])], [s for s in range(10, 21) if s != 12])
        self.assertEqual([r['step'] for r in policy_rows(run, run.steps[1])], [s for s in range(21, 31) if s != 25])
        unfinished = load_run(write_run(self.root / 'A5', [started('iron', 'mine_iron_text', 0.5, 10)], rows))
        self.assertEqual([r['step'] for r in policy_rows(unfinished, unfinished.steps[0])][-1], 34)

    def test_row_before_hold_stays_with_held_step(self):
        # Recorded 2026-11-05 r3/E02: a hold freezes the step counter while the LLM answers, so the next step starts
        # at the same step number 15 s later. The last pre-hold row belongs to the held step only.
        timeline = [started('iron', 'mine_iron_text', 0.5, 10), finished('iron', 'mine_iron_text', 16., 20, 'anomaly_hold'),
                    started('logs', 'wood_visual', 16., 20, target='log', block='log'),
                    finished('logs', 'wood_visual', 17., 40, target='log', block='log')]
        rows = [row(s, hit='oak_log' if s == 20 else 'air') for s in range(10, 21)] + \
            [row(s, t=16. + (s - 20) * .05) for s in range(21, 40)]
        run = load_run(write_run(self.root / 'held', timeline, rows))
        iron, logs = run.steps
        self.assertEqual(policy_rows(run, iron)[-1]['step'], 20)
        self.assertEqual(policy_rows(run, logs)[0]['step'], 21)
        self.assertIsNone(found_time(policy_rows(run, logs), logs))


SPEC_RAW = {
    'version': 'replay-test', 'author': 'human', 'margin': 0.2,
    'subgoals': {
        'craft': {'ending': 'bounded', 'unit': 'step'},
        'climb_out_text': {'ending': 'bounded', 'unit': 'block'},
        'mine_iron_text': {'ending': 'until_found', 'unit': 'item', 'found': 'iron_ore',
                           'anomaly': ['no_progress', 'repeated_cycle', 'stationary_repetition'],
                           'waste': ['moving_but_stuck', 'looking_down', 'revisiting_area']},
    },
    'constraints': [],
}
SPEC = parse_spec(SPEC_RAW)
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


def load_script():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'replay_monitor_spec.py'
    spec = importlib.util.spec_from_file_location('replay_monitor_spec_under_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def craft_timeline(durations, status='completed'):
    timeline, t = [], 0.
    for i, d in enumerate(durations):
        timeline += [started(f'c{i}', 'craft', t, 10 * i, target='stick'),
                     finished(f'c{i}', 'craft', t + d, 10 * i + 10, status, target='stick')]
        t += d
    return timeline


class EvaluateAndScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec_path = self.root / 'spec.json'; self.spec_path.write_text(json.dumps(SPEC_RAW), encoding='utf-8')

    def test_evaluate_counts_problem_and_success(self):
        table = {'craft': TimeEntry(3, 10., 20.)}
        slow = steps_run(self.root, 'slow', [('craft', 30., 'time_limit', 1, 'stick')], 104729)
        done = steps_run(self.root, 'done', [('craft', 10., 'completed', 1, 'stick'),
                                             ('craft', 25., 'completed', 1, 'stick')], 104729)
        result = evaluate([slow, done], SPEC, table, (0.2, 1.0))
        tight, loose = result['margins']['0.2'], result['margins']['1']
        self.assertEqual((tight['problem']['n'], tight['problem']['detected']), (1, 1))
        self.assertAlmostEqual(tight['problem']['saved_s_median'], 6.)
        self.assertEqual((tight['success']['n'], tight['success']['false_checks']), (2, 1))
        self.assertEqual((loose['problem']['detected'], loose['success']['false_checks']), (0, 0))
        self.assertEqual(tight['by_skill']['craft']['success']['false_checks'], 1)
        self.assertEqual(len(result['steps']), 3)

    def test_unfinished_step_is_neither_problem_nor_success(self):
        rows = moving(200, hit_from=100)
        cut = load_run(write_run(self.root / 'cut', [started('iron', 'mine_iron_text', rows[0]['t'], 1, block='iron_ore'),
                                                     {'t': rows[-1]['t'], 'step': 200, 'kind': 'stop_acknowledged'}], rows))
        self.assertIsNotNone(replay(cut)['found_s'])
        summary = evaluate([cut], SPEC, AFTER_FOUND, (0.2,))['margins']['0.2']
        self.assertEqual((summary['problem']['n'], summary['success']['n'], summary['other']['n']), (0, 0, 1))

    def test_fidelity_reproduces_recorded_event(self):
        rows = stationary(402)
        timeline = [started('iron', 'mine_iron_text', rows[0]['t'], 1, block='iron_ore'),
                    {'t': 3., 'step': 60, 'kind': 'monitor_event', 'step_id': 'iron',
                     'event': {'kind': 'harvest_without_drop', 'hold': False}},
                    {'t': 20.06, 'step': 402, 'kind': 'anomaly', 'step_id': 'iron',
                     'event': {'kind': 'stationary_repetition', 'hold': True, 'tick': 401, 'step': 'iron'}},
                    finished('iron', 'mine_iron_text', 20.1, 402, 'anomaly_hold', block='iron_ore')]
        run = load_run(write_run(self.root / 'held', timeline, rows))
        fidelity = stall_fidelity(run)
        self.assertEqual((fidelity['recorded'], fidelity['reproduced']), (1, 1))

    def dev_and_eval(self, eval_seed=104729):
        dev = [write_run(self.root / 'dev' / f'A{i}', craft_timeline([10. + i, 12.]), []) for i in range(3)]
        evaluation = [write_run(self.root / 'eval' / 'E1', craft_timeline([30.], 'time_limit'), [], eval_seed)]
        return dev, evaluation

    def run_main(self, dev, evaluation, out):
        args = ['--spec', str(self.spec_path), '--fit', *map(str, dev), '--eval', *map(str, evaluation), '--out', str(out)]
        return load_script().main(args)

    def test_overlapping_runs_rejected(self):
        dev, _ = self.dev_and_eval()
        out = self.root / 'out'
        self.assertEqual(self.run_main(dev, dev[:1], out), 2)
        self.assertFalse(out.exists())

    def test_shared_seed_rejected(self):
        dev, evaluation = self.dev_and_eval(eval_seed=130363)
        out = self.root / 'out'
        self.assertEqual(self.run_main(dev, evaluation, out), 2)
        self.assertFalse(out.exists())

    def test_existing_out_dir_rejected(self):
        dev, evaluation = self.dev_and_eval()
        out = self.root / 'out'; out.mkdir()
        self.assertEqual(self.run_main(dev, evaluation, out), 2)
        self.assertEqual(list(out.iterdir()), [])

    def test_main_writes_report(self):
        dev, evaluation = self.dev_and_eval()
        (self.root / 'dev' / 'kb').mkdir()
        out = self.root / 'out'
        self.assertEqual(self.run_main(dev + [self.root / 'dev' / 'kb'], evaluation, out), 0)
        report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual([r['seed'] for r in report['inputs']['fit']], [130363] * 3)
        self.assertTrue(all(len(r['timeline_sha256']) == 64 and len(r['telemetry_sha256']) == 64
                            for r in report['inputs']['fit'] + report['inputs']['eval']))
        self.assertEqual(report['inputs']['skipped'], [str(self.root / 'dev' / 'kb')])
        self.assertEqual(sorted(report['source']), ['monitor_replay.py', 'monitor_spec.py', 'spec'])
        self.assertEqual(report['times']['craft']['worst'], 12.)
        self.assertEqual(report['evaluation']['margins']['0.2']['problem']['detected'], 1)
        self.assertIn('| 0.2 |', (out / 'report.md').read_text(encoding='utf-8'))


ENGAGED = ReplayOptions(found='engaged', lost_ticks=100)


def glimpse_rows(tail, *, glimpse_until=20, contact_from=600):
    """Walking east; the crosshair touches the ore briefly at the start, then again from `contact_from` on."""
    return [row(k, pos=(0.5 + 0.5 * k, 64., 0.5), keys=('forward',),
                hit='iron_ore' if k < glimpse_until or k >= contact_from else 'air') for k in range(1, tail, 2)]


class ReplayV2Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_default_options_are_v1(self):
        self.assertEqual(ReplayOptions(), ReplayOptions(found='first_hit', lost_ticks=100, clock='wall',
                                                        separate_goal_budget=False))

    def test_engagements_split_on_lost_contact(self):
        run = search_run(self.root, 'eval', glimpse_rows(700))
        spans = engagements(policy_rows(run, run.steps[0]), 'iron_ore', 100)
        self.assertEqual([(a['step'], b['step'], None if c is None else c['step']) for a, b, c in spans],
                         [(1, 19, 119), (601, 699, None)])

    def test_engaged_after_found_clock_restarts_after_a_glimpse(self):
        run = search_run(self.root, 'eval', glimpse_rows(700))
        step = run.steps[0]; sub = SPEC.for_step(step.skill, step.target)
        first_hit = replay_step(run, step, sub, AFTER_FOUND, .2)
        self.assertAlmostEqual(first_hit['check_s'], 6.)          # 0 s found + 5 s worst x 1.2
        engaged = replay_step(run, step, sub, AFTER_FOUND, .2, ENGAGED)
        self.assertIsNone(engaged['check_s'])                     # contact spans of 5.9 s and 4.95 s
        self.assertAlmostEqual(engaged['found_s'], 0.)

    def test_engaged_check_inside_one_long_contact(self):
        run = search_run(self.root, 'eval', moving(1200, hit_from=600))
        step = run.steps[0]
        result = replay_step(run, step, SPEC.for_step(step.skill, step.target), AFTER_FOUND, .2, ENGAGED)
        self.assertAlmostEqual(result['check_s'], 36.)
        self.assertEqual(result['check_reason'], 'over_check_time')

    def test_engaged_fit_uses_last_contact_span(self):
        runs = [search_run(self.root, f'dev{i}', glimpse_rows(tail), 'completed') for i, tail in enumerate((640, 660, 700))]
        self.assertAlmostEqual(fit_times(runs, SPEC)[0]['mine_iron_text:after_found'].best, 31.95)
        entry = fit_times(runs, SPEC, options=ENGAGED)[0]['mine_iron_text:after_found']
        self.assertAlmostEqual(entry.best, 1.95)
        self.assertAlmostEqual(entry.worst, 4.95)

    def slow_craft(self, name, status='completed'):
        """A craft step on a slow server: 4 ticks per wall second, rows for ticks 10..50."""
        rows = [row(k, t=k / 4., controller='rule') for k in range(10, 51)]
        timeline = [started('c', 'craft', 2.5, 10, target='stick'), finished('c', 'craft', 12.75, 51, status, target='stick')]
        return load_run(write_run(self.root / name, timeline, rows))

    def test_tick_clock_times_server_ticks(self):
        table = {'craft': TimeEntry(3, 2., 3.)}
        run = self.slow_craft('slow', 'time_limit'); step = run.steps[0]; sub = SPEC.for_step('craft', 'stick')
        wall = replay_step(run, step, sub, table, .2)
        self.assertAlmostEqual(wall['duration_s'], 10.25); self.assertAlmostEqual(wall['check_s'], 3.6)
        ticks = replay_step(run, step, sub, table, .2, ReplayOptions(clock='ticks'))
        self.assertAlmostEqual(ticks['duration_s'], 2.); self.assertIsNone(ticks['check_s'])
        fitted = fit_times([self.slow_craft(f'd{i}') for i in range(3)], SPEC, options=ReplayOptions(clock='ticks'))[0]
        self.assertEqual(fitted['craft'], TimeEntry(3, 2., 2.))

    def test_tick_clock_falls_back_to_wall_without_rows(self):
        run = steps_run(self.root, 'norows', [('craft', 30., 'time_limit', 1, 'stick')])
        result = replay_step(run, run.steps[0], SPEC.for_step('craft', 'stick'), {'craft': TimeEntry(3, 10., 20.)}, .2,
                             ReplayOptions(clock='ticks'))
        self.assertAlmostEqual(result['duration_s'], 30.)
        self.assertEqual(result['clock'], 'wall')

    def test_report_and_cli_record_options(self):
        spec_path = self.root / 'spec.json'; spec_path.write_text(json.dumps(SPEC_RAW), encoding='utf-8')
        dev = [write_run(self.root / 'dev' / f'A{i}', craft_timeline([10. + i, 12.]), []) for i in range(3)]
        evaluation = [write_run(self.root / 'eval' / 'E1', craft_timeline([30.], 'time_limit'), [], 104729)]
        out = self.root / 'out'
        args = ['--spec', str(spec_path), '--fit', *map(str, dev), '--eval', *map(str, evaluation), '--out', str(out),
                '--found', 'engaged', '--lost-ticks', '60', '--clock', 'ticks', '--separate-goal-budget']
        self.assertEqual(load_script().main(args), 0)
        report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['options'], {'found': 'engaged', 'lost_ticks': 60, 'clock': 'ticks',
                                             'separate_goal_budget': True})
        self.assertIn('engaged', (out / 'report.md').read_text(encoding='utf-8'))

    def test_bad_options_rejected(self):
        for kw in (dict(found='glance'), dict(clock='sun'), dict(lost_ticks=0)):
            with self.subTest(**kw), self.assertRaises(ValueError):
                ReplayOptions(**kw)


if __name__ == '__main__':
    unittest.main()
