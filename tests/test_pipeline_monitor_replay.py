import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.monitor_replay import load_run, policy_rows, sample_from_row
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


if __name__ == '__main__':
    unittest.main()
