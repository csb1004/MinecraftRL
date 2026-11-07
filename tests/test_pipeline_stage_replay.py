import hashlib
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path

from minecraftrl.pipeline.monitor_replay import load_run, policy_rows, sample_from_row
from minecraftrl.pipeline.stage_replay import (Geometry, StageTracker, build_report, destroy_events, facing_index,
                                               fixture_blocks, fixture_origin, report_markdown, step_stages)

FIXTURE = {'schema': 'iron_demo_fixture_v1', 'rotate_with_facing': True,
           'fill': [{'from': [-1, -2, 0], 'to': [1, -1, 1], 'block': 'stone'}],
           'blocks': [{'dx': 0, 'dy': 1, 'dz': 8, 'block': 'iron_ore'}, {'dx': 0, 'dy': -1, 'dz': 0, 'block': 'iron_ore'}]}


def row(step, *, tick=None, pos=(0.5, 64., 0.5), hit='air', hit_pos=None, mining='air', mining_pos=None,
        inventory=None, destroys=0, destroyed='air', destroyed_pos=None, screen=0., controller='policy', yaw=0.,
        fixture_state=2.):
    """One telemetry row in the recorded format (20 ticks per second)."""
    tick = step if tick is None else tick
    hx, hy, hz = hit_pos or (-999, -999, -999)
    mx, my, mz = mining_pos or (-999, -999, -999)
    dx, dy, dz = destroyed_pos or (-999, -999, -999)
    return {'t': tick / 20., 'step': step, 'world_time': float(tick), 'server_tick': float(tick),
            'server_x': pos[0], 'server_y': pos[1], 'server_z': pos[2], 'yaw': yaw, 'pitch': 0., 'screen': screen,
            'server_selected_item': 'air', 'server_health': 20., 'server_available': 1.,
            'hit_x': float(hx), 'hit_y': float(hy), 'hit_z': float(hz), 'controller': controller,
            'action': {'camera': [0., 0.]}, 'inventory': dict(inventory or {}),
            'demo': {'demo_available': 1., 'demo_destroys': float(destroys), 'demo_last_destroyed_block': destroyed,
                     'demo_last_destroy_x': float(dx), 'demo_last_destroy_y': float(dy), 'demo_last_destroy_z': float(dz),
                     'demo_last_destroy_harvestable': 1., 'demo_mining_block': mining,
                     'demo_mining_x': float(mx), 'demo_mining_y': float(my), 'demo_mining_z': float(mz),
                     'demo_hit_block': hit, 'demo_fixture_state': fixture_state, 'demo_fixture_origin_y': 64.}}


def timeline(skill='mine_iron_text', target='iron_ore', block='iron_ore', start=10, end=400, status='completed'):
    return [{'t': 0., 'step': 0, 'kind': 'session_started'},
            {'t': start / 20., 'step': start, 'kind': 'step_started', 'id': 'mine', 'skill': skill, 'target': target,
             'count': 1, 'block': block, 'status': 'active', 'start_step': start},
            {'t': end / 20., 'step': end, 'kind': 'step_finished', 'id': 'mine', 'skill': skill, 'target': target,
             'count': 1, 'block': block, 'status': status}]


def walk_rows():
    """far, far, closing, near, aimed, mining, collect, then nothing left (far); iron at (0, 65, 8)."""
    iron = (0, 65, 8)
    return [row(10, pos=(0.5, 64., -10.5)),          # far: 18.5 blocks away, not closing
            row(30, pos=(0.5, 64., -10.5)),          # far
            row(50, pos=(0.5, 64., -2.5)),           # closing: 8 blocks nearer than 20 ticks ago
            row(70, pos=(0.5, 64., 5.5)),            # near: within reach, crosshair elsewhere
            row(90, pos=(0.5, 64., 5.5), hit='iron_ore', hit_pos=iron),
            row(110, pos=(0.5, 64., 5.5), hit='iron_ore', hit_pos=iron, mining='iron_ore', mining_pos=iron),
            row(130, pos=(0.5, 64., 5.5), destroys=1, destroyed='iron_ore', destroyed_pos=iron),
            row(150, pos=(0.5, 64., 5.5), destroys=1, destroyed='iron_ore', destroyed_pos=iron,
                inventory={'iron_ore': 1})]


def write_run(directory, rows, tl, fixture_path, seed=130363):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    request = {'mode': 'probe', 'seed': seed, 'fixture': str(fixture_path),
               'fixture_sha256': hashlib.sha256(Path(fixture_path).read_bytes()).hexdigest()}
    (directory / 'request.json').write_text(json.dumps(request), encoding='utf-8')
    for name, items in (('timeline.jsonl', tl), ('telemetry.jsonl', rows)):
        (directory / name).write_text(''.join(json.dumps(x) + '\n' for x in items), encoding='utf-8')
    return directory


class FixtureGeometryTest(unittest.TestCase):
    def test_facing_index_follows_minecraft_yaw(self):
        self.assertEqual([facing_index(y) for y in (0., 89., 91., 180., -90., 270., 359.)], [0, 1, 1, 2, 3, 3, 0])

    def test_offsets_rotate_with_start_facing(self):
        one = {'rotate_with_facing': True, 'blocks': [{'dx': 1, 'dy': 0, 'dz': 2, 'block': 'iron_ore'}]}
        # dx is to the right of the start facing, dz forward (DemoFixture.at).
        self.assertEqual(fixture_blocks(one, (0, 64, 0), 0.), {(-1, 64, 2): 'iron_ore'})    # south: right is west
        self.assertEqual(fixture_blocks(one, (0, 64, 0), 180.), {(1, 64, -2): 'iron_ore'})  # north: right is east
        self.assertEqual(fixture_blocks(one, (0, 64, 0), 90.), {(-2, 64, -1): 'iron_ore'})  # west: right is north
        fixed = dict(one, rotate_with_facing=False)
        self.assertEqual(fixture_blocks(fixed, (0, 64, 0), 180.), {(-1, 64, 2): 'iron_ore'})

    def test_single_blocks_override_fill(self):
        blocks = fixture_blocks(FIXTURE, (0, 64, 0), 0.)
        self.assertEqual(blocks[(0, 63, 0)], 'iron_ore')
        self.assertEqual(sum(1 for b in blocks.values() if b == 'stone'), 3 * 2 * 2 - 1)

    def test_origin_from_first_applied_fixture_row(self):
        rows = [row(1, pos=(9.7, 70., 3.2), fixture_state=1.), row(2, pos=(-4.5, 64., 7.5), yaw=180.),
                row(3, pos=(0.5, 64., 0.5))]
        self.assertEqual(fixture_origin(rows), ((-5, 64, 7), 180.))
        self.assertIsNone(fixture_origin([row(1, fixture_state=1.)]))

    def test_origin_falls_back_to_feet_block_on_older_records(self):
        old = row(2, pos=(-4.5, 70.4, 7.5)); del old['demo']['demo_fixture_origin_y']
        self.assertEqual(fixture_origin([old]), ((-5, 70, 7), 0.))

    def test_nearest_distance_skips_removed_targets(self):
        geometry = Geometry(fixture_blocks(FIXTURE, (0, 64, 0), 0.))
        eye = (0.5, 64. + 1.62, 0.5)
        self.assertAlmostEqual(geometry.nearest('iron_ore', eye), 1.62 + .5)   # centre of the block under the feet
        geometry.remove((0, 63, 0))
        self.assertAlmostEqual(geometry.nearest('iron_ore', eye), ((65.5 - 65.62) ** 2 + 8 ** 2) ** .5)
        self.assertIsNone(Geometry({}).nearest('iron_ore', eye))

    def test_log_family_matches_any_log_kind(self):
        geometry = Geometry({(3, 64, 0): 'birch_log', (9, 64, 0): 'oak_log'})
        self.assertAlmostEqual(geometry.nearest('log', (0.5, 64.5, 0.5)), 3.)
        self.assertTrue(geometry.knows((9, 64, 0), 'oak_log'))
        self.assertFalse(geometry.knows((9, 64, 0), 'iron_ore'))


class StepStagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixture = self.root / 'fixture.json'
        self.fixture.write_text(json.dumps({'rotate_with_facing': True,
                                            'blocks': [{'dx': 0, 'dy': 1, 'dz': 8, 'block': 'iron_ore'}]}),
                                encoding='utf-8')

    def stages(self, rows, **kw):
        run = load_run(write_run(self.root / 'R1', [row(1, controller='prepare')] + rows, timeline(**kw), self.fixture))
        geometry = Geometry(fixture_blocks(json.loads(self.fixture.read_text()), (0, 64, 0), 0.))
        return step_stages(run, run.steps[0], geometry)

    def tracked(self, rows):
        run = load_run(write_run(self.root / 'R1', [row(1, controller='prepare')] + rows, timeline(), self.fixture))
        step = run.steps[0]
        tracker = StageTracker(Geometry(fixture_blocks(json.loads(self.fixture.read_text()), (0, 64, 0), 0.)),
                               'iron_ore', 'iron_ore', step.start_t)
        pending, out = deque(destroy_events(run.rows)), []
        for r in policy_rows(run, step):
            s = sample_from_row(r)
            due = []
            while pending and pending[0][0] <= r['t']: due.append(pending.popleft())
            tracker.apply_destroys(due, s)
            out.append(tracker.update(s).stage)
        return out

    def test_tracker_stages_match_step_stages_row_by_row(self):
        self.assertEqual(self.tracked(walk_rows()),
                         ['far', 'far', 'closing', 'near', 'aimed', 'mining', 'collect', 'far'])

    def test_peek_does_not_change_closing_history(self):
        rows = [sample_from_row(r) for r in walk_rows()]
        tracker = StageTracker(Geometry(fixture_blocks(json.loads(self.fixture.read_text()), (0, 64, 0), 0.)),
                               'iron_ore', 'iron_ore', 0.)
        tracker.update(rows[0]); tracker.update(rows[1])
        before = list(tracker.recent)                        # (tick, distance) history for the closing test
        self.assertEqual(tracker.peek(rows[2]).stage, 'closing')
        self.assertEqual(list(tracker.recent), before)
        self.assertEqual(tracker.update(rows[2]).stage, 'closing')

    def test_rows_split_into_stages_by_ticks(self):
        out = self.stages(walk_rows())
        self.assertEqual(out['seconds'], {'far': 2., 'closing': 1., 'near': 1., 'aimed': 1., 'mining': 1., 'collect': 1.,
                                          'gui': 0.})
        self.assertEqual(out['total_s'], 7.)
        self.assertEqual((out['geometry_checked'], out['geometry_mismatch']), (2, 0))
        self.assertEqual(out['target'], 'iron_ore')

    def test_crosshair_while_near_and_broken_target_heights(self):
        fixture = {'blocks': [{'dx': 0, 'dy': 1, 'dz': 3, 'block': 'iron_ore'}, {'dx': 0, 'dy': -2, 'dz': 1, 'block': 'iron_ore'}]}
        self.fixture.write_text(json.dumps(fixture), encoding='utf-8')
        rows = [row(10, hit='stone', hit_pos=(0, 65, 2)), row(30, hit='stone', hit_pos=(0, 65, 2)), row(50),
                row(70, destroys=1, destroyed='iron_ore', destroyed_pos=(0, 65, 3)),
                row(90, destroys=2, destroyed='iron_ore', destroyed_pos=(0, 62, 1)), row(110)]
        out = self.stages(rows)
        self.assertEqual(out['near_hits'], {'stone': 2., 'air': 1.})
        self.assertEqual(out['target_broken'], {'level_or_above': 1, 'below': 1})

    def test_view_angle_to_the_target_while_near(self):
        fixture = {'blocks': [{'dx': 0, 'dy': 1, 'dz': 3, 'block': 'iron_ore'}]}
        self.fixture.write_text(json.dumps(fixture), encoding='utf-8')
        # The ore is 3 blocks south of the eye: yaw 0 faces it, 60 turns away by ~60 degrees, 180 faces north.
        rows = [row(10, yaw=0.), row(30, yaw=180.), row(50, yaw=60.), row(70, yaw=0.)]
        out = self.stages(rows)
        self.assertEqual(out['near_angle_s'], {'<15': 1., '15-45': 0., '45-90': 1., '>=90': 1.})

    def test_gui_rows_and_pit_time(self):
        rows = [row(10, pos=(0.5, 62., 0.5), screen=1.), row(30, pos=(0.5, 62., 0.5)), row(50, pos=(0.5, 64., 0.5))]
        out = self.stages(rows)
        self.assertEqual(out['seconds']['gui'], 1.)
        self.assertEqual(out['pit_s'], 2.)

    def test_hit_on_unknown_position_counts_as_mismatch(self):
        out = self.stages([row(10, hit='iron_ore', hit_pos=(5, 64, 5)), row(30)])
        self.assertEqual((out['geometry_checked'], out['geometry_mismatch']), (1, 1))

    def test_hits_on_placed_target_blocks_join_the_geometry(self):
        rows = [row(10, hit='iron_ore', hit_pos=(0, 64, 2)), row(30), row(50)]   # the fixture's iron is 8 blocks off
        out = self.stages(rows)
        self.assertEqual((out['geometry_checked'], out['geometry_mismatch']), (1, 1))
        self.assertEqual((out['seconds']['aimed'], out['seconds']['near'], out['seconds']['far']), (1., 1., 0.))

    def test_pit_uses_geometry_origin_on_older_records(self):
        old = row(10, pos=(0.5, 62., 0.5)); del old['demo']['demo_fixture_origin_y']
        run = load_run(write_run(self.root / 'R2', [old, row(30)], timeline(), self.fixture))
        geometry = Geometry(fixture_blocks(json.loads(self.fixture.read_text()), (0, 64, 0), 0.), origin_y=64)
        self.assertEqual(step_stages(run, run.steps[0], geometry)['pit_s'], 1.)

    def test_tool_at_start_and_long_gaps(self):
        rows = [row(10, inventory={'stone_pickaxe': 1}), row(400, inventory={'stone_pickaxe': 1}), row(410)]
        out = self.stages(rows, end=500)
        self.assertEqual(out['tool_at_start'], ['stone_pickaxe'])
        self.assertEqual(out['total_s'], 1. + .5)   # a gap over 20 ticks counts as 20 ticks


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixture = self.root / 'fixture.json'
        self.fixture.write_text(json.dumps({'rotate_with_facing': True,
                                            'blocks': [{'dx': 0, 'dy': 1, 'dz': 8, 'block': 'iron_ore'}]}),
                                encoding='utf-8')

    def test_report_groups_by_skill_tool_and_outcome(self):
        prep = row(1, pos=(0.5, 64., 0.5), controller='prepare')
        a = write_run(self.root / 'A', [prep, row(10), row(30)], timeline(status='time_limit'), self.fixture)
        b = write_run(self.root / 'B', [prep, row(10, inventory={'stone_pickaxe': 1}), row(30), row(50)], timeline(),
                      self.fixture, seed=104729)
        report = build_report([a, b], root=self.root)
        groups = {(g['skill'], g['tool'], g['outcome']): g for g in report['groups']}
        self.assertEqual(groups[('mine_iron_text', 'no_tool', 'time_limit')]['n'], 1)
        self.assertEqual(groups[('mine_iron_text', 'stone_pickaxe', 'completed')]['seconds']['far'], 2.)
        self.assertEqual(groups[('mine_iron_text', 'stone_pickaxe', 'completed')]['near_angle_s'],
                         {'<15': 0., '15-45': 0., '45-90': 0., '>=90': 0.})
        self.assertEqual(report['skipped'], [])
        self.assertIn('| mine_iron_text |', report_markdown(report))

    def test_changed_fixture_file_skips_the_run(self):
        prep = row(1, controller='prepare')
        a = write_run(self.root / 'A', [prep, row(10), row(30)], timeline(), self.fixture)
        self.fixture.write_text('{"blocks": []}', encoding='utf-8')
        report = build_report([a], root=self.root)
        self.assertEqual(report['steps'], [])
        self.assertEqual(report['skipped'][0]['reason'], 'fixture sha256 differs from the record')


def load_script():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'replay_skill_stages.py'
    spec = importlib.util.spec_from_file_location('replay_skill_stages_under_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class ScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture = self.root / 'fixture.json'
        fixture.write_text(json.dumps({'blocks': [{'dx': 0, 'dy': 1, 'dz': 8, 'block': 'iron_ore'}]}), encoding='utf-8')
        self.run_dir = write_run(self.root / 'A', [row(1, controller='prepare'), row(10), row(30)], timeline(), fixture)

    def test_writes_report_and_refuses_existing_output(self):
        out = self.root / 'out'
        self.assertEqual(load_script().main(['--runs', str(self.run_dir), '--out', str(out), '--root', str(self.root)]), 0)
        report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(len(report['steps']), 1)
        self.assertTrue((out / 'report.md').exists())
        self.assertEqual(load_script().main(['--runs', str(self.run_dir), '--out', str(out), '--root', str(self.root)]), 2)

    def test_folders_without_records_are_listed_not_fatal(self):
        out = self.root / 'out2'; (self.root / 'kb').mkdir()
        self.assertEqual(load_script().main(['--runs', str(self.run_dir), str(self.root / 'kb'), '--out', str(out),
                                             '--root', str(self.root)]), 0)
        report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['skipped'], [{'dir': str(self.root / 'kb'), 'reason': 'no records'}])


if __name__ == '__main__':
    unittest.main()
