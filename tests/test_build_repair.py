"""Building round 5, removing stray blocks (design docs/specs/2026-11-15-build-repair-design.md)."""
import dataclasses
import importlib.util
import unittest
from pathlib import Path

from minecraftrl import build_recover as br
from minecraftrl import build_repair as rp
from minecraftrl import build_task as ct
from minecraftrl.place_task import ARENA, EYE, _at, _centre
from minecraftrl.reacquire_teacher import point_delta

ORIGIN = (0, 64, 0)        # start block; yaw 0 faces +z and task dx is world -x
EYES = (.5, 64 + EYE, .5)


def task(cells, strays=(), clutter=()):
    t = dict(schema='build_task_v1', seed=7, shape='custom', cells=[list(c) for c in cells], block='cobblestone',
             strays=[list(s) for s in strays])
    if clutter: t['clutter'] = [list(c) for c in clutter]
    return t


def tracker(cells, strays=(), clutter=(), held='cobblestone'):
    tr = rp.RepairTracker(rp.world_repair(task(cells, strays, clutter), ORIGIN, 0.)); tr.held = held
    return tr


def aim_at(point):
    dpitch, dyaw = point_delta(EYES, 0., 0., point)
    return dyaw, dpitch                                   # (yaw, pitch) from yaw 0, pitch 0


class RepairBuildTest(unittest.TestCase):
    def test_the_stray_count_is_the_id_mod_3_next_to_the_blueprint_floor_cells(self):
        for k in range(1101, 1401):
            with self.subTest(task=k):
                t = rp.repair_build(k); plain = ct.generate_build(k)
                self.assertEqual(t['cells'], plain['cells']); self.assertEqual(t['shape'], plain['shape'])
                strays = [tuple(s) for s in t['strays']]
                self.assertEqual(len(strays), k % 3); self.assertEqual(len(set(strays)), len(strays))
                ground = {(x, z) for x, y, z in t['cells'] if y == 0}
                for x, y, z in strays:
                    self.assertEqual(y, 0)
                    self.assertNotIn([x, y, z], t['cells']); self.assertNotEqual((x, z), (0, 0))
                    self.assertTrue(ARENA['x'][0] <= x <= ARENA['x'][1] and ARENA['z'][0] <= z <= ARENA['z'][1])
                    self.assertTrue(any(abs(x - gx) + abs(z - gz) == 1 for gx, gz in ground))
        self.assertEqual(rp.repair_build(1104), rp.repair_build(1104))

    def test_the_fixture_sets_the_strays_and_gives_the_block_then_a_stone_pickaxe(self):
        t = rp.repair_build(1103)                             # 1103 % 3 = 2 strays
        f = rp.repair_fixture(t)
        self.assertEqual(f['build']['teacher'], 'repair')
        self.assertEqual(f['blocks'], [dict(dx=s[0], dy=s[1], dz=s[2], block='cobblestone') for s in t['strays']])
        self.assertEqual(f['give'], [dict(item='cobblestone', count=32), dict(item='stone_pickaxe', count=1)])
        self.assertIn('2 stray blocks', f['description'])
        self.assertIsInstance(br.feed_for(f), rp.RepairFeed); self.assertEqual(br.stages_for(f), rp.REPAIR_STAGES)
        self.assertTrue(br.recovering(f))

    def test_the_careful_fixture_of_a_repair_task_keeps_the_strays_as_fixed_blocks(self):
        t = rp.repair_build(1103)
        f = rp.repair_fixture(t, teacher='careful')
        self.assertEqual(f['build']['teacher'], 'careful'); self.assertEqual(f['build']['clutter'], t['strays'])
        self.assertEqual(f['blocks'], rp.repair_fixture(t)['blocks']); self.assertEqual(f['give'], rp.repair_fixture(t)['give'])
        self.assertIsInstance(br.feed_for(f), br.CarefulFeed); self.assertNotIsInstance(br.feed_for(f), rp.RepairFeed)
        self.assertNotIn('Occlusion check', f['description'])
        with self.assertRaises(ValueError): rp.repair_fixture(t, teacher='recover')


class TrackerTest(unittest.TestCase):
    def test_a_stray_in_reach_is_dug_first_through_its_nearest_seen_face(self):
        tr = tracker([(0, 0, 3)], strays=[(1, 0, 3)])
        stray = _at(ORIGIN, 0., 1, 0, 3)
        self.assertIn(stray, tr.solids); self.assertEqual(tr.strays, {stray})
        v = tr.update(.5, 64., .5, 0., 20., None, None)
        self.assertEqual((v.stage, v.target), ('dig', stray))
        self.assertEqual((v.face['block'], v.face['normal']), (stray, (0, 0, -1)))
        self.assertFalse(v.pick); self.assertTrue(v.block); self.assertFalse(v.strike)
        self.assertFalse(tr.done)
        tr.held = None                                       # a record without the selected item: the block in hand
        v = tr.update(.5, 64., .5, 0., 20., None, None); self.assertFalse(v.pick); self.assertTrue(v.block)
        tr.held = 'air'
        self.assertFalse(tr.update(.5, 64., .5, 0., 20., None, None).block)

    def test_a_far_stray_is_reached_from_a_standpoint_that_sees_one_of_its_faces(self):
        tr = tracker([(-2, 0, 3)], strays=[(0, 0, 7)])
        v = tr.update(.5, 64., .5, 0., 0., None, None)
        self.assertEqual((v.stage, v.target), ('reach', _at(ORIGIN, 0., 0, 0, 7)))
        self.assertIsNotNone(v.stand)
        eye = (v.stand[0], v.stand[1] + EYE, v.stand[2])
        self.assertIsNotNone(rp.visible_block_face(eye, v.target, tr.solids))

    def test_the_crosshair_well_inside_the_stray_face_with_the_pickaxe_is_a_strike(self):
        tr = tracker([(0, 0, 3)], strays=[(1, 0, 3)], held='stone_pickaxe')
        stray = _at(ORIGIN, 0., 1, 0, 3)
        yaw, pitch = aim_at((stray[0] + .5, stray[1] + .5, stray[2]))          # north face centre
        v = tr.update(.5, 64., .5, yaw, pitch, stray, 2)
        self.assertEqual(v.stage, 'dig'); self.assertTrue(v.pick); self.assertFalse(v.block); self.assertTrue(v.strike)
        edge = aim_at((stray[0] + .97, stray[1] + .5, stray[2]))               # near the face's edge
        fresh = tracker([(0, 0, 3)], strays=[(1, 0, 3)], held='stone_pickaxe')
        self.assertFalse(fresh.update(.5, 64., .5, *edge, stray, 2).strike)     # a strike does not start at the rim
        self.assertTrue(tr.update(.5, 64., .5, *edge, stray, 2).strike)         # but a started one holds there
        off = aim_at((stray[0] + 1.3, stray[1] + .5, stray[2]))                 # past the edge: off the stray
        self.assertFalse(tr.update(.5, 64., .5, *off, None, None).strike)
        self.assertFalse(tr.update(.5, 64., .5, *edge, stray, 2).strike)        # off once: the hold is gone
        bare = tracker([(0, 0, 3)], strays=[(1, 0, 3)])                         # no pickaxe: switch, do not attack
        a = rp.teacher_action(.5, 64., .5, yaw, pitch, bare.update(.5, 64., .5, yaw, pitch, stray, 2))
        self.assertEqual((a['attack'], a['hotbar.2']), (0, 1))

    def test_a_wrong_placement_becomes_a_stray_and_removing_it_returns_to_building(self):
        tr = tracker([(0, 0, 3)])
        wrong = _at(ORIGIN, 0., 1, 0, 3)
        self.assertEqual(tr.update(.5, 64., .5, 0., 20., None, None).stage, 'aim')
        tr.placed(wrong)
        self.assertEqual((tr.wrong, tr.strays), (1, {wrong}))
        self.assertEqual(tr.update(.5, 64., .5, 0., 20., None, None).stage, 'dig')
        tr.destroyed(wrong)
        self.assertEqual((tr.strays, tr.dug, tr.wrong), (set(), 1, 1)); self.assertNotIn(wrong, tr.solids)
        self.assertEqual(tr.update(.5, 64., .5, 0., 20., None, None).stage, 'aim')

    def test_breaking_a_blueprint_block_opens_its_cell_again(self):
        tr = tracker([(0, 0, 3)])
        cell = _at(ORIGIN, 0., 0, 0, 3)
        tr.placed(cell); self.assertTrue(tr.done)
        self.assertEqual(tr.update(.5, 64., .5, 0., 20., None, None).stage, 'done')
        tr.destroyed(cell)
        self.assertEqual((tr.broken, tr.filled_cells), (1, set())); self.assertFalse(tr.done)

    def test_done_needs_the_blueprint_filled_and_no_stray_left(self):
        tr = tracker([(0, 0, 3)], strays=[(1, 0, 3)])
        tr.placed(_at(ORIGIN, 0., 0, 0, 3))
        self.assertFalse(tr.done); self.assertEqual(tr.update(.5, 64., .5, 0., 20., None, None).stage, 'dig')
        tr.destroyed(_at(ORIGIN, 0., 1, 0, 3))
        self.assertTrue(tr.done); self.assertEqual(tr.update(.5, 64., .5, 0., 20., None, None).stage, 'done')


class TeacherTest(unittest.TestCase):
    def test_dig_switches_to_the_pickaxe_before_attacking(self):
        tr = tracker([(0, 0, 3)], strays=[(1, 0, 3)])
        v = tr.update(.5, 64., .5, 0., 20., None, None)
        a = rp.teacher_action(.5, 64., .5, 0., 20., v)
        self.assertEqual((a['hotbar.2'], a['attack'], a['use'], a['hotbar.1']), (1, 0, 0, 0))

    def test_dig_attacks_without_turning_on_a_strike_and_turns_to_the_face_otherwise(self):
        tr = tracker([(0, 0, 3)], strays=[(1, 0, 3)], held='stone_pickaxe')
        stray = _at(ORIGIN, 0., 1, 0, 3)
        yaw, pitch = aim_at((stray[0] + .5, stray[1] + .5, stray[2]))
        a = rp.teacher_action(.5, 64., .5, yaw, pitch, tr.update(.5, 64., .5, yaw, pitch, stray, 2))
        self.assertEqual((a['attack'], a['hotbar.2'], list(a['camera'])), (1, 0, [0., 0.]))
        a = rp.teacher_action(.5, 64., .5, 0., 20., tr.update(.5, 64., .5, 0., 20., None, None))
        self.assertEqual(a['attack'], 0); self.assertGreater(abs(a['camera'][1]), 0.)

    def test_building_with_the_pickaxe_in_hand_switches_back_and_does_not_click(self):
        tr = tracker([(0, 0, 3)], held='stone_pickaxe')
        cell = _at(ORIGIN, 0., 0, 0, 3)
        yaw, pitch = aim_at((cell[0] + .5, cell[1], cell[2] + .5))            # the floor face under the cell
        v = tr.update(.5, 64., .5, yaw, pitch, (cell[0], 63, cell[2]), 1)
        self.assertEqual(v.stage, 'place')
        a = rp.teacher_action(.5, 64., .5, yaw, pitch, v)
        self.assertEqual((a['hotbar.1'], a['use']), (1, 0))
        tr.held = 'cobblestone'
        a = rp.teacher_action(.5, 64., .5, yaw, pitch, tr.update(.5, 64., .5, yaw, pitch, (cell[0], 63, cell[2]), 1))
        self.assertEqual((a['hotbar.1'], a['use']), (0, 1))

    def test_done_has_no_label(self):
        tr = tracker([(0, 0, 3)]); tr.placed(_at(ORIGIN, 0., 0, 0, 3))
        self.assertIsNone(rp.teacher_action(.5, 64., .5, 0., 20., tr.update(.5, 64., .5, 0., 20., None, None)))


class FeatureTest(unittest.TestCase):
    def test_the_round_4_features_plus_the_pickaxe_in_hand(self):
        self.assertEqual(rp.REPAIR_FEATURES, br.BUILD2_FEATURES + ('held',))
        self.assertEqual(rp.REPAIR_STAGES, ('reach', 'move', 'aim', 'place', 'dig', 'done'))
        tr = tracker([(0, 0, 3)], strays=[(1, 0, 3)])
        v = tr.update(.5, 64., .5, 0., 20., None, None)
        f = rp.repair_features(.5, 64., .5, 0., 20., v)
        self.assertEqual(len(f), len(rp.REPAIR_FEATURES)); self.assertEqual(f[-1], 0.)
        self.assertEqual(f[:-1], br.build2_features(.5, 64., .5, 0., 20., dataclasses.replace(v, stage='aim')))
        tp, ty = point_delta(EYES, 0., 20., _centre(_at(ORIGIN, 0., 1, 0, 3)))
        self.assertAlmostEqual(f[2], max(-1., min(1., tp / 60.)))
        tr.held = 'stone_pickaxe'
        self.assertEqual(rp.repair_features(.5, 64., .5, 0., 20., tr.update(.5, 64., .5, 0., 20., None, None))[-1], 1.)
        self.assertEqual(rp.repair_features(.5, 64., .5, 0., 20., None), [0.] * len(rp.REPAIR_FEATURES))


def row(step, *, place=0, last=None, destroys=0, destroyed=None, held='cobblestone', yaw=0., pitch=20., hit=None, face=-999):
    d = dict(demo_available=1., demo_fixture_state=2., demo_fixture_origin_y=64., demo_hit_block='air' if hit is None else 'cobblestone',
             demo_destroys=float(destroys), demo_last_destroy_removed=1. if destroyed is not None else 0.)
    for a, v in zip('xyz', destroyed or (-999, -999, -999)): d['demo_last_destroy_' + a] = float(v)
    p = dict(place_available=1., place_count=float(place))
    for a, v in zip('xyz', last or (-999, -999, -999)): p['place_last_' + a] = float(v)
    r = dict(step=step, server_available=1., server_x=.5, server_y=64., server_z=.5, client_x=.5, client_y=64., client_z=.5,
             yaw=yaw, pitch=pitch, server_selected_item=held, demo=d, place=p, hit_face=float(face))
    for a, v in zip('xyz', hit or (-999, -999, -999)): r['hit_' + a] = float(v)
    return r


class FeedTest(unittest.TestCase):
    def test_the_feed_follows_destroys_the_hand_and_placements_to_done(self):
        t = task([(0, 0, 3)], strays=[(1, 0, 3)])
        feed = rp.RepairFeed(rp.repair_fixture(t))
        stray, cell = _at(ORIGIN, 0., 1, 0, 3), _at(ORIGIN, 0., 0, 0, 3)
        feed.observe(row(1))
        self.assertEqual(feed.decision_view().stage, 'dig'); self.assertFalse(feed.done())
        self.assertEqual(feed.features()[-1], 0.)
        feed.observe(row(2, held='stone_pickaxe'))
        self.assertEqual(feed.features()[-1], 1.)
        feed.observe(row(3, held='stone_pickaxe', destroys=1, destroyed=stray))
        self.assertEqual(feed.decision_view().stage, 'aim')
        self.assertEqual(feed.repair_extra(), dict(strays_left=0, dug=1, broken=0, unknown_destroys=0))
        feed.observe(row(4, place=1, last=cell, destroys=1, destroyed=stray))
        self.assertTrue(feed.done()); self.assertEqual((feed.filled(), feed.wrong()), (1, 0))

    def test_a_missed_destroy_is_counted_and_the_first_count_is_the_baseline(self):
        feed = rp.RepairFeed(rp.repair_fixture(task([(0, 0, 3)], strays=[(1, 0, 3), (-1, 0, 3)])))
        feed.observe(row(1, destroys=4))
        feed.observe(row(2, destroys=6, destroyed=_at(ORIGIN, 0., 1, 0, 3)))
        self.assertEqual(feed.repair_extra(), dict(strays_left=1, dug=1, broken=0, unknown_destroys=1))

    def test_the_label_uses_the_teacher_and_the_row_pose(self):
        feed = rp.RepairFeed(rp.repair_fixture(task([(0, 0, 3)], strays=[(1, 0, 3)])))
        feed.observe(row(1)); seen = feed.last_row
        self.assertEqual(feed.label(seen, feed.decision_view())['hotbar.2'], 1)

    def test_the_outcome_replays_any_run_with_the_repair_rules(self):
        t = task([(0, 0, 3)], strays=[(1, 0, 3)])
        stray, cell = _at(ORIGIN, 0., 1, 0, 3), _at(ORIGIN, 0., 0, 0, 3)
        left = rp.repair_outcome(t, [row(1), row(2, place=1, last=cell)])
        self.assertEqual((left['done'], left['strays_left'], left['filled'], left['wrong']), (False, 1, 1, 0))
        fixed = rp.repair_outcome(t, [row(1), row(2, place=1, last=cell), row(3, place=1, last=cell, destroys=1, destroyed=stray)])
        self.assertEqual((fixed['done'], fixed['strays_left'], fixed['dug']), (True, 0, 1))


class SimulationTest(unittest.TestCase):
    def sim(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/simulate_build_recover.py'
        spec = importlib.util.spec_from_file_location('simulate_build_recover', path)
        sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
        return sim

    def test_the_repair_teacher_removes_every_stray_and_fills_every_blueprint(self):
        sim = self.sim()
        for k in range(1101, 1113):
            for yaw in (0., 90., 180., 270.):
                with self.subTest(task=k, yaw=yaw):
                    r = sim.run(k, yaw, version=4, make=rp.repair_build)
                    self.assertTrue(r['ok'], r); self.assertEqual((r['wrong'], r['strays_left']), (0, 0))
                    self.assertEqual(r['dug'], k % 3)

    def test_the_repair_teacher_also_removes_its_own_stray_presses(self):
        sim = self.sim(); own = 0
        for k in range(1101, 1113):
            for yaw in (0., 180.):
                with self.subTest(task=k, yaw=yaw):
                    r = sim.run(k, yaw, version=4, make=rp.repair_build, stray=.1, noise=3., rng_seed=1)
                    self.assertTrue(r['ok'], r); self.assertEqual(r['strays_left'], 0)
                    own += r['wrong']
        self.assertGreater(own, 0)                    # some stray presses did land outside the blueprint


if __name__ == '__main__':
    unittest.main()
