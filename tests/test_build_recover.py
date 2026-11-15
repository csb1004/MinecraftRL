"""Building with recovery (design 2026-11-15): visible faces, standpoints, the move stage, the teacher and the feed."""
import dataclasses
import importlib.util
import math
import unittest
from pathlib import Path

from minecraftrl import build_recover as br
from minecraftrl import build_task as ct
from minecraftrl.place_task import EYE, _at

ORIGIN = (0, 64, 0)        # start block; yaw 0 faces +z and task dx is world -x


def task(cells): return dict(schema='build_task_v1', seed=7, shape='custom', cells=[list(c) for c in cells], block='cobblestone')


def floor():
    return {(x, 63, z) for x in range(-5, 6) for z in range(-2, 9)}


class VisibleFaceTest(unittest.TestCase):
    def test_the_floor_face_is_seen_until_a_block_stands_in_front(self):
        eye = (.5, 64 + EYE, .5); solids = floor()
        face = br.visible_face(eye, (0, 64, 3), solids)
        self.assertEqual(face['block'], (0, 63, 3)); self.assertEqual(face['normal'], (0, 1, 0))
        solids.add((0, 64, 2))
        self.assertIsNone(br.visible_face(eye, (0, 64, 3), solids))

    def test_a_raycast_finds_the_first_solid_and_its_face(self):
        hit, face = br.raycast((.5, 64 + EYE, .5), (0., -math.sin(math.radians(40)), math.cos(math.radians(40))), floor())
        self.assertEqual(hit[1], 63); self.assertEqual(face, 1)                 # the top face of a floor block


class TrackerTest(unittest.TestCase):
    def tracker(self, cells):
        return br.RecoverTracker(br.world_build2(task(cells), ORIGIN, 0.))

    def test_a_hidden_cell_in_reach_is_the_move_stage_with_a_standpoint_that_sees_it(self):
        tr = self.tracker([(0, 0, 3)]); tr.placed((0, 64, 2))                  # a wrong block right in front of it
        view = tr.update(.5, 64., .5, 0., 30., None, None)
        self.assertEqual(view.stage, 'move'); self.assertIsNone(view.face); self.assertIsNotNone(view.stand)
        sx, sy, sz = view.stand
        for a in (-br.ARRIVE, 0., br.ARRIVE):            # also from where a walk that arrives may stop
            for b in (-br.ARRIVE, 0., br.ARRIVE):
                self.assertIsNotNone(br.visible_face((sx + a, sy + EYE, sz + b), (0, 64, 3), tr.solids))
        self.assertNotIn((math.floor(sx), 64, math.floor(sz)), tr.solids)
        self.assertEqual(tr.update(sx, sy, sz, 0., 30., None, None).stage, 'aim')

    def test_far_is_reach_with_a_standpoint_and_a_seen_face_in_reach_is_aim(self):
        tr = self.tracker([(0, 0, 6)])
        far = tr.update(.5, 64., .5, 0., 0., None, None)
        self.assertEqual(far.stage, 'reach'); self.assertIsNotNone(far.stand); self.assertLess(far.stand[2], 6)
        self.assertEqual(far.way, far.stand)                            # nothing in the way: straight to it
        near = tr.update(.5, 64., 3.5, 0., 0., None, None)
        self.assertEqual(near.stage, 'aim'); self.assertEqual(near.face['block'], (0, 63, 6)); self.assertIsNone(near.stand)

    def test_the_crosshair_landing_in_the_cell_is_place_and_done_after_the_last_cell(self):
        tr = self.tracker([(0, 0, 3)])
        self.assertEqual(tr.update(.5, 64., .5, 0., 30., (0, 63, 3), 1).stage, 'place')
        tr.placed((0, 64, 3))
        self.assertEqual(tr.update(.5, 64., .5, 0., 30., None, None).stage, 'done')

    def test_standpoints_avoid_open_blueprint_cells_and_solids(self):
        tr = self.tracker([(0, 0, 3), (1, 0, 3), (-1, 0, 3)])
        for c in tr.cells: tr.solids.discard(c)
        tr.placed((0, 64, 2)); tr.placed((-1, 64, 2)); tr.placed((1, 64, 2))
        view = tr.update(.5, 64., .5, 0., 30., None, None)
        cell = (math.floor(view.stand[0]), 64, math.floor(view.stand[2]))
        self.assertNotIn(cell, tr.solids); self.assertNotIn(cell, set(tr.cells))


class TeacherTest(unittest.TestCase):
    def test_move_walks_to_the_standpoint_and_keeps_the_pitch_on_the_cell(self):
        tr = br.RecoverTracker(br.world_build2(task([(0, 0, 3)]), ORIGIN, 0.)); tr.placed((0, 64, 2))
        view = tr.update(.5, 64., .5, 0., 30., None, None)
        sx, _, sz = view.way
        yaw = math.degrees(math.atan2(-(sx - .5), sz - .5))           # facing the waypoint
        a = br.teacher_action(.5, 64., .5, yaw, 30., view)
        self.assertEqual((a['forward'], a['use'], a['back']), (1, 0, 0))
        self.assertLess(abs(float(a['camera'][1])), 1e-6)
        away = br.teacher_action(.5, 64., .5, yaw + 180., 30., view)
        self.assertEqual(away['forward'], 0); self.assertEqual(abs(float(away['camera'][1])), 10.)

    def test_aim_and_place_are_the_block_placing_teacher_on_the_seen_face(self):
        tr = br.RecoverTracker(br.world_build2(task([(0, 0, 3)]), ORIGIN, 0.))
        aim = tr.update(.5, 64., .5, 0., 0., None, None)
        self.assertEqual(br.teacher_action(.5, 64., .5, 0., 0., aim)['use'], 0)
        place = tr.update(.5, 64., .5, 0., 30., (0, 63, 3), 1)
        click = br.teacher_action(.5, 64., .5, 0., 30., place)
        self.assertEqual(click['use'], 1)
        # The crosshair already names the cell: no turn in the same action (a turn moved round 2 blocks one cell over).
        self.assertEqual([float(v) for v in click['camera']], [0., 0.])
        self.assertIsNone(br.teacher_action(.5, 64., .5, 0., 30., None))

    def test_features_add_the_standpoint_direction_only_when_moving(self):
        tr = br.RecoverTracker(br.world_build2(task([(0, 0, 3)]), ORIGIN, 0.))
        aim = tr.update(.5, 64., .5, 0., 0., None, None)
        f = br.build2_features(.5, 64., .5, 0., 0., aim)
        self.assertEqual(len(f), len(br.BUILD2_FEATURES)); self.assertEqual(f[-3:], [0., 0., 0.])
        self.assertTrue(any(f[4:7]))                                     # the seen face direction
        tr.placed((0, 64, 2))
        move = tr.update(.5, 64., .5, 0., 0., None, None)
        g = br.build2_features(.5, 64., .5, 0., 0., move)
        self.assertEqual(g[4:7], [0., 0., 0.]); self.assertGreater(g[-1], 0.)
        self.assertEqual(br.build2_features(.5, 64., .5, 0., 0., None), [0.] * len(br.BUILD2_FEATURES))

    def test_the_open_cell_count_is_capped_at_the_trained_range(self):
        # Size check (design 2026-11-15): heads trained on blueprints of at most 4 cells; more open cells read as 4.
        i = br.BUILD2_FEATURES.index('left')
        view = br.RecoverView('aim', (0, 64, 3), None, 3., 0, 16, None, None)
        self.assertEqual(br.build2_features(.5, 64., .5, 0., 0., view)[i], 1.)
        for left, value in ((4, 1.), (3, .75), (1, .25)):
            self.assertEqual(br.build2_features(.5, 64., .5, 0., 0., dataclasses.replace(view, left=left))[i], value)


class FeedTest(unittest.TestCase):
    def test_the_fixture_flag_picks_the_recovering_feed(self):
        t = ct.generate_build(101)
        self.assertIsInstance(br.feed_for(ct.build_fixture(t)), ct.BuildFeed)
        fixture = br.build2_fixture(t)
        self.assertEqual(fixture['build']['teacher'], 'recover')
        self.assertIsInstance(br.feed_for(fixture), br.RecoverFeed)
        self.assertEqual(br.stages_for(fixture), br.BUILD2_STAGES); self.assertEqual(br.stages_for(ct.build_fixture(t)), ct.BUILD_STAGES)


class OccludedTest(unittest.TestCase):
    """Occlusion check (design 2026-11-15): a block of the task's kind stands right in front of the first cell."""

    def test_an_occluded_task_puts_a_block_in_front_of_the_first_cell_when_that_cell_is_free(self):
        t = br.occluded_build(401)
        self.assertEqual(t['cells'], ct.generate_build(401)['cells'])
        self.assertEqual(t['clutter'], [[-2, 0, 3]])
        self.assertIsNone(br.occluded_build(403))                  # line3: the cell in front is a blueprint cell

    def test_the_block_is_solid_in_the_world_and_set_by_the_fixture(self):
        t = br.occluded_build(401)
        self.assertIn(_at(ORIGIN, 0., -2, 0, 3), br.world_build2(t, ORIGIN, 0.)['solids'])
        self.assertEqual(br.build2_fixture(t)['blocks'], [dict(dx=-2, dy=0, dz=3, block='cobblestone')])
        plain = br.build2_fixture(ct.generate_build(401))
        self.assertEqual(plain['blocks'], []); self.assertNotIn('clutter', plain['build'])

    def test_behind_the_block_the_cell_is_hidden_and_the_stage_is_move(self):
        cell = _at(ORIGIN, 0., -2, 0, 1); feet = (cell[0] + .5, 64., cell[2] + .5)
        hidden = br.RecoverTracker(br.world_build2(br.occluded_build(401), ORIGIN, 0.)).update(*feet, 0., 30., None, None)
        self.assertEqual(hidden.stage, 'move'); self.assertIsNotNone(hidden.stand)
        seen = br.RecoverTracker(br.world_build2(ct.generate_build(401), ORIGIN, 0.)).update(*feet, 0., 30., None, None)
        self.assertEqual(seen.stage, 'aim')


class CarefulTest(unittest.TestCase):
    """Round 4 (design 2026-11-15 careful place): place only when the straight crosshair ray meets the face well inside."""

    def trackers(self):
        w = br.world_build2(task([(0, 0, 3)]), ORIGIN, 0.)                # the cell is world (0, 64, 3)
        return br.RecoverTracker(w), br.CarefulTracker(w)

    @staticmethod
    def pitch_to(z):                         # from feet (.5, 64, .5), the pitch whose ray meets the floor top at z
        return math.degrees(math.atan2(EYE, z - .5))

    def test_look_follows_the_yaw_and_the_pitch(self):
        for (yaw, pitch), want in (((0., 0.), (0., 0., 1.)), ((90., 0.), (-1., 0., 0.)), ((0., 90.), (0., -1., 0.))):
            for a, b in zip(br.look(yaw, pitch), want): self.assertAlmostEqual(a, b, places=6)

    def test_a_crosshair_near_the_face_edge_is_aim_for_the_careful_teacher_only(self):
        recover, careful = self.trackers()
        edge = self.pitch_to(3.05)                                       # 0.05 inside the floor face's near edge
        self.assertEqual(recover.update(.5, 64., .5, 0., edge, (0, 63, 3), 1).stage, 'place')
        self.assertEqual(careful.update(.5, 64., .5, 0., edge, (0, 63, 3), 1).stage, 'aim')
        centre = self.pitch_to(3.5)
        self.assertEqual(careful.update(.5, 64., .5, 0., centre, (0, 63, 3), 1).stage, 'place')

    def test_the_careful_place_needs_the_straight_ray_to_land_in_the_cell(self):
        _, careful = self.trackers()
        short = self.pitch_to(2.9)                                       # the ray meets the floor of the cell before
        self.assertEqual(careful.update(.5, 64., .5, 0., short, (0, 63, 3), 1).stage, 'aim')
        self.assertFalse(br.inner_hit((.5, 64 + EYE, .5), br.look(0., short), careful.solids, (0, 64, 3)))
        self.assertTrue(br.inner_hit((.5, 64 + EYE, .5), br.look(0., self.pitch_to(3.5)), careful.solids, (0, 64, 3)))

    def test_the_fixture_names_the_careful_teacher_and_picks_its_feed(self):
        t = ct.generate_build(101)
        fixture = br.build2_fixture(t, teacher='careful')
        self.assertEqual(fixture['build']['teacher'], 'careful')
        self.assertIsInstance(br.feed_for(fixture), br.CarefulFeed); self.assertEqual(br.stages_for(fixture), br.BUILD2_STAGES)
        self.assertEqual(br.build2_fixture(t), br.build2_fixture(t, teacher='recover'))
        self.assertNotIsInstance(br.feed_for(br.build2_fixture(t)), br.CarefulFeed)
        with self.assertRaises(ValueError): br.build2_fixture(t, teacher='other')

    def test_the_careful_teacher_fills_every_blueprint_in_the_simulator(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/simulate_build_recover.py'
        spec = importlib.util.spec_from_file_location('simulate_build_recover', path)
        sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
        for seed in (101, 106, 110, 201, 401, 504):
            for yaw in (0., 90., 180., 270.):
                for clutter in (False, True):
                    with self.subTest(seed=seed, yaw=yaw, clutter=clutter):
                        r = sim.run(seed, yaw, version=3, clutter=clutter)
                        self.assertTrue(r['ok'], r); self.assertEqual(r['wrong'], 0)


class SimulationTest(unittest.TestCase):
    def sim(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/simulate_build_recover.py'
        spec = importlib.util.spec_from_file_location('simulate_build_recover', path)
        sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
        return sim

    def test_the_recovering_teacher_fills_every_blueprint_without_mistakes(self):
        sim = self.sim()
        for seed in (101, 102, 103, 201, 301):
            for yaw in (0., 90., 180., 270.):
                with self.subTest(seed=seed, yaw=yaw):
                    r = sim.run(seed, yaw, version=2)
                    self.assertTrue(r['ok'], r); self.assertEqual(r['wrong'], 0)

    def test_a_block_in_front_of_the_first_cell_stops_the_old_teacher_but_not_the_new_one(self):
        sim = self.sim()
        old = [sim.run(s, 0., version=1, clutter=True)['ok'] for s in range(101, 113)]
        new = [sim.run(s, yaw, version=2, clutter=True)['ok'] for s in range(101, 113) for yaw in (0., 90., 180., 270.)]
        self.assertTrue(all(new), new); self.assertLess(sum(old), len(old))

    def test_the_walk_goes_around_a_block_on_the_straight_line(self):
        r = self.sim().run(110, 180., version=2, clutter=True)          # its straight walk ran into the block
        self.assertTrue(r['ok'], r)


if __name__ == '__main__':
    unittest.main()
