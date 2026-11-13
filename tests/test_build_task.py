"""Building skill (design 2026-11-14): blueprints, fixture, world geometry, stages, teacher, features and the feed."""
import importlib.util
import json
import unittest
from pathlib import Path

from minecraftrl import build_task as ct
from minecraftrl import place_task as pt


class TaskTest(unittest.TestCase):
    def test_same_seed_same_blueprint_and_rules(self):
        self.assertEqual(ct.generate_build(7), ct.generate_build(7))
        self.assertEqual(json.loads(json.dumps(ct.generate_build(7))), ct.generate_build(7))
        shapes = set()
        for seed in range(300):
            t = ct.generate_build(seed); shapes.add(t['shape'])
            cells = [tuple(c) for c in t['cells']]
            self.assertEqual(len(cells), len(set(cells)))
            self.assertTrue(all(-2 <= c[0] <= 2 and 0 <= c[1] <= 1 and 3 <= c[2] <= 6 for c in cells))
            for i, c in enumerate(cells):                        # every cell rests on the floor or an earlier cell
                self.assertTrue(c[1] == 0 or (c[0], c[1] - 1, c[2]) in cells[:i], (seed, c))
        self.assertEqual(shapes, set(ct.SHAPES))

    def test_fixture_is_the_place_arena_with_blocks_to_place(self):
        task = ct.generate_build(3); f = ct.build_fixture(task)
        self.assertEqual((f['schema'], f['rotate_with_facing'], f['build'], f['blocks']), ('iron_demo_fixture_v1', True, task, []))
        self.assertEqual(f['give'], [dict(item='cobblestone', count=pt.GIVE_COUNT)])
        self.assertEqual(f['fill'], pt.place_fixture(pt.generate_target(3))['fill'])


class TrackerTest(unittest.TestCase):
    task = dict(schema=ct.SCHEMA, seed=0, shape='pillar2', cells=[[0, 0, 3], [0, 1, 3]], block='cobblestone')

    def setUp(self):
        self.w = ct.world_build(self.task, (100, 64, 200), 0.)                 # south: +z forward
        self.t = ct.BuildTracker(self.w)

    def test_cells_in_order_and_stages_per_cell(self):
        self.assertEqual(self.w['cells'], [(100, 64, 203), (100, 65, 203)])
        v = self.t.update(100.5, 64., 200.5, 0., 40., (100, 63, 203), 1)       # the floor top under the first cell
        self.assertEqual((v.stage, v.target, v.left), ('place', (100, 64, 203), 2))
        self.t.placed((100, 64, 203))
        v = self.t.update(100.5, 64., 200.5, 0., 40., (100, 63, 203), 1)       # that face now lands in a filled cell
        self.assertEqual((v.stage, v.target, v.filled), ('aim', (100, 65, 203), 1))
        self.assertEqual(self.t.update(100.5, 64., 200.5, 0., 30., (100, 64, 203), 1).stage, 'place')   # the first block's top

    def test_far_cells_are_reached_first_and_wrong_ones_counted(self):
        self.assertEqual(self.t.update(100.5, 64., 196.5, 0., 0., None, None).stage, 'reach')
        self.t.placed((101, 64, 203)); self.assertEqual(self.t.wrong, 1)
        self.t.placed((100, 64, 203)); self.t.placed((100, 65, 203))
        self.assertTrue(self.t.done)
        self.assertEqual(self.t.update(100.5, 64., 200.5, 0., 0., None, None).stage, 'done')


class TeacherFeatureTest(unittest.TestCase):
    def test_the_place_teacher_and_features_on_the_current_cell(self):
        w = ct.world_build(TrackerTest.task, (100, 64, 200), 0.)
        t = ct.BuildTracker(w)
        v = t.update(100.5, 64., 200.5, 0., 40., (100, 63, 203), 1)
        a = ct.teacher_action(100.5, 64., 200.5, 0., 40., v)
        same = pt.teacher_action(100.5, 64., 200.5, 0., 40., v.place_view())
        self.assertEqual({k: x for k, x in a.items() if k != 'camera'}, {k: x for k, x in same.items() if k != 'camera'})
        self.assertEqual(list(a['camera']), list(same['camera']))
        self.assertEqual(a['use'], 1)
        f = ct.build_features(100.5, 64., 200.5, 0., 40., v)
        self.assertEqual(ct.BUILD_FEATURES, pt.PLACE_FEATURES + ('left',))
        self.assertEqual(f, pt.place_features(100.5, 64., 200.5, 0., 40., v.place_view()) + [2 / 4.])
        for c in w['cells']: t.placed(c)
        v = t.update(100.5, 64., 200.5, 0., 40., None, None)
        self.assertIsNone(ct.teacher_action(100.5, 64., 200.5, 0., 40., v))
        self.assertEqual(ct.build_features(100.5, 64., 200.5, 0., 40., v), [0.] * 8)


class FeedTest(unittest.TestCase):
    def rows(self):
        demo = dict(demo_available=1, demo_fixture_state=2, demo_fixture_origin_y=64, demo_hit_block='grass_block')
        base = dict(t=1., controller='policy', server_x=100.5, server_y=64., server_z=200.5, server_on_ground=1, yaw=0.,
                    pitch=40., client_x=100.5, client_y=64., client_z=200.5, client_on_ground=1, server_available=1,
                    hit_x=100, hit_y=63, hit_z=203, hit_face=1, demo=demo,
                    place=dict(place_available=1, place_count=0, place_last_x=-999, place_last_y=-999, place_last_z=-999))
        one = dict(base, place=dict(base['place'], place_count=1, place_last_x=100, place_last_y=64, place_last_z=203))
        two = dict(base, place=dict(base['place'], place_count=2, place_last_x=100, place_last_y=65, place_last_z=203))
        return base, one, two

    def test_cells_fill_from_the_server_record(self):
        feed = ct.BuildFeed(ct.build_fixture(TrackerTest.task))
        base, one, two = self.rows()
        feed.observe(base); self.assertEqual(feed.decision_view().stage, 'place')
        self.assertEqual(feed.label(base, feed.decision_view())['use'], 1)
        self.assertEqual(feed.features(), ct.build_features(100.5, 64., 200.5, 0., 40., feed.decision_view()))
        feed.observe(one); self.assertEqual((feed.filled(), feed.done()), (1, False))
        feed.observe(two); self.assertEqual((feed.filled(), feed.done(), feed.wrong()), (2, True, 0))

    def test_a_count_from_before_the_step_is_not_a_placement(self):
        feed = ct.BuildFeed(ct.build_fixture(TrackerTest.task))
        feed.observe(self.rows()[2]); self.assertEqual(feed.filled(), 0)


class SimulationTest(unittest.TestCase):
    def test_the_simulated_teacher_fills_every_blueprint(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/simulate_build_teacher.py'
        spec = importlib.util.spec_from_file_location('simulate_build_teacher', path)
        sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
        for seed in (101, 102, 103, 201, 301):
            for yaw in (0., 90., 180., 270.):
                with self.subTest(seed=seed, yaw=yaw):
                    r = sim.run(seed, yaw)
                    self.assertTrue(r['ok'], r); self.assertEqual(r['wrong'], 0)


if __name__ == '__main__':
    unittest.main()
