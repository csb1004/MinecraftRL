"""Bridging skill (design 2026-11-14): tasks, fixture, world geometry, stages, teacher, features and the feed."""
import json
import math
import unittest

from minecraftrl import bridge_task as bt


class TaskTest(unittest.TestCase):
    def test_same_seed_same_task_and_rules(self):
        self.assertEqual(bt.generate_bridge(7), bt.generate_bridge(7))
        self.assertEqual(json.loads(json.dumps(bt.generate_bridge(7))), bt.generate_bridge(7))
        seen = set()
        for seed in range(200):
            t = bt.generate_bridge(seed)
            self.assertIn(t['gap'], bt.GAPS); self.assertIn(t['goal_dx'], bt.GOAL_SHIFT)
            self.assertEqual(t['block'], 'cobblestone')
            seen.add((t['gap'], t['goal_dx']))
        self.assertEqual(len(seen), len(bt.GAPS) * len(bt.GOAL_SHIFT))

    def test_cells_run_from_the_start_edge_to_the_goal(self):
        for seed in range(50):
            t = bt.generate_bridge(seed)
            start, goal = bt.platforms(t)
            cells = bt.bridge_cells(t)
            self.assertEqual([c[2] for c in cells], list(range(start['z1'] + 1, goal['z0'])))
            self.assertTrue(all(c[0] == 0 and c[1] == start['top'] - 1 for c in cells))
            self.assertTrue(goal['x0'] <= 0 <= goal['x1'])                 # the line always reaches the goal


class FixtureTest(unittest.TestCase):
    def test_fixture_builds_platforms_floor_walls_and_gives_blocks(self):
        task = bt.generate_bridge(3)
        f = bt.bridge_fixture(task)
        self.assertEqual((f['schema'], f['rotate_with_facing'], f['bridge']), ('iron_demo_fixture_v1', True, task))
        self.assertEqual(f['give'], [dict(item='cobblestone', count=bt.GIVE_COUNT)])
        gold = [b for b in f['blocks'] if b['block'] == 'gold_block']; stone = [b for b in f['blocks'] if b['block'] == 'stone_bricks']
        self.assertEqual((len(gold), len(stone)), (6, 12))
        self.assertTrue(all(b['dy'] == -1 for b in f['blocks']))
        cells = {(c[0], c[2]) for c in bt.bridge_cells(task)}
        self.assertFalse(cells & {(b['dx'], b['dz']) for b in f['blocks']})        # the gap is open
        values = [v for r in f['fill'] for v in r['from'] + r['to']] + [b['dz'] for b in f['blocks']]
        self.assertLessEqual(max(abs(v) for v in values), 16)
        self.assertEqual([r['from'][1] for r in f['fill'] if r['block'] == 'grass_block'], [-4])
        self.assertTrue(any(r['block'] == 'barrier' for r in f['fill']))


class WorldTest(unittest.TestCase):
    task = dict(schema=bt.SCHEMA, seed=0, gap=3, goal_dx=1, block='cobblestone')

    def test_world_follows_the_start_facing(self):
        w = bt.world_bridge(self.task, (100, 64, 200), 0.)                   # south: +z forward, right = -x
        self.assertEqual(w['cells'], [(100, 63, 203), (100, 63, 204), (100, 63, 205)])
        self.assertEqual((w['goal']['Z0'], w['goal']['Z1'], w['goal']['X0'], w['goal']['X1']), (206., 208., 98., 101.))
        self.assertEqual(w['top'], 64.)
        self.assertAlmostEqual(bt.along(w, 100.5, 202.75), 2.25)
        w = bt.world_bridge(self.task, (100, 64, 200), 90.)                  # west: -x forward
        self.assertEqual(w['cells'][0], (97, 63, 200))

    def test_walk_heading_faces_away_from_the_line_ahead(self):
        w = bt.world_bridge(self.task, (100, 64, 200), 0.)
        self.assertAlmostEqual(bt.walk_heading(w, 100.5, 200.5), 180.)       # on the line: face the start (north)
        off = bt.walk_heading(w, 101.0, 200.5)                               # half a block left of the line (+x)
        self.assertGreater(abs(bt.wrap(off - 180.)), 5.)


class StageTest(unittest.TestCase):
    def setUp(self):
        self.w = bt.world_bridge(WorldTest.task, (100, 64, 200), 0.)
        self.t = bt.BridgeTracker(self.w)

    def test_turn_walk_aim_place(self):
        self.assertEqual(self.t.update(100.5, 64., 200.5, 0., 0., True).stage, 'turn')       # facing the gap
        self.assertEqual(self.t.update(100.5, 64., 200.5, 180., 80., True).stage, 'walk')    # facing the start
        v = self.t.update(100.5, 64., 203.27, 180., 80., True, (100, 63, 202), 1)            # past the edge, crosshair on the top
        self.assertEqual(v.stage, 'aim')
        v = self.t.update(100.5, 64., 203.27, 180., 82., True, (100, 63, 202), 3)            # the south face: lands in the cell
        self.assertEqual(v.stage, 'place')
        self.assertEqual(v.face, (100.5, 63.5, 203.))

    def test_a_server_placement_moves_the_edge_and_wrong_ones_are_counted(self):
        self.t.placed((100, 64, 202)); self.assertEqual(self.t.wrong, 1)
        self.t.placed((100, 63, 203)); self.assertEqual(self.t.filled(), 1)
        v = self.t.update(100.5, 64., 203.27, 180., 82., True, (100, 63, 202), 3)
        self.assertEqual((v.stage, v.left), ('walk', 2))                                      # the edge is one block further
        self.assertEqual(self.t.update(100.5, 64., 204.27, 180., 82., True, (100, 63, 203), 3).stage, 'place')

    def test_goal_and_fallen(self):
        self.assertEqual(self.t.update(100.5, 62.7, 203.5, 180., 80., False).stage, 'fallen')
        self.assertEqual(self.t.update(100.5, 64., 203.4, 180., 80., False).stage, 'walk')   # off the ground: no aiming
        for cell in self.w['cells']: self.t.placed(cell)
        self.assertEqual(self.t.update(100.5, 64., 205.9, 180., 80., True).stage, 'walk')   # the centre is not on the goal yet
        self.assertEqual(self.t.update(100.5, 64., 206.6, 180., 80., False).stage, 'walk')   # not on the ground yet
        self.assertEqual(self.t.update(100.5, 64., 206.6, 180., 80., True).stage, 'goal')


class TeacherTest(unittest.TestCase):
    def setUp(self):
        self.w = bt.world_bridge(WorldTest.task, (100, 64, 200), 0.)
        self.t = bt.BridgeTracker(self.w)

    def act(self, x, y, z, yaw, pitch, ground=True, hit=None, face=None):
        view = self.t.update(x, y, z, yaw, pitch, ground, hit, face)
        return view, bt.teacher_action(x, y, z, yaw, pitch, view)

    def test_turn_and_walk_sneak_backward(self):
        _, a = self.act(100.5, 64., 200.5, 0., 0.)
        self.assertEqual((a['sneak'], a['back'], a['forward'], a['use']), (1, 0, 0, 0))
        self.assertTrue(all(abs(c) <= bt.MAX_TURN for c in a['camera']))
        self.assertEqual(a['camera'][0], bt.MAX_TURN)                                        # toward 80 degrees down
        _, a = self.act(100.5, 64., 200.5, 180., 80.)
        self.assertEqual((a['sneak'], a['back'], a['forward']), (1, 1, 0))

    def test_aim_turns_to_the_face_and_place_uses(self):
        _, a = self.act(100.5, 64., 203.27, 180., 70., True, (100, 63, 202), 1)
        self.assertEqual((a['use'], a['back'], a['sneak']), (0, 0, 1))
        self.assertGreater(a['camera'][0], 0)
        _, a = self.act(100.5, 64., 203.27, 180., 82., True, (100, 63, 202), 3)
        self.assertEqual((a['use'], a['sneak']), (1, 1))

    def test_terminal_stages_get_no_label(self):
        self.assertIsNone(self.act(100.5, 64., 206.6, 180., 80.)[1])
        self.assertIsNone(self.act(100.5, 62., 203.5, 180., 80., False)[1])


class FeatureTest(unittest.TestCase):
    def test_features_follow_the_teacher_targets(self):
        w = bt.world_bridge(WorldTest.task, (100, 64, 200), 0.)
        t = bt.BridgeTracker(w)
        self.assertEqual(bt.BRIDGE_FEATURES, ('h_sin', 'h_cos', 'a_pitch', 'edge', 'left'))
        view = t.update(100.5, 64., 200.5, 0., 0., True)
        f = dict(zip(bt.BRIDGE_FEATURES, bt.bridge_features(100.5, 64., 200.5, 0., 0., view)))
        self.assertAlmostEqual(abs(f['h_sin']), 0., places=6); self.assertAlmostEqual(f['h_cos'], -1.)   # turn around
        self.assertAlmostEqual(f['a_pitch'], 80. / 60. if 80. / 60. <= 1 else 1.)
        self.assertAlmostEqual(f['edge'], 2.5 / 3.)                          # the start edge is 2.5 blocks ahead
        self.assertAlmostEqual(f['left'], 3. / 5.)
        view = t.update(100.5, 64., 203.27, 180., 82., True, (100, 63, 202), 3)
        f = dict(zip(bt.BRIDGE_FEATURES, bt.bridge_features(100.5, 64., 203.27, 180., 82., view)))
        self.assertLess(f['edge'], 0.)
        self.assertEqual(bt.bridge_features(0., 0., 0., 0., 0., None), [0.] * 5)


class FeedTest(unittest.TestCase):
    def rows(self):
        demo = dict(demo_available=1, demo_fixture_state=2, demo_fixture_origin_y=64, demo_hit_block='air')
        place = dict(place_available=1, place_count=0, place_last_x=-999, place_last_y=-999, place_last_z=-999)
        base = dict(t=1., controller='policy', server_x=100.5, server_y=64., server_z=200.5, server_on_ground=1, yaw=0.,
                    pitch=0., client_x=100.5, client_y=64., client_z=200.5, client_on_ground=1, server_available=1,
                    hit_x=0, hit_y=0, hit_z=0, hit_face=0, demo=demo, place=place)
        edge = dict(base, yaw=180., pitch=82., client_z=203.27, server_z=203.1, hit_x=100, hit_y=63, hit_z=202, hit_face=3,
                    demo=dict(demo, demo_hit_block='stone_bricks'))
        placed = dict(edge, place=dict(place, place_count=1, place_last_x=100, place_last_y=63, place_last_z=203))
        goal = dict(base, server_z=206.6, client_z=206.7, yaw=180.)
        fell = dict(base, server_y=62.5, server_on_ground=0, client_y=62.3, client_on_ground=0)
        return base, edge, placed, goal, fell

    def test_decisions_on_the_client_success_and_falls_on_the_server(self):
        feed = bt.BridgeFeed(bt.bridge_fixture(WorldTest.task))
        base, edge, placed, goal, fell = self.rows()
        feed.observe(base)
        self.assertEqual(feed.decision_view().stage, 'turn')
        feed.observe(edge)
        self.assertEqual(feed.decision_view().stage, 'place')
        self.assertEqual(feed.server_view().stage, 'walk')                    # the server is not past the edge yet
        self.assertEqual(feed.label(edge, feed.decision_view())['use'], 1)
        self.assertEqual(feed.features(), bt.bridge_features(100.5, 64., 203.27, 180., 82., feed.decision_view()))
        feed.observe(placed)
        self.assertEqual((feed.filled(), feed.wrong()), (1, 0))
        feed.observe(goal)
        self.assertEqual(feed.server_view().stage, 'goal')
        feed2 = bt.BridgeFeed(bt.bridge_fixture(WorldTest.task)); feed2.observe(base); feed2.observe(fell)
        self.assertEqual(feed2.server_view().stage, 'fallen')

    def test_a_count_from_before_the_step_is_not_a_placement(self):
        feed = bt.BridgeFeed(bt.bridge_fixture(WorldTest.task))
        feed.observe(self.rows()[2])
        self.assertEqual(feed.filled(), 0)


class SimulationTest(unittest.TestCase):
    def test_the_simulated_teacher_fills_every_cell_and_reaches_the_goal(self):
        # Offline check 2026-11-14 (scripts/simulate_bridge_teacher.py): sneaking backward with the sneak edge back-off,
        # aiming from the sneaking eye and placing every 4 ticks or slower.
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / 'scripts/simulate_bridge_teacher.py'
        spec = importlib.util.spec_from_file_location('simulate_bridge_teacher', path)
        sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
        for seed in (101, 102, 201, 301):
            for yaw in (0., 90., 180., 270.):
                with self.subTest(seed=seed, yaw=yaw):
                    r = sim.run(seed, yaw)
                    self.assertTrue(r['ok'], r); self.assertEqual(r['wrong'], 0)
                    self.assertLess(r['steps'], 200)


if __name__ == '__main__':
    unittest.main()
