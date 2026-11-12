"""Jump course (design 2026-11-13): generator, fixture, world geometry, stages and the privileged teacher."""
import unittest

from minecraftrl import jump_course as jc


def standing(tracker, x, y, z, yaw=0.):
    return tracker.update(x, y, z, yaw, True)


class GeneratorTest(unittest.TestCase):
    def test_same_seed_same_course_and_seeds_differ(self):
        self.assertEqual(jc.generate_course(7), jc.generate_course(7))
        self.assertGreater(len({repr(jc.generate_course(s)['platforms']) for s in range(20)}), 15)

    def test_course_rules(self):
        for seed in range(200):
            with self.subTest(seed=seed):
                p = jc.generate_course(seed)['platforms']
                self.assertEqual(len(p), 5)
                self.assertEqual(p[0], dict(x0=-1, x1=1, z0=-1, z1=1, top=0, gap=0, block='stone_bricks'))
                self.assertEqual(p[-1]['block'], 'gold_block')
                self.assertEqual((p[-1]['x1'] - p[-1]['x0'], p[-1]['z1'] - p[-1]['z0']), (2, 2))
                for prev, cur in zip(p, p[1:]):
                    self.assertIn(cur['gap'], (1, 2))
                    self.assertEqual(cur['z0'], prev['z1'] + cur['gap'] + 1)
                    rise = cur['top'] - prev['top']
                    self.assertIn(rise, (-1, 0, 1))
                    if rise == 1: self.assertEqual(cur['gap'], 1)
                    self.assertTrue(-1 <= cur['top'] <= 2)
                    self.assertGreaterEqual(min(cur['x1'], prev['x1']) - max(cur['x0'], prev['x0']), 0)  # lateral overlap
                    self.assertTrue(-3 <= cur['x0'] and cur['x1'] <= 3)
                for mid in p[1:-1]:
                    self.assertEqual((mid['x1'] - mid['x0'], mid['z1'] - mid['z0'], mid['block']), (1, 2, 'stone_bricks'))


class FixtureTest(unittest.TestCase):
    def test_fixture_places_every_platform_block_after_the_fills(self):
        course = jc.generate_course(3)
        f = jc.course_fixture(course)
        self.assertEqual((f['schema'], f['rotate_with_facing'], f['give']), ('iron_demo_fixture_v1', True, []))
        self.assertEqual(f['course'], course)
        want = {(x, q['top'] - 1, z): q['block'] for q in course['platforms']
                for x in range(q['x0'], q['x1'] + 1) for z in range(q['z0'], q['z1'] + 1)}
        self.assertEqual({(b['dx'], b['dy'], b['dz']): b['block'] for b in f['blocks']}, want)
        self.assertEqual(len(f['blocks']), len(want))
        self.assertEqual(want[(0, -1, 0)], 'stone_bricks')                  # the block under the player
        floor = [r for r in f['fill'] if r['block'] == 'grass_block']
        self.assertEqual([(r['from'][1], r['to'][1]) for r in floor], [(-4, -4)])
        air = [r for r in f['fill'] if r['block'] == 'air'][0]
        self.assertGreaterEqual(air['to'][2], course['platforms'][-1]['z1'] + 2)
        self.assertTrue(any(r['block'] == 'barrier' for r in f['fill']))


class WorldTest(unittest.TestCase):
    course = dict(schema='jump_course_v1', seed=0, platforms=[
        dict(x0=-1, x1=1, z0=-1, z1=1, top=0, gap=0, block='stone_bricks'),
        dict(x0=0, x1=1, z0=3, z1=5, top=1, gap=1, block='stone_bricks'),
        dict(x0=-1, x1=1, z0=8, z1=10, top=0, gap=2, block='gold_block')])

    def test_south_facing_keeps_axes(self):
        w = jc.world_platforms(self.course, (100, 64, 200), 0.)       # yaw 0 = south = +z forward, right = -x
        self.assertEqual(w[0], dict(X0=99., X1=102., Z0=199., Z1=202., top=64., gap=0))
        self.assertEqual(w[1], dict(X0=99., X1=101., Z0=203., Z1=206., top=65., gap=1))

    def test_west_facing_turns_the_course(self):
        w = jc.world_platforms(self.course, (100, 64, 200), 90.)      # yaw 90 = west = -x forward, right = -z
        self.assertEqual((w[2]['X0'], w[2]['X1']), (90., 93.))
        self.assertEqual((w[1]['Z0'], w[1]['Z1']), (199., 201.))


class StageTest(unittest.TestCase):
    def setUp(self):
        self.t = jc.JumpTracker(jc.world_platforms(WorldTest.course, (100, 64, 200), 0.))

    def test_run_edge_turn_on_the_start(self):
        self.assertEqual(standing(self.t, 100.5, 64., 200.5).stage, 'run')
        self.assertEqual(standing(self.t, 100.4, 64., 201.8).stage, 'edge')    # 0.2 from the edge toward the next
        self.assertEqual(standing(self.t, 100.5, 64., 200.5, yaw=90.).stage, 'turn')

    def test_air_landing_progress_fall_and_goal(self):
        self.assertEqual(self.t.update(100.5, 65.1, 202.4, 0., False).stage, 'air')
        v = standing(self.t, 100.2, 65., 204.)
        self.assertEqual((v.stage, v.k, v.nxt), ('run', 1, 2))
        v = standing(self.t, 100.2, 64., 200.5)                   # back on the start: progress goes back too
        self.assertEqual((v.k, v.nxt), (0, 1))
        self.assertEqual(self.t.update(100.5, 62.7, 202.5, 0., False).stage, 'fallen')
        self.assertEqual(standing(self.t, 100.5, 64., 209.5).stage, 'goal')

    def test_overhanging_hitbox_still_stands_on_the_platform(self):
        v = standing(self.t, 100.5, 64., 202.25)                  # centre past the edge, hitbox still on it
        self.assertEqual((v.stage, v.k), ('edge', 0))
        self.assertEqual(v.edge, 0.)

    def test_landing_on_the_near_edge_is_a_whole_platform_from_the_next_jump(self):
        # Simulation 2026-11-13: after a jump up the centre hung 0.04 before platform 1; reading that as the edge sent
        # the teacher into a 5-block jump from the near side.
        v = standing(self.t, 100.2, 65., 202.96)
        self.assertEqual((v.stage, v.k), ('run', 1))
        self.assertGreater(v.edge, 3.)


class TeacherTest(unittest.TestCase):
    def setUp(self):
        self.w = jc.world_platforms(WorldTest.course, (100, 64, 200), 0.)
        self.t = jc.JumpTracker(self.w)

    def act(self, x, y, z, yaw, pitch, ground):
        view = self.t.update(x, y, z, yaw, ground)
        return view, jc.teacher_action(x, y, z, yaw, pitch, view, self.w)

    def test_run_walks_without_jumping_and_sprints_for_a_rise(self):
        _, a = self.act(100.5, 64., 200.5, 0., 30., True)
        self.assertEqual((a['forward'], a['jump'], a['sprint']), (1, 0, 1))     # platform 1 is one block higher
        self.assertTrue(all(abs(c) <= jc.MAX_TURN for c in a['camera']))

    def test_edge_jumps(self):
        _, a = self.act(100.4, 64., 201.8, 0., 30., True)
        self.assertEqual((a['forward'], a['jump']), (1, 1))

    def test_turn_only_turns(self):
        _, a = self.act(100.5, 64., 200.5, 90., 30., True)
        self.assertEqual((a['forward'], a['jump'], a['sprint']), (0, 0, 0))
        self.assertLess(a['camera'][1], 0)                       # target at yaw ~0: turn back from 90
        self.assertEqual(abs(a['camera'][1]), jc.MAX_TURN)

    def test_air_lets_go_near_the_landing(self):
        _, a = self.act(100.5, 65.4, 202.6, 0., 30., False)
        self.assertEqual((a['forward'], a['jump']), (1, 0))
        _, a = self.act(100.3, 65.2, 204.3, 0., 30., False)
        self.assertEqual(a['forward'], 0)

    def test_terminal_stages_get_no_label(self):
        self.assertIsNone(self.act(100.5, 62., 202., 0., 0., False)[1])
        self.assertIsNone(self.act(100.5, 64., 209.5, 0., 0., True)[1])


class RowTest(unittest.TestCase):
    def test_feed_reads_rows_after_the_fixture_is_applied(self):
        fixture = jc.course_fixture(WorldTest.course)
        feed = jc.JumpFeed(fixture)
        base = dict(t=1., controller='policy', server_x=100.5, server_y=64., server_z=200.5, yaw=0., pitch=20.,
                    server_on_ground=1, server_available=1)
        feed.observe(dict(base, demo=dict(demo_available=1, demo_fixture_state=1)))
        self.assertIsNone(feed.decision_view())                   # no origin yet
        feed.observe(dict(base, demo=dict(demo_available=1, demo_fixture_state=2, demo_fixture_origin_y=64)))
        self.assertEqual(feed.decision_view().stage, 'run')
        feed.observe(dict(base, server_on_ground=0, server_y=64.4, demo=dict(demo_available=1, demo_fixture_state=2)))
        self.assertEqual(feed.decision_view().stage, 'air')
        feed.observe(dict(base, server_on_ground=None, demo=dict(demo_available=1, demo_fixture_state=2)))
        self.assertIsNone(feed.decision_view())                   # a row without the ground flag is unknown


if __name__ == '__main__':
    unittest.main()
