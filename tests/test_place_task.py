"""Block placing skill (design 2026-11-14): targets, fixture, world geometry, stages, teacher, features and the feed."""
import math
import unittest

from minecraftrl import place_task as pt


class TargetTest(unittest.TestCase):
    def test_same_seed_same_target_and_rules(self):
        self.assertEqual(pt.generate_target(7), pt.generate_target(7))
        import json
        self.assertEqual(json.loads(json.dumps(pt.generate_target(7))), pt.generate_target(7))   # JSON round trip
        kinds = set()
        for seed in range(300):
            with self.subTest(seed=seed):
                t = pt.generate_target(seed)
                dx, dy, dz = t['target']
                self.assertTrue(2 <= dz <= 5 and -2 <= dx <= 2 and dy in (0, 1))
                if dy == 0: self.assertEqual(t['supports'], [])            # the floor holds it
                else:
                    self.assertEqual(len(t['supports']), 1)
                    s = t['supports'][0]
                    self.assertEqual(sum(abs(a - b) for a, b in zip(s, t['target'])), 1)   # one face touches the target
                    self.assertNotEqual(list(s), [dx, 2, dz])              # never hanging from above
                    kinds.add('below' if s[1] == 0 else 'side')
                self.assertEqual(t['block'], 'cobblestone')
        self.assertEqual(kinds, {'below', 'side'})
        self.assertGreater(len({repr(pt.generate_target(s)) for s in range(40)}), 25)

    def test_every_target_has_a_face_seen_from_the_start_and_reached_after_the_walk(self):
        # Offline check 2026-11-14: a side support between the start column and the target hides its only face.
        for seed in list(range(300)) + list(range(101, 113)) + list(range(201, 205)) + list(range(301, 309)):
            task = pt.generate_target(seed)
            w = pt.world_task(task, (0, 64, 0), 0.)
            with self.subTest(seed=seed):
                self.assertIsNotNone(pt.choose_face((0.5, 64. + pt.EYE, 0.5), w['target'], w['solids']))
                centre = tuple(c + .5 for c in w['target'])
                z = .5                                              # walk straight ahead until the target is in reach
                while math.dist((.5, 64. + pt.EYE, z), centre) > pt.REACH: z += .2
                face = pt.choose_face((.5, 64. + pt.EYE, z), w['target'], w['solids'])
                self.assertIsNotNone(face)
                self.assertLessEqual(math.dist((.5, 64. + pt.EYE, z), face['centre']), 4.5)

    def test_fixture_builds_the_arena_supports_and_gives_blocks(self):
        task = pt.generate_target(3)
        f = pt.place_fixture(task)
        self.assertEqual((f['schema'], f['rotate_with_facing']), ('iron_demo_fixture_v1', True))
        self.assertEqual(f['place'], task)
        self.assertEqual(f['give'], [dict(item='cobblestone', count=pt.GIVE_COUNT)])
        self.assertEqual([(b['dx'], b['dy'], b['dz']) for b in f['blocks']], [tuple(s) for s in task['supports']])
        self.assertTrue(all(b['block'] == 'stone_bricks' for b in f['blocks']))
        values = [v for r in f['fill'] for v in r['from'] + r['to']]
        self.assertLessEqual(max(abs(v) for v in values), 16)
        floor = [r for r in f['fill'] if r['block'] == 'grass_block']
        self.assertEqual([(r['from'][1], r['to'][1]) for r in floor], [(-1, -1)])
        self.assertTrue(any(r['block'] == 'barrier' for r in f['fill']))


class WorldTest(unittest.TestCase):
    task = dict(schema=pt.SCHEMA, seed=0, target=(1, 1, 4), supports=[(1, 0, 4)], block='cobblestone')

    def test_world_cells_follow_the_start_facing(self):
        w = pt.world_task(self.task, (100, 64, 200), 0.)                 # south: +z forward, right = -x
        self.assertEqual(w['target'], (99, 65, 204))
        self.assertIn((99, 64, 204), w['solids'])                         # the support
        self.assertIn((100, 63, 203), w['solids'])                        # floor under the arena
        w = pt.world_task(self.task, (100, 64, 200), 90.)                # west: -x forward, right = -z
        self.assertEqual(w['target'], (96, 65, 199))

    def test_the_face_is_the_visible_support_face_nearest_the_eye(self):
        w = pt.world_task(self.task, (100, 64, 200), 0.)
        face = pt.choose_face((100.5, 65.62, 200.5), w['target'], w['solids'])
        self.assertEqual(face['block'], (99, 64, 204))
        self.assertEqual(face['normal'], (0, 1, 0))                       # the top of the support
        self.assertEqual(face['centre'], (99.5, 65., 204.5))
        side = dict(self.task, supports=[(2, 1, 4)])                     # support to the right of the target (world -x)
        w = pt.world_task(side, (100, 64, 200), 0.)
        face = pt.choose_face((100.5, 65.62, 200.5), w['target'], w['solids'])
        self.assertEqual((face['block'], face['normal']), ((98, 65, 204), (1, 0, 0)))


class StageTest(unittest.TestCase):
    def setUp(self):
        self.w = pt.world_task(WorldTest.task, (100, 64, 200), 0.)
        self.t = pt.PlaceTracker(self.w)

    def test_reach_aim_place(self):
        self.assertEqual(self.t.update(100.5, 64., 199.5, 0., 0., None, None).stage, 'reach')
        v = self.t.update(100.5, 64., 201.5, 0., 30., (100, 63, 203), 1)        # looking at the floor in front
        self.assertEqual(v.stage, 'aim')
        v = self.t.update(100.5, 64., 201.5, 0., 30., (99, 64, 204), 1)         # the top of the support: lands in the target
        self.assertEqual(v.stage, 'place')
        self.assertEqual(self.t.update(100.5, 64., 201.5, 0., 30., (99, 64, 204), 2).stage, 'aim')   # its north face

    def test_done_on_a_server_placement_at_the_target_and_wrong_ones_are_counted(self):
        self.t.placed((100, 64, 203))
        self.assertEqual(self.t.wrong, 1)
        self.t.placed((99, 65, 204))
        self.assertTrue(self.t.done)
        self.assertEqual(self.t.update(100.5, 64., 201.5, 0., 30., None, None).stage, 'done')


class TeacherTest(unittest.TestCase):
    def setUp(self):
        self.w = pt.world_task(WorldTest.task, (100, 64, 200), 0.)
        self.t = pt.PlaceTracker(self.w)

    def act(self, x, y, z, yaw, pitch, hit, face):
        view = self.t.update(x, y, z, yaw, pitch, hit, face)
        return view, pt.teacher_action(x, y, z, yaw, pitch, view)

    def test_reach_walks_toward_the_target(self):
        _, a = self.act(100.5, 64., 197.5, 0., 0., None, None)
        self.assertEqual((a['forward'], a['use'], a['back']), (1, 0, 0))
        self.assertTrue(all(abs(c) <= pt.MAX_TURN for c in a['camera']))

    def test_aim_turns_to_the_face_and_place_uses(self):
        _, a = self.act(100.5, 64., 201.5, 0., 0., (100, 63, 203), 1)
        self.assertEqual((a['forward'], a['use']), (0, 0))
        self.assertGreater(a['camera'][0], 0)                                    # look down to the support top
        _, a = self.act(100.5, 64., 201.5, 0., 30., (99, 64, 204), 1)
        self.assertEqual(a['use'], 1)

    def test_aim_steps_back_when_standing_on_the_target(self):
        _, a = self.act(99.5, 64., 204.2, 180., 60., None, None)
        self.assertEqual(a['back'], 1)

    def test_done_gets_no_label(self):
        self.t.placed((99, 65, 204))
        self.assertIsNone(self.act(100.5, 64., 201.5, 0., 0., None, None)[1])


class FeatureTest(unittest.TestCase):
    def test_features_point_at_the_target_and_the_face(self):
        from minecraftrl.reacquire_teacher import point_delta
        w = pt.world_task(WorldTest.task, (100, 64, 200), 0.)
        t = pt.PlaceTracker(w)
        view = t.update(100.5, 64., 201.5, 0., 0., None, None)
        f = dict(zip(pt.PLACE_FEATURES, pt.place_features(100.5, 64., 201.5, 0., 0., view)))
        self.assertEqual(pt.PLACE_FEATURES, ('t_sin', 't_cos', 't_pitch', 't_dist', 'f_sin', 'f_cos', 'f_pitch'))
        dp, dy = point_delta((100.5, 64. + pt.EYE, 201.5), 0., 0., (99.5, 65.5, 204.5))
        self.assertAlmostEqual(f['t_sin'], math.sin(math.radians(dy)))
        self.assertAlmostEqual(f['t_pitch'], max(-1., min(1., dp / 60.)))
        self.assertAlmostEqual(f['t_dist'], min(1., math.dist((100.5, 65.62, 201.5), (99.5, 65.5, 204.5)) / 6.))
        fp, fy = point_delta((100.5, 64. + pt.EYE, 201.5), 0., 0., (99.5, 65., 204.5))
        self.assertAlmostEqual(f['f_cos'], math.cos(math.radians(fy)))
        self.assertAlmostEqual(f['f_pitch'], max(-1., min(1., fp / 60.)))
        t.placed((99, 65, 204))
        self.assertEqual(pt.place_features(100.5, 64., 201.5, 0., 0., t.update(100.5, 64., 201.5, 0., 0., None, None)), [0.] * 7)
        self.assertEqual(pt.place_features(0., 0., 0., 0., 0., None), [0.] * 7)


class FeedTest(unittest.TestCase):
    def rows(self):
        demo = dict(demo_available=1, demo_fixture_state=2, demo_fixture_origin_y=64, demo_hit_block='air')
        base = dict(t=1., controller='policy', server_x=100.5, server_y=64., server_z=200.5, server_on_ground=1, yaw=0.,
                    pitch=30., client_x=100.5, client_y=64., client_z=201.5, client_on_ground=1, server_available=1,
                    hit_x=0, hit_y=0, hit_z=0, hit_face=0, demo=demo,
                    place=dict(place_available=1, place_count=0, place_last_x=-999, place_last_y=-999, place_last_z=-999))
        aimed = dict(base, hit_x=99, hit_y=64, hit_z=204, hit_face=1, demo=dict(demo, demo_hit_block='stone_bricks'))
        wrong = dict(aimed, place=dict(place_available=1, place_count=1, place_last_x=100, place_last_y=64, place_last_z=203))
        done = dict(aimed, place=dict(place_available=1, place_count=2, place_last_x=99, place_last_y=65, place_last_z=204))
        return base, aimed, wrong, done

    def test_decisions_follow_the_client_pose_and_the_crosshair_and_success_the_server_record(self):
        feed = pt.PlaceFeed(pt.place_fixture(WorldTest.task))
        base, aimed, wrong, done = self.rows()
        feed.observe(base)
        self.assertEqual(feed.decision_view().stage, 'aim')                   # within reach, crosshair on nothing
        self.assertEqual(feed.features(), pt.place_features(100.5, 64., 201.5, 0., 30., feed.decision_view()))
        feed.observe(aimed)
        self.assertEqual(feed.decision_view().stage, 'place')
        self.assertEqual(feed.label(aimed, feed.decision_view())['use'], 1)
        feed.observe(wrong)
        self.assertFalse(feed.done()); self.assertEqual(feed.wrong(), 1)
        feed.observe(done)
        self.assertTrue(feed.done()); self.assertEqual(feed.decision_view().stage, 'done')

    def test_a_count_from_before_the_step_is_not_a_placement(self):
        feed = pt.PlaceFeed(pt.place_fixture(WorldTest.task))
        base, _, _, done = self.rows()
        feed.observe(done)                                 # the first row already reports a placement at the target
        self.assertFalse(feed.done())


if __name__ == '__main__':
    unittest.main()
