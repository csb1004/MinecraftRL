"""Sensor-tier building feed (plan 2026-11-22 sensor build): blueprint instruction + client pose + crosshair + seen map."""
import copy
import unittest
from dataclasses import replace

from minecraftrl.build_recover import RecoverTracker, build2_features, teacher_action, world_build2
from minecraftrl.build_sensor import (BLUEPRINT_SCHEMA, SENSOR, SENSOR_BUILD_FEATURES, SensorBuildFeed, SensorBuildTracker,
                                      blueprint_cells, blueprint_of, blueprint_world)
from minecraftrl.build_task import build_fixture
from minecraftrl.place_task import ARENA, _at
from minecraftrl.pipeline.seen_map import SeenMap
from minecraftrl.pipeline.sensor import sensor_frame
from test_pipeline_sensor import info

TASK = dict(schema='build_task_v1', seed=7, shape='line3', cells=[[0, 0, 4], [0, 0, 3], [0, 0, 2]], block='cobblestone')
ORIGIN, YAW = (0, 64, 0), 0.


def plain(action):
    import numpy as np
    return None if action is None else {k: np.asarray(v).tolist() for k, v in action.items()}


def fixture(**extra):
    value = build_fixture(dict(TASK, teacher=SENSOR)); value.update(extra)
    return value


def frame(x=0.5, z=0.5, yaw=0., pitch=20., hit=None, face=None):
    base = sensor_frame(info(), 0.)
    kinds = list(base.ray_kinds); cells = list(base.ray_cells)
    kinds[-1], cells[-1] = ('air', None) if hit is None else ('grass_block', hit)
    return replace(base, position=(x, 64., z), eye=(x, 65.62, z), yaw=yaw, pitch=pitch, foot=(int(x // 1), 64, int(z // 1)),
                   ray_kinds=tuple(kinds), ray_cells=tuple(cells), hit_face=face)


def full_arena(origin=ORIGIN, yaw=YAW):
    """A seen map holding what the fixture builds: the floor solid, the two cells above it air."""
    seen = SeenMap()
    for dx in range(ARENA['x'][0], ARENA['x'][1] + 1):
        for dz in range(ARENA['z'][0], ARENA['z'][1] + 1):
            seen.mark(_at(origin, yaw, dx, -1, dz), 'grass_block', 1, 'ray')
            for dy in (0, 1): seen.mark(_at(origin, yaw, dx, dy, dz), 'air', 1, 'ray')
    return seen


class BlueprintTests(unittest.TestCase):
    def test_blueprint_is_a_block_matrix_with_its_build_order_and_nothing_of_the_world(self):
        bp = blueprint_of(fixture()['build'])
        self.assertEqual((bp['schema'], bp['size'], bp['min']), (BLUEPRINT_SCHEMA, [1, 1, 3], [0, 0, 2]))
        self.assertEqual(bp['blocks'], [[['cobblestone', 'cobblestone', 'cobblestone']]])      # blocks[x][y][z]
        self.assertEqual(bp['order'], [[0, 0, 4], [0, 0, 3], [0, 0, 2]])
        self.assertEqual(bp['frame'], 'start block; x right, y up, z forward')
        self.assertFalse({'fill', 'give', 'clutter', 'description'} & set(bp))

    def test_gaps_are_null_and_cells_come_in_build_order(self):
        corner = dict(TASK, shape='corner3', cells=[[1, 0, 4], [2, 0, 4], [1, 0, 3]])
        bp = blueprint_of(corner)
        self.assertEqual((bp['size'], bp['min']), ([2, 1, 2], [1, 0, 3]))
        self.assertEqual(bp['blocks'], [[['cobblestone', 'cobblestone']], [[None, 'cobblestone']]])
        self.assertEqual(blueprint_cells(bp), [((1, 0, 4), 'cobblestone'), ((2, 0, 4), 'cobblestone'), ((1, 0, 3), 'cobblestone')])

    def test_without_an_order_cells_go_bottom_up_far_first(self):
        bp = blueprint_of(dict(TASK, shape='stair3', cells=[[0, 0, 4], [0, 1, 4], [0, 0, 3]])); del bp['order']
        self.assertEqual([c for c, _ in blueprint_cells(bp)], [(0, 0, 4), (0, 0, 3), (0, 1, 4)])

    def test_world_cells_follow_the_start_pose(self):
        bp = blueprint_of(TASK)
        self.assertEqual(blueprint_world(bp, ORIGIN, 90.), [_at(ORIGIN, 90., *c) for c in TASK['cells']])


class SensorTrackerTests(unittest.TestCase):
    def test_a_fully_seen_arena_gives_the_privileged_teacher_view(self):
        # Parity: with everything seen, the sensor tracker sees the world the fixture builds.
        privileged = RecoverTracker(world_build2(TASK, ORIGIN, YAW))
        sensor = SensorBuildTracker(blueprint_of(TASK), ORIGIN, YAW); sensor.sync(full_arena())
        for pose in ((0.5, 64., 0.5, 0., 20.), (2.5, 64., -1.5, 40., 0.), (0.5, 64., 6.5, 180., 30.)):
            with self.subTest(pose=pose):
                a, b = privileged.update(*pose, None, None), sensor.update(*pose, None, None)
                self.assertEqual(a, b)
                self.assertEqual(build2_features(*pose, a), build2_features(*pose, b))
                self.assertEqual(plain(teacher_action(*pose, a)), plain(teacher_action(*pose, b)))

    def test_unseen_world_has_no_standpoint_and_unknown_cells_block(self):
        sensor = SensorBuildTracker(blueprint_of(TASK), ORIGIN, YAW); sensor.sync(SeenMap())
        view = sensor.update(0.5, 64., 0.5, 0., 20., None, None)
        self.assertEqual((view.stage, view.stand, view.way), ('reach', None, None))

    def test_a_block_seen_in_a_blueprint_cell_fills_it(self):
        seen = full_arena(); sensor = SensorBuildTracker(blueprint_of(TASK), ORIGIN, YAW)
        first = _at(ORIGIN, YAW, 0, 0, 4); seen.mark(first, 'cobblestone', 2, 'ray'); sensor.sync(seen)
        view = sensor.update(0.5, 64., 0.5, 0., 20., None, None)
        self.assertEqual((view.target, view.filled, view.left), (_at(ORIGIN, YAW, 0, 0, 3), 1, 2))
        self.assertIn(first, sensor.solids)


class SensorFeedTests(unittest.TestCase):
    def test_feed_starts_at_the_first_frame_and_gives_sensor_features_and_labels(self):
        feed = SensorBuildFeed(fixture()); seen = full_arena()
        self.assertIsNone(feed.decision_view())
        feed.observe_frame(frame(), seen)
        view = feed.decision_view()
        self.assertEqual(view.target, _at(ORIGIN, YAW, 0, 0, 4))
        self.assertEqual(len(feed.features()), len(SENSOR_BUILD_FEATURES))
        self.assertTrue(all(name.startswith('s_') for name in SENSOR_BUILD_FEATURES))
        self.assertIsNotNone(feed.label(None, view))
        self.assertFalse(feed.done())                     # success stays with the server placement record (judge)

    def test_fixture_world_values_never_change_the_sensor_decision(self):
        # Leak check: the arena fill, clutter, gifts and description are the fixture's world, not instruction.
        a = SensorBuildFeed(fixture()); b = SensorBuildFeed(fixture(fill=[], give=[], description='x'))
        b_build = copy.deepcopy(fixture()); b_build['build']['clutter'] = [[0, 0, 1]]
        c = SensorBuildFeed(b_build)
        for feed in (a, b, c): feed.observe_frame(frame(), full_arena())
        self.assertEqual(a.decision_view(), b.decision_view()); self.assertEqual(a.decision_view(), c.decision_view())
        self.assertEqual(a.features(), c.features())

    def test_place_stage_uses_the_client_crosshair_and_face(self):
        feed = SensorBuildFeed(fixture()); seen = full_arena()
        target = _at(ORIGIN, YAW, 0, 0, 2); floor = (target[0], target[1] - 1, target[2])
        feed.observe_frame(frame(z=0.5, pitch=50.), seen)                      # first cell: the far one
        seen.mark(_at(ORIGIN, YAW, 0, 0, 4), 'cobblestone', 2, 'ray'); seen.mark(_at(ORIGIN, YAW, 0, 0, 3), 'cobblestone', 2, 'ray')
        feed.observe_frame(frame(z=0.5, pitch=50., hit=floor, face=1), seen)   # face 1 = up: lands on the target
        self.assertEqual((feed.decision_view().stage, feed.decision_view().target), ('place', target))


if __name__ == '__main__':
    unittest.main()
