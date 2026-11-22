"""Beacon pyramid demo (user request 2026-11-23): a multi-material blueprint, kind-aware judging and seen-map filling,
and the block the current cell needs (for the session's slot-selection rule)."""
import unittest

from minecraftrl.build_beacon import BEACON_KINDS, beacon_build
from minecraftrl.build_recover import RecoverTracker, world_build2
from minecraftrl.build_sensor import SensorBuildFeed, SensorBuildTracker, blueprint_cells, blueprint_of, sensor_fixture
from minecraftrl.build_task import generate_build
from minecraftrl.place_task import GIVE_COUNT, _at
from test_build_sensor import ORIGIN, YAW, frame, full_arena

GOLD, IRON, DIAMOND, BEACON = 'gold_block', 'iron_block', 'diamond_block', 'beacon'


class BeaconBlueprintTests(unittest.TestCase):
    def test_a_3x3_base_of_gold_corners_iron_edges_and_a_diamond_centre_then_the_beacon_on_top(self):
        task = beacon_build(1501)
        self.assertEqual((task['shape'], len(task['cells'])), ('beacon', 10))
        base, top = task['cells'][:9], task['cells'][9]
        dx, dz = base[4][0], base[4][2] - 1                    # the centre cell is (dx, 0, dz + 1)
        self.assertIn(dx, (-1, 0, 1)); self.assertIn(dz, (3, 4))
        self.assertEqual(top, [dx, 1, dz + 1, BEACON])
        # Far row first, then the middle and the near row, left to right; bottom layer before the beacon.
        self.assertEqual([c[:3] for c in base], [[dx + x, 0, dz + z] for z in (2, 1, 0) for x in (-1, 0, 1)])
        kinds = {(c[0] - dx, c[2] - dz): c[3] for c in base}
        self.assertEqual({kinds[k] for k in ((-1, 0), (1, 0), (-1, 2), (1, 2))}, {GOLD})
        self.assertEqual({kinds[k] for k in ((0, 0), (-1, 1), (1, 1), (0, 2))}, {IRON})
        self.assertEqual(kinds[(0, 1)], DIAMOND)
        self.assertEqual(beacon_build(1501), task)            # the id fixes the anchor
        self.assertEqual(BEACON_KINDS, (GOLD, IRON, DIAMOND, BEACON))

    def test_the_blueprint_matrix_holds_each_cells_block(self):
        task = beacon_build(1502); bp = blueprint_of(task)
        self.assertEqual(bp['size'], [3, 2, 3])
        self.assertEqual(bp['blocks'][1][0][1], DIAMOND); self.assertEqual(bp['blocks'][1][1][1], BEACON)
        self.assertIsNone(bp['blocks'][0][1][0])               # nothing asked above a corner
        self.assertEqual(blueprint_cells(bp)[-1], (tuple(task['cells'][9][:3]), BEACON))

    def test_the_fixture_gives_every_kind_in_first_use_order_and_single_kind_tasks_keep_their_gift(self):
        fixture = sensor_fixture(beacon_build(1501))
        self.assertEqual([g['item'] for g in fixture['give']], [GOLD, IRON, DIAMOND, BEACON])   # hotbar slots 1-4
        self.assertEqual([g['count'] for g in fixture['give']], [8, 8, 5, 5])                  # need + 4 spare
        self.assertIn('beacon', fixture['description'])
        self.assertEqual(sensor_fixture(generate_build(103))['give'], [dict(item='cobblestone', count=GIVE_COUNT)])


class KindJudgeTests(unittest.TestCase):
    def test_the_judge_counts_a_wrong_kind_in_a_blueprint_cell_as_wrong(self):
        task = beacon_build(1501); world = world_build2(task, ORIGIN, YAW)
        first = _at(ORIGIN, YAW, *task['cells'][0][:3])
        self.assertEqual(world['cells'][0], first); self.assertEqual(world['kinds'][first], GOLD)
        tracker = RecoverTracker(world)
        tracker.placed(first, IRON)
        self.assertEqual((tracker.wrong, tracker.filled_cells), (1, set()))
        self.assertIn(first, tracker.solids)
        second = _at(ORIGIN, YAW, *task['cells'][1][:3])
        tracker.placed(second, IRON)
        self.assertEqual((tracker.wrong, tracker.filled_cells), (1, {second}))

    def test_single_kind_tasks_ignore_the_placed_block_name(self):
        task = generate_build(103); tracker = RecoverTracker(world_build2(task, ORIGIN, YAW))
        self.assertEqual(world_build2(task, ORIGIN, YAW)['kinds'], {})
        cell = _at(ORIGIN, YAW, *task['cells'][0]); tracker.placed(cell, 'dirt')
        self.assertEqual((tracker.wrong, tracker.filled_cells), (0, {cell}))
        tracker.placed(_at(ORIGIN, YAW, *task['cells'][1]))       # callers without a block name still work
        self.assertEqual(len(tracker.filled_cells), 2)


class SensorKindTests(unittest.TestCase):
    def test_a_seen_block_fills_its_cell_only_when_it_is_the_blueprint_kind(self):
        task = beacon_build(1501); bp = blueprint_of(task)
        first = _at(ORIGIN, YAW, *task['cells'][0][:3]); seen = full_arena()
        seen.mark(first, IRON, 2, 'ray')
        tracker = SensorBuildTracker(bp, ORIGIN, YAW); tracker.sync(seen)
        self.assertEqual(tracker.filled_cells, set()); self.assertIn(first, tracker.solids)
        seen.mark(first, GOLD, 3, 'ray'); tracker.sync(seen)
        self.assertEqual(tracker.filled_cells, {first})

    def test_the_feed_names_the_block_of_the_current_cell(self):
        task = beacon_build(1501); feed = SensorBuildFeed(sensor_fixture(task)); seen = full_arena()
        self.assertIsNone(feed.needed_kind())                  # no frame yet
        feed.observe_frame(frame(), seen)
        self.assertEqual(feed.needed_kind(), GOLD)
        seen.mark(_at(ORIGIN, YAW, *task['cells'][0][:3]), GOLD, 2, 'ray'); feed.observe_frame(frame(), seen)
        self.assertEqual(feed.needed_kind(), IRON)
        for c in task['cells']: seen.mark(_at(ORIGIN, YAW, *c[:3]), c[3], 3, 'ray')
        feed.observe_frame(frame(), seen)
        self.assertIsNone(feed.needed_kind())                  # everything seen in place

    def test_single_kind_feeds_need_their_one_block(self):
        feed = SensorBuildFeed(sensor_fixture(generate_build(103))); feed.observe_frame(frame(), full_arena())
        self.assertEqual(feed.needed_kind(), 'cobblestone')


if __name__ == '__main__':
    unittest.main()
