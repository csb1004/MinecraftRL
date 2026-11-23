"""Sensor-tier repair (user request 2026-11-23): the anomaly monitor (seen map, in-hand counts, blueprint), the dig head's
stages and teacher (reach, move, dig, collect), the composite feed of the beacon demo (building head + dig head), the
server-record judge, and the dig training tasks with exact gifts."""
import math
import unittest
from dataclasses import replace

from minecraftrl.build_beacon import beacon_build
from minecraftrl.build_recover import feed_for, look, stages_for
from minecraftrl.build_sensor import blueprint_cells, blueprint_of
from minecraftrl.build_sensor_repair import (COLLECT_POINT_REACH, COLLECT_ROWS, COMBINED_STAGES, DIG2_FEATURES, DIG_FEATURES, DIG_KINDS,
                                             DIG_STAGES, ITEM_HALF, LEDGER_ROWS, PICKAXE, SENSOR_DIG, SENSOR_REPAIR, SUPPORT,
                                             RepairJudge, RepairMonitor, SensorRepairFeed, blueprint_empty, dig_action,
                                             dig_task, item_centre, repair_fixture)
from minecraftrl.place_task import FACE_NORMALS, _at, _centre
from minecraftrl.reacquire_teacher import point_delta
from test_build_sensor import ORIGIN, YAW, full_arena
from test_build_sensor import frame as base_frame

GOLD, IRON, DIAMOND, BEACON = 'gold_block', 'iron_block', 'diamond_block', 'beacon'
ONE = dict(schema='build_task_v1', seed=1, shape='dig1', cells=[], empty=[[0, 0, 3]], wrong=[[0, 0, 3, GOLD]], supports=[],
           block=GOLD)


def world(*c): return _at(ORIGIN, YAW, *c)


def frame(inventory=None, **kw):
    f = base_frame(**kw)
    return f if inventory is None else replace(f, inventory=dict(inventory))


def sighted(f, *sightings):
    """Frame `f` whose rays met these (kind, point) entities."""
    return replace(f, ray_entities=tuple(k for k, _ in sightings), ray_entity_points=tuple(p for _, p in sightings),
                   ray_entity_dist=tuple(2. for _ in sightings))


def aimed(cell, normal, x=0.5, z=0.5, **extra):
    """A frame at (x, z) looking straight at the centre of `cell`'s face `normal`, inventory overridable."""
    centre = tuple(c + .5 * k for c, k in zip(_centre(cell), normal))
    dpitch, dyaw = point_delta((x, 64. + 1.62, z), 0., 0., centre)
    return replace(frame(x=x, z=z, yaw=dyaw, pitch=dpitch), **extra)


def row(n=0, place=None, destroy=None, inventory=None):
    """A telemetry row with the fixture applied at ORIGIN facing YAW; place = (count, cell, block), destroy = (count, cell,
    block)."""
    demo = dict(demo_available=1, demo_fixture_state=2, demo_fixture_origin_y=ORIGIN[1], demo_destroys=0)
    if destroy is not None:
        count, cell, block = destroy
        demo.update(demo_destroys=count, demo_last_destroy_removed=1, demo_last_destroyed_block=block,
                    **{'demo_last_destroy_' + a: v for a, v in zip('xyz', cell)})
    p = dict(place_available=1, place_count=0)
    if place is not None:
        count, cell, block = place
        p.update(place_count=count, place_last_block=block, **{'place_last_' + a: v for a, v in zip('xyz', cell)})
    return dict(step=n, server_available=1, server_x=ORIGIN[0] + .5, server_y=float(ORIGIN[1]), server_z=ORIGIN[2] + .5,
                yaw=YAW, demo=demo, place=p, inventory=dict(inventory or {}))


class TaskTests(unittest.TestCase):
    def test_dig_tasks_set_wrong_blocks_the_blueprint_marks_empty_some_on_a_support(self):
        tasks = [dig_task(s) for s in range(1601, 1613)]
        self.assertEqual(dig_task(1601), tasks[0])                       # the id fixes the task
        self.assertEqual({len(t['wrong']) for t in tasks}, {1, 2, 3})
        for t in tasks:
            self.assertEqual(t['cells'], [])
            self.assertEqual(t['empty'], [w[:3] for w in t['wrong']])
            self.assertTrue(all(w[3] in DIG_KINDS for w in t['wrong']))
            self.assertEqual(t['supports'], [[w[0], 0, w[2]] for w in t['wrong'] if w[1] == 1])
            for w in t['wrong']:
                self.assertTrue(-3 <= w[0] <= 3 and 2 <= w[2] <= 6)
        self.assertTrue(any(t['supports'] for t in tasks) and any(w[1] == 0 for t in tasks for w in t['wrong']))

    def test_the_blueprint_marks_cells_that_must_stay_empty_as_air(self):
        bp = blueprint_of(ONE)
        self.assertEqual((bp['size'], bp['min'], bp['blocks']), ([1, 1, 1], [0, 0, 3], [[['air']]]))
        self.assertEqual(blueprint_cells(bp), []); self.assertEqual(blueprint_empty(bp), [(0, 0, 3)])
        self.assertNotIn('air', str(blueprint_of(beacon_build(1501))['blocks']))   # tasks without `empty` keep their matrix

    def test_fixtures_give_exactly_what_the_blueprint_needs_and_the_pickaxe_last(self):
        fixture = repair_fixture(beacon_build(1501), SENSOR_REPAIR)
        self.assertEqual([(g['item'], g['count']) for g in fixture['give']],
                         [(GOLD, 4), (IRON, 4), (DIAMOND, 1), (BEACON, 1), (PICKAXE, 1)])
        self.assertEqual(fixture['build']['teacher'], SENSOR_REPAIR); self.assertEqual(fixture['blocks'], [])
        task = dict(ONE, supports=[[1, 0, 4]], wrong=[[0, 0, 3, GOLD], [1, 1, 4, BEACON]], empty=[[0, 0, 3], [1, 1, 4]])
        dig = repair_fixture(task, SENSOR_DIG)
        self.assertEqual([g['item'] for g in dig['give']], list(DIG_KINDS) + [PICKAXE])
        self.assertEqual([g['count'] for g in dig['give']], [1] * 5)
        self.assertEqual(dig['blocks'], [dict(dx=1, dy=0, dz=4, block=SUPPORT), dict(dx=0, dy=0, dz=3, block=GOLD),
                                         dict(dx=1, dy=1, dz=4, block=BEACON)])
        with self.assertRaises(ValueError): repair_fixture(ONE, SENSOR_REPAIR)
        with self.assertRaises(ValueError): repair_fixture(beacon_build(1501), SENSOR_DIG)

    def test_the_build_module_names_the_repair_feeds_and_stages(self):
        self.assertIsInstance(feed_for(repair_fixture(ONE, SENSOR_DIG)), SensorRepairFeed)
        self.assertEqual(stages_for(repair_fixture(ONE, SENSOR_DIG)), DIG_STAGES)
        self.assertEqual(stages_for(repair_fixture(beacon_build(1501), SENSOR_REPAIR)), COMBINED_STAGES)
        self.assertEqual(COMBINED_STAGES, ('reach', 'move', 'aim', 'place', 'done', 'dig_reach', 'dig_move', 'dig_dig',
                                           'dig_collect', 'dig_done'))
        self.assertEqual(len(DIG_FEATURES), 11)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        task = beacon_build(1501)
        self.kind_of = {world(*c[:3]): c[3] for c in task['cells']}
        self.first, self.second = world(*task['cells'][0][:3]), world(*task['cells'][1][:3])
        self.monitor = RepairMonitor(self.kind_of, [], {GOLD: 4, IRON: 4, DIAMOND: 1, BEACON: 1}, check_kinds=True)

    def test_a_block_of_another_kind_in_a_blueprint_cell_is_an_anomaly_and_an_out_of_order_fill_is_not(self):
        seen = full_arena(); self.monitor.update(seen, {GOLD: 4})
        seen.mark(self.first, IRON, 2, 'ray')
        self.assertEqual(self.monitor.update(seen, {GOLD: 4}), {self.first: ('wrong_kind', IRON)})
        seen.mark(self.first, 'air', 3, 'ray'); seen.mark(self.second, IRON, 3, 'ray')   # iron where iron belongs
        self.assertEqual(self.monitor.update(seen, {GOLD: 4}), {})
        events = self.monitor.pop_events()
        self.assertEqual([e['event'] for e in events], ['anomaly', 'anomaly_cleared'])
        self.assertEqual((events[0]['rule'], events[0]['block'], events[0]['expected']), ('wrong_kind', IRON, GOLD))

    def test_a_given_block_appearing_outside_the_blueprint_is_a_stray_but_a_block_there_from_the_start_is_not(self):
        seen = full_arena(); outside = world(4, 0, 1); start_block = world(-4, 0, 1)
        seen.mark(start_block, GOLD, 1, 'ray')                      # never measured air: part of the world
        self.monitor.update(seen, {})
        seen.mark(outside, GOLD, 2, 'ray')
        self.assertEqual(self.monitor.update(seen, {}), {outside: ('stray', GOLD)})
        seen.mark(outside, 'dirt', 3, 'ray')                        # not one of the given blocks
        self.assertEqual(self.monitor.update(seen, {}), {})

    def test_a_cell_the_blueprint_keeps_empty_holding_a_block_is_an_anomaly(self):
        empty = world(0, 0, 3); monitor = RepairMonitor({}, [empty], {GOLD: 1}, check_kinds=False)
        seen = full_arena(); seen.mark(empty, GOLD, 1, 'voxel')
        self.assertEqual(monitor.update(seen, {}), {empty: ('must_be_empty', GOLD)})

    def test_a_ledger_shortfall_that_lasts_is_one_unaccounted_event_and_an_owed_pickup_excuses_it(self):
        seen = full_arena(); given = {GOLD: 4, IRON: 4, DIAMOND: 1, BEACON: 1}
        for _ in range(LEDGER_ROWS + 3): self.monitor.update(seen, dict(given, gold_block=3), owed={GOLD: 1})
        self.assertEqual(self.monitor.unaccounted, 0)
        for _ in range(LEDGER_ROWS + 3): self.monitor.update(seen, dict(given, gold_block=3))
        self.assertEqual(self.monitor.unaccounted, 1)
        self.assertEqual([e['blocks'] for e in self.monitor.pop_events() if e['event'] == 'unaccounted'], [[GOLD]])
        seen.mark(self.first, GOLD, 5, 'ray')                       # the missing block is seen: balanced again
        self.monitor.update(seen, dict(given, gold_block=3)); self.assertEqual(self.monitor.short_rows, 0)


class DigTests(unittest.TestCase):
    def setUp(self):
        self.target = world(0, 0, 3)
        self.feed = SensorRepairFeed(repair_fixture(ONE, SENSOR_DIG)); self.seen = full_arena()
        self.seen.mark(self.target, GOLD, 1, 'voxel')
        self.normal = (0, 0, -1)                     # the face toward the start (north for yaw 0)

    def step(self, f):
        self.feed.observe_frame(f, self.seen); return self.feed.decision_view()

    def test_an_anomaly_in_reach_with_a_seen_face_is_dig_and_the_teacher_attacks_only_on_a_strike(self):
        view = self.step(aimed(self.target, self.normal, inventory={PICKAXE: 1}))
        self.assertEqual((view.stage, view.target, view.strike), ('dig', self.target, True))
        self.assertEqual(self.feed.needed_kind(), PICKAXE)
        label = self.feed.label(None, view)
        self.assertEqual((label['attack'], list(label['camera'])), (1, [0., 0.]))
        view = self.step(frame(yaw=40., pitch=0.))                  # looking away: turn, no attack
        self.assertEqual((view.stage, view.strike), ('dig', False))
        self.assertEqual(self.feed.label(None, view)['attack'], 0)
        self.assertEqual(len(self.feed.features()), len(DIG_FEATURES))

    def test_far_from_the_anomaly_the_stage_is_reach(self):
        far = world(0, 0, 8); self.seen.mark(self.target, 'air', 2, 'ray'); self.seen.mark(far, IRON, 2, 'ray')
        self.feed = SensorRepairFeed(repair_fixture(dict(ONE, empty=[[0, 0, 8]], wrong=[[0, 0, 8, IRON]]), SENSOR_DIG))
        self.assertEqual(self.step(frame(z=0.5)).stage, 'reach')

    def test_after_the_block_turns_to_air_the_dig_collects_until_the_hand_count_is_back(self):
        self.step(aimed(self.target, self.normal, inventory={GOLD: 1, PICKAXE: 1}))
        self.seen.mark(self.target, 'air', 3, 'ray')
        view = self.step(aimed(self.target, self.normal, inventory={GOLD: 1, PICKAXE: 1}))
        self.assertEqual(view.stage, 'collect'); self.assertIsNotNone(view.stand)
        self.assertLessEqual(abs(view.stand[0] - 0.5 - self.target[0]) + abs(view.stand[2] - 0.5 - self.target[2]), 1.)
        self.assertIsNone(self.feed.needed_kind())                  # no tool needed to walk over the drop
        self.assertEqual(self.feed.label(None, view)['attack'], 0)
        view = self.step(frame(inventory={GOLD: 2, PICKAXE: 1}))
        self.assertEqual(view.stage, 'done')
        self.assertEqual((self.feed.dig.dug, self.feed.dig.collected, self.feed.dig.lost), (1, 1, 0))
        self.assertEqual([e['event'] for e in self.feed.pop_events()], ['anomaly', 'anomaly_cleared', 'dug', 'collected'])

    def test_the_collect_follows_the_seen_drop_to_its_cell_and_keeps_its_last_point(self):
        # Entity rays (2026-11-23): a dropped block can slide; the collect walks to where the rays saw it.
        self.step(aimed(self.target, self.normal, inventory={GOLD: 1, PICKAXE: 1}))
        self.seen.mark(self.target, 'air', 3, 'ray')
        moved = (self.target[0] + 1.6, self.target[1] + .1, self.target[2] + .5)      # one cell over
        f = sighted(frame(inventory={GOLD: 1}), ('item:gold_block', moved), ('cow', (9., 64., 9.)))
        centre = item_centre(moved, f.eye); cell = tuple(math.floor(v) for v in centre)
        view = self.step(f)
        self.assertEqual((view.stage, view.target, view.face['centre'], view.seen_item), ('collect', cell, centre, True))
        self.assertLessEqual(math.hypot(view.stand[0] - centre[0], view.stand[2] - centre[2]), COLLECT_POINT_REACH)
        view = self.step(frame(inventory={GOLD: 1}))                     # out of the rays: its last point stays
        self.assertEqual((view.target, view.seen_item), (cell, True))
        feats = self.feed.features()
        self.assertNotEqual(feats[4:7], [0., 0., 0.])                    # the face slot points at the drop
        self.feed.dig.owed = None

    def test_the_hit_point_on_the_items_near_face_is_moved_to_its_centre(self):
        # Smoke 2026-11-23 (1609): the ray met the item's near face at z = -496.01, a cell short of the item.
        eye = (-252.51, 71.62, -497.42)
        centre = item_centre((-252.54, 70.25, -496.01), eye)
        self.assertAlmostEqual(math.dist(centre, (-252.54, 70.25, -496.01)), ITEM_HALF)
        self.assertEqual(math.floor(centre[2]), -496)

    def test_an_arrived_collect_walks_on_toward_a_seen_drop_out_of_reach(self):
        self.step(aimed(self.target, self.normal, inventory={GOLD: 1}))
        self.seen.mark(self.target, 'air', 3, 'ray')
        drop = (self.target[0] + .5, self.target[1] + .05, self.target[2] + .9)
        view = self.step(sighted(frame(inventory={GOLD: 1}), ('item:gold_block', drop)))
        far = (view.stand[0], view.stand[2] - 1.)                          # stopped a block short of the standpoint
        view = self.step(frame(x=far[0], z=far[1], yaw=0., pitch=30., inventory={GOLD: 1}))
        view = replace(view, stand=(far[0], 64., far[1]))                   # as if it counted as arrived there
        label = dig_action(far[0], 64., far[1], 0., 30., view)
        self.assertEqual((label['forward'], label['attack']), (1, 0))

    def test_a_drop_of_another_block_or_seen_before_the_dig_is_not_followed(self):
        old = (self.target[0] + 2.5, self.target[1] + .1, self.target[2] + .5)
        self.step(sighted(aimed(self.target, self.normal, inventory={GOLD: 1}), ('item:gold_block', old)))
        self.seen.mark(self.target, 'air', 3, 'ray')
        view = self.step(sighted(frame(inventory={GOLD: 1}), ('item:iron_block', old)))
        self.assertEqual((view.target, view.seen_item, view.face), (self.target, False, None))

    def test_an_arrived_collect_that_sees_no_drop_looks_around(self):
        self.step(aimed(self.target, self.normal, inventory={GOLD: 1}))
        self.seen.mark(self.target, 'air', 3, 'ray')
        view = self.step(frame(inventory={GOLD: 1}))
        stand = view.stand
        view = self.step(frame(x=stand[0], z=stand[2], pitch=0., inventory={GOLD: 1}))
        label = self.feed.label(None, view)
        self.assertEqual((view.stage, label['forward'], label['attack']), ('collect', 0, 0))
        self.assertGreater(abs(label['camera'][1]), 0.)                     # turns to look around
        self.assertGreater(label['camera'][0], 0.)                          # and down toward the ground

    def test_a_drop_no_floor_cell_reaches_is_lost_after_the_collect_limit(self):
        high = world(0, 2, 3)
        self.feed = SensorRepairFeed(repair_fixture(dict(ONE, empty=[[0, 2, 3]], wrong=[[0, 2, 3, GOLD]]), SENSOR_DIG))
        self.seen.mark(self.target, 'air', 1, 'ray'); self.seen.mark(high, GOLD, 1, 'ray')
        self.step(aimed(high, self.normal, inventory={GOLD: 1}))
        self.seen.mark(high, 'air', 2, 'ray')
        views = [self.step(frame(inventory={GOLD: 1})) for _ in range(COLLECT_ROWS + 2)]
        self.assertEqual(views[0].stage, 'collect'); self.assertIsNone(views[0].stand)
        self.assertEqual(views[-1].stage, 'done'); self.assertEqual(self.feed.dig.lost, 1)


class CrosshairFeatureTests(unittest.TestCase):
    """The dig head's crosshair inputs (user decision 2026-11-24): d_strike and d_other after the d_ features."""

    def setUp(self):
        self.target = world(0, 0, 3); self.seen = full_arena(); self.seen.mark(self.target, GOLD, 1, 'voxel')
        self.feed = SensorRepairFeed(repair_fixture(ONE, SENSOR_DIG), dig_features=DIG2_FEATURES)
        self.old = SensorRepairFeed(repair_fixture(ONE, SENSOR_DIG))

    def values(self, f):
        self.feed.observe_frame(f, self.seen); self.old.observe_frame(f, self.seen)
        return self.feed.decision_view(), self.feed.features()

    def test_the_list_adds_two_crosshair_inputs_and_other_lists_are_refused(self):
        self.assertEqual(DIG2_FEATURES, DIG_FEATURES + ('d_strike', 'd_other'))
        with self.assertRaises(ValueError): SensorRepairFeed(repair_fixture(ONE, SENSOR_DIG), dig_features=('d_x',))
        with self.assertRaises(ValueError): self.old.use_dig_features(('d_x',))

    def test_a_strike_on_the_target_sets_d_strike_and_keeps_the_earlier_values(self):
        view, values = self.values(aimed(self.target, (0, 0, -1), inventory={PICKAXE: 1}))
        self.assertEqual((view.stage, view.strike, view.other), ('dig', True, False))
        self.assertEqual(len(values), len(DIG2_FEATURES)); self.assertEqual(values[-2:], [1., 0.])
        self.assertEqual(values[:-2], self.old.features())

    def test_a_crosshair_on_another_solid_sets_d_other_and_the_teacher_does_not_attack(self):
        other = world(1, 0, 3); self.seen.mark(other, 'stone', 1, 'voxel')
        view, values = self.values(aimed(other, (0, 0, -1), inventory={PICKAXE: 1}))
        self.assertEqual((view.stage, view.strike, view.other), ('dig', False, True))
        self.assertEqual(values[-2:], [0., 1.])
        self.assertEqual(self.feed.label(None, view)['attack'], 0)
        view, values = self.values(frame(yaw=0., pitch=-60., inventory={PICKAXE: 1}))    # the sky: neither
        self.assertEqual((view.stage, values[-2:]), ('dig', [0., 0.]))

    def test_outside_the_dig_stage_both_are_zero(self):
        far = world(0, 0, 8); self.seen.mark(self.target, 'air', 2, 'ray'); self.seen.mark(far, IRON, 2, 'ray')
        self.feed = SensorRepairFeed(repair_fixture(dict(ONE, empty=[[0, 0, 8]], wrong=[[0, 0, 8, IRON]]), SENSOR_DIG),
                                     dig_features=DIG2_FEATURES)
        self.old = SensorRepairFeed(repair_fixture(dict(ONE, empty=[[0, 0, 8]], wrong=[[0, 0, 8, IRON]]), SENSOR_DIG))
        view, values = self.values(frame(z=0.5, pitch=20.))                               # the floor in the crosshair
        self.assertEqual((view.stage, values[-2:]), ('reach', [0., 0.]))

    def test_the_building_head_gets_its_own_values_padded_to_the_dig_width(self):
        task = beacon_build(1501); seen = full_arena()
        feed = SensorRepairFeed(repair_fixture(task, SENSOR_REPAIR), dig_features=DIG2_FEATURES)
        old = SensorRepairFeed(repair_fixture(task, SENSOR_REPAIR))
        feed.observe_frame(frame(), seen); old.observe_frame(frame(), seen)
        self.assertEqual(feed.mode, 'build')
        self.assertEqual(feed.features(), old.features() + [0., 0.])


class CompositeTests(unittest.TestCase):
    def test_the_building_head_works_until_an_anomaly_then_the_dig_head_and_back(self):
        task = beacon_build(1501); feed = SensorRepairFeed(repair_fixture(task, SENSOR_REPAIR)); seen = full_arena()
        first = world(*task['cells'][0][:3])
        feed.observe_frame(frame(), seen)
        self.assertIn(feed.decision_view().stage, ('reach', 'move', 'aim', 'place'))
        self.assertEqual((feed.mode, feed.needed_kind()), ('build', GOLD))
        seen.mark(first, IRON, 2, 'ray')                            # iron in the first gold corner
        near = aimed(first, (0, 0, -1), x=first[0] + .5, z=first[2] - 1.5, inventory={IRON: 1})
        feed.observe_frame(near, seen)
        self.assertEqual(feed.decision_view().stage, 'dig_dig')
        self.assertEqual((feed.mode, feed.needed_kind()), ('dig', PICKAXE))
        self.assertIn(feed.decision_view().stage, COMBINED_STAGES)
        seen.mark(first, 'air', 3, 'ray'); feed.observe_frame(frame(inventory={IRON: 1}), seen)
        self.assertEqual(feed.decision_view().stage, 'dig_collect')
        feed.observe_frame(frame(inventory={IRON: 2}), seen)
        self.assertEqual(feed.mode, 'build'); self.assertEqual(feed.needed_kind(), GOLD)
        self.assertEqual([e['head'] for e in feed.pop_events() if e['event'] == 'head'], ['dig', 'build'])


class JudgeTests(unittest.TestCase):
    def test_the_demo_succeeds_when_every_cell_holds_its_block_and_no_stray_is_left(self):
        task = beacon_build(1501); judge = RepairJudge(repair_fixture(task, SENSOR_REPAIR))
        cells = [(world(*c[:3]), c[3]) for c in task['cells']]
        judge.observe(row(0)); n = 0
        judge.observe(row(1, place=(1, cells[0][0], IRON)))          # wrong kind in a gold cell
        judge.observe(row(2, place=(2, world(5, 0, 1), GOLD)))       # a stray outside
        self.assertEqual((judge.wrong(), len(judge.wrong_cells()), len(judge.strays())), (2, 1, 1))
        judge.observe(row(3, place=(2, world(5, 0, 1), GOLD), destroy=(1, cells[0][0], IRON)))
        judge.observe(row(4, place=(2, world(5, 0, 1), GOLD), destroy=(2, world(5, 0, 1), GOLD)))
        self.assertEqual((judge.dug, judge.broken, judge.strays(), judge.wrong_cells()), (2, 0, [], []))
        for i, (cell, block) in enumerate(cells):
            n = 3 + i; judge.observe(row(5 + i, place=(n, cell, block), destroy=(2, world(5, 0, 1), GOLD)))
            self.assertEqual(judge.done(), i == len(cells) - 1)
        self.assertEqual((judge.filled(), judge.wrong()), (10, 2))

    def test_a_dig_task_needs_the_wrong_blocks_gone_and_back_in_hand(self):
        task = dict(ONE, supports=[[1, 0, 4]], wrong=[[0, 0, 3, GOLD], [1, 1, 4, BEACON]], empty=[[0, 0, 3], [1, 1, 4]])
        judge = RepairJudge(repair_fixture(task, SENSOR_DIG)); gifts = {k: 1 for k in DIG_KINDS}
        judge.observe(row(0, inventory=gifts))
        self.assertEqual(len(judge.strays()), 2); self.assertFalse(judge.done())
        judge.observe(row(1, destroy=(1, world(0, 0, 3), GOLD), inventory=gifts))
        judge.observe(row(2, destroy=(2, world(1, 1, 4), BEACON), inventory=gifts))
        self.assertEqual((judge.strays(), judge.done()), ([], False))                  # not picked up yet
        judge.observe(row(3, destroy=(2, world(1, 1, 4), BEACON), inventory=dict(gifts, gold_block=2, beacon=2)))
        self.assertTrue(judge.done())
        judge.observe(row(4, destroy=(3, world(1, 0, 4), SUPPORT), inventory=dict(gifts, gold_block=2, beacon=2)))
        self.assertTrue(judge.done())                                # a support is part of the world, not a stray


if __name__ == '__main__':
    unittest.main()
