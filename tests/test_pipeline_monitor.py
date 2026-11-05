import unittest
from dataclasses import replace

from minecraftrl.pipeline.monitor import HOLD_KINDS, FastMonitor, inefficient
from minecraftrl.pipeline.planner import Plan
from minecraftrl.pipeline.rm import Sample, StageMachine, sample_from
from minecraftrl.pipeline.steps import Step

BASE = Sample(t=0., tick=1000, valid=True, gui_open=False, position=(0.5, 64., 0.5), yaw=0., pitch=0., action_keys=(),
              inventory={}, selected_item='air', destroys=0, last_destroyed='air', last_destroyed_pos=None,
              last_harvestable=False, mining_block='air', mining_pos=None, hit_block='air', hit_pos=None, health=20.)
IRON = Step('iron', 'mine_iron_ore', 'policy', 'iron_ore', 1, block='iron_ore')
CLIMB = Step('escape', 'climb_out', 'policy', 'height', 6, params={'expect': 'rise'})


def destroyed(s, block, harvestable, tick):
    return replace(s, tick=tick, destroys=s.destroys + 1, last_destroyed=block, last_destroyed_pos=(0, 65, 3),
                   last_harvestable=harvestable)


class MonitorTests(unittest.TestCase):
    def run_samples(self, monitor, samples):
        return [e for e in (monitor.update(s) for s in samples) if e]

    def test_destroy_not_harvestable_is_post_hoc_hold(self):
        m = FastMonitor(IRON, 'r1', frozenset({'iron_ore'})); m.update(BASE)
        event = m.update(destroyed(BASE, 'iron_ore', False, 1001))
        self.assertEqual((event['kind'], event['hold']), ('destroy_not_harvestable', True))

    def test_unguarded_destroy_ignored(self):
        m = FastMonitor(IRON, 'r1', frozenset({'iron_ore'})); m.update(BASE)
        self.assertIsNone(m.update(destroyed(BASE, 'dirt', False, 1001)))

    def test_goal_decides_whether_iron_loss_matters(self):
        logs = Step('logs', 'collect_log', 'policy', 'log', 3, block='log')
        m = FastMonitor(logs, 'r1', frozenset({'log'})); m.update(BASE)
        self.assertIsNone(m.update(destroyed(BASE, 'iron_ore', False, 1001)))

    def test_harvest_without_drop_is_recorded_not_held(self):
        m = FastMonitor(IRON, 'r1', frozenset({'iron_ore'}), drop_window_ticks=20); m.update(BASE)
        s = destroyed(BASE, 'iron_ore', True, 1001); m.update(s)
        event = m.update(replace(s, tick=1022))
        self.assertEqual((event['kind'], event['hold']), ('harvest_without_drop', False))

    def test_normal_harvest_no_event(self):
        m = FastMonitor(IRON, 'r1', frozenset({'iron_ore'})); m.update(BASE)
        s = destroyed(BASE, 'iron_ore', True, 1001); m.update(s)
        self.assertEqual(self.run_samples(m, [replace(s, tick=1005 + i, inventory={'iron_ore': 1}) for i in range(30)]), [])

    def test_needed_item_lost_is_rm_regression(self):
        m = FastMonitor(IRON, 'r1', frozenset({'iron_ore'}), needs=frozenset({'stone_pickaxe', 'iron_ore'}))
        m.update(replace(BASE, inventory={'stone_pickaxe': 1}))
        event = m.update(replace(BASE, tick=1001, inventory={}))
        self.assertEqual((event['kind'], event['hold'], event['evidence']['item']), ('needed_tool_lost', True, 'stone_pickaxe'))
        m2 = FastMonitor(IRON, 'r1', frozenset({'iron_ore'}), needs=frozenset({'iron_ore'}))
        m2.update(replace(BASE, inventory={'stick': 2}))
        self.assertIsNone(m2.update(replace(BASE, tick=1001, inventory={})))

    def test_items_moved_inside_an_open_gui_are_not_lost(self):
        # Real check P3: the wooden pickaxe sat in the crafting grid (not in the server inventory) while the GUI was open.
        m = FastMonitor(IRON, 'r1', frozenset({'iron_ore'}), needs=frozenset({'wooden_pickaxe', 'iron_ore'}))
        m.update(replace(BASE, inventory={'wooden_pickaxe': 1}))
        self.assertIsNone(m.update(replace(BASE, tick=1001, inventory={}, gui_open=True)))
        self.assertIsNone(m.update(replace(BASE, tick=1002, inventory={'wooden_pickaxe': 1})))

    def test_placing_a_needed_block_is_recorded_not_held(self):
        # Replayed log trial: the policy placed one of its logs; the block can be mined back, so it is not irreversible.
        logs = Step('logs', 'collect_log', 'policy', 'log', 3, block='log')
        m = FastMonitor(logs, 'r1', frozenset({'log'}), needs=frozenset({'log'})); m.update(replace(BASE, inventory={'oak_log': 2}))
        event = m.update(replace(BASE, tick=1001, inventory={'oak_log': 1}))
        self.assertEqual((event['kind'], event['hold']), ('needed_item_placed', False))

    def test_blocked_overhead_only_while_climbing(self):
        # The ceiling probe held 16 dirt: the block is in hand, the head hits the ceiling.
        jumping = [replace(BASE, tick=1000 + i, t=i * .05, action_keys=('jump',), position=(0.5, 64. + (i % 2) * .5, 0.5),
                           selected_item='dirt') for i in range(40)]
        events = self.run_samples(FastMonitor(CLIMB, 'r1', frozenset()), jumping)
        self.assertEqual(events[0]['kind'], 'blocked_overhead'); self.assertTrue(events[0]['hold'])
        self.assertAlmostEqual(events[0]['evidence']['remaining'], 6. - (jumping[events[0]['tick'] - 1000].position[1] - 64.))
        self.assertLessEqual(events[0]['tick'] - 1000, 25)
        self.assertEqual(self.run_samples(FastMonitor(IRON, 'r1', frozenset({'iron_ore'})), jumping), [])

    def test_rising_pillar_is_not_blocked(self):
        rising = [replace(BASE, tick=1000 + i, action_keys=('jump', 'use'), position=(0.5, 64. + i / 8, 0.5), selected_item='dirt')
                  for i in range(48)]
        self.assertEqual(self.run_samples(FastMonitor(CLIMB, 'r1', frozenset()), rising), [])

    def test_climbing_without_a_block_in_hand_holds(self):
        # Pillar segment: the equipped stack ran out with more than one block of height still to go.
        climb = Step('climb-1', 'escape_pit', 'policy', 'rise', 3, params={'until': 'rise', 'expect': 'rise'})
        m = FastMonitor(climb, 'r1', frozenset())
        self.assertIsNone(m.update(replace(BASE, selected_item='dirt', position=(0.5, 65., 0.5))))
        self.assertIsNone(m.update(replace(BASE, tick=1001, selected_item='dirt', position=(0.5, 66., 0.5))))
        event = m.update(replace(BASE, tick=1002, selected_item='air', position=(0.5, 66., 0.5)))
        self.assertEqual((event['kind'], event['hold'], event['evidence']['remaining']), ('climb_without_block', True, 2.))

    def test_empty_hand_within_a_block_of_the_target_is_not_blocked(self):
        climb = Step('climb-2', 'escape_pit', 'policy', 'rise', 3, params={'until': 'rise', 'expect': 'rise'})
        m = FastMonitor(climb, 'r1', frozenset()); m.update(replace(BASE, selected_item='oak_planks', position=(0.5, 65., 0.5)))
        self.assertIsNone(m.update(replace(BASE, tick=1001, selected_item='air', position=(0.5, 67.2, 0.5))))
        # Not a climb step: an empty hand means nothing.
        other = FastMonitor(IRON, 'r1', frozenset({'iron_ore'})); other.update(BASE)
        self.assertIsNone(other.update(replace(BASE, tick=1001)))

    def test_low_health_is_always_an_anomaly(self):
        m = FastMonitor(IRON, 'r1', frozenset({'iron_ore'})); m.update(BASE)
        self.assertEqual(m.update(replace(BASE, tick=1001, health=4.))['kind'], 'low_health')

    def test_invalid_sample_never_fires(self):
        m = FastMonitor(IRON, 'r1', frozenset({'iron_ore'})); m.update(BASE)
        self.assertIsNone(m.update(replace(destroyed(BASE, 'iron_ore', False, 1001), valid=False)))

    def test_stationary_repetition_holds_but_slow_progress_does_not(self):
        same = [replace(BASE, tick=1000 + i, action_keys=('attack',)) for i in range(420)]
        events = self.run_samples(FastMonitor(IRON, 'r1', frozenset({'iron_ore'})), same)
        self.assertEqual((events[0]['kind'], events[0]['hold']), ('stationary_repetition', True))
        wander = [replace(BASE, tick=1000 + i, action_keys=('forward',), position=(0.5 + (i % 50) / 5, 64., 0.5 + (i // 50)))
                  for i in range(900)]
        self.assertTrue(all(not e['hold'] for e in self.run_samples(FastMonitor(IRON, 'r1', frozenset({'iron_ore'})), wander)))

    def test_inefficiency_never_holds(self):
        self.assertTrue(inefficient(130, 60)); self.assertFalse(inefficient(100, None))
        self.assertNotIn('inefficiency', HOLD_KINDS)


class RewardMachineTests(unittest.TestCase):
    def plan(self, steps): return Plan('p', steps, False, [], frozenset(), frozenset())

    def test_stage_machine_policy_gain_and_rule_possession(self):
        steps = [Step('logs', 'collect_log', 'policy', 'log', 3, block='log'),
                 Step('craft', 'craft', 'rule', 'wooden_pickaxe', 1, params={'goals': ['wooden_pickaxe'], 'equip': 'wooden_pickaxe'})]
        sm = StageMachine(self.plan(steps), {'oak_log': 1}, goal_item='wooden_pickaxe')
        self.assertEqual(sm.observe({'oak_log': 3}, 'air'), [])
        self.assertEqual(sm.observe({'oak_log': 2, 'birch_log': 2}, 'air')[0]['step_id'], 'logs')
        self.assertEqual(sm.observe({'wooden_pickaxe': 1}, 'air'), [])
        self.assertEqual(sm.observe({'wooden_pickaxe': 1}, 'wooden_pickaxe')[0]['step_id'], 'craft'); self.assertTrue(sm.done)

    def test_stage_machine_replace_keeps_completed_history(self):
        sm = StageMachine(self.plan([Step('logs', 'collect_log', 'policy', 'log', 1, block='log')]), {})
        sm.observe({'oak_log': 1}, 'air'); sm.replace(self.plan([IRON]), {'oak_log': 1})
        self.assertEqual([h['step_id'] for h in sm.history], ['logs']); self.assertEqual(sm.current.id, 'iron')

    def test_root_goal_done_even_mid_plan(self):
        sm = StageMachine(self.plan([Step('logs', 'collect_log', 'policy', 'log', 4, block='log'), IRON]), {})
        sm.observe({'iron_ore': 1}, 'air'); self.assertTrue(sm.done)

    def test_sample_from_info(self):
        info = {'workshop': dict(server_available=1, error=0, screen=0, world_time=5, server_x=1., server_y=2., server_z=3.,
                                 yaw=0., pitch=10., server_inventory={'dirt': 2, 'stick': 0}, server_selected_item='dirt',
                                 server_health=20.),
                'demo': dict(demo_available=1, demo_destroys=1, demo_last_destroyed_block='iron_ore', demo_last_destroy_x=0,
                             demo_last_destroy_y=1, demo_last_destroy_z=2, demo_last_destroy_harvestable=0,
                             demo_mining_block='air', demo_hit_block='stone')}
        s = sample_from(info, {'attack': 1, 'jump': 0, 'camera': [0., 1.]}, 7.)
        self.assertEqual((s.tick, s.action_keys, s.inventory, s.last_destroyed_pos, s.mining_pos, s.hit_block),
                         (5, ('attack',), {'dirt': 2}, (0, 1, 2), None, 'stone'))


if __name__ == '__main__':
    unittest.main()
