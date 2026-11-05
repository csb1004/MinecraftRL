import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from minecraftrl.pipeline.guard import Guard
from minecraftrl.pipeline.kb import Evidence, KnowledgeBase
from minecraftrl.pipeline.rm import Sample

CTX = {'game_version': '1.16.5'}
WEB = [Evidence('web', 'support', 'https://minecraft.wiki')]
BASE = Sample(t=0., tick=1000, valid=True, gui_open=False, position=(0.5, 64., 0.5), yaw=0., pitch=0., action_keys=(),
              inventory={}, selected_item='air', destroys=0, last_destroyed='air', last_destroyed_pos=None,
              last_harvestable=False, mining_block='air', mining_pos=None, hit_block='iron_ore', hit_pos=(0, 65, 3), health=20.)
ATTACK = {'attack': 1}


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.kb = KnowledgeBase(Path(self.temp.name) / 'kb.sqlite')

    def tearDown(self):
        self.kb.close(); self.temp.cleanup()

    def iron_requires_stone(self):
        self.kb.add_claim('iron_ore', 'harvest_requires', dict(tool_class='pickaxe', min_tier='stone'), {}, WEB, run_id='t')

    def test_unknown_holds_before_dispatch(self):
        event = Guard(self.kb, CTX, frozenset({'iron_ore'})).pre_dispatch(BASE, ATTACK)
        self.assertEqual((event['kind'], event['hold'], event['phase'], event['mining_ticks']), ('guard_unknown', True, 'pre_dispatch', 0))

    def test_violation_masks_attack_without_hold(self):
        self.iron_requires_stone()
        event = Guard(self.kb, CTX, frozenset({'iron_ore'})).pre_dispatch(replace(BASE, selected_item='wooden_pickaxe'), ATTACK)
        self.assertEqual((event['kind'], event['mask_attack'], event['hold']), ('guard_violation', True, False))
        self.assertEqual(event['required'], dict(tool_class='pickaxe', min_tier='stone'))

    def test_ok_and_unguarded_pass(self):
        self.iron_requires_stone(); guard = Guard(self.kb, CTX, frozenset({'iron_ore'}))
        self.assertIsNone(guard.pre_dispatch(replace(BASE, selected_item='stone_pickaxe'), ATTACK))
        self.assertIsNone(guard.pre_dispatch(replace(BASE, hit_block='dirt'), ATTACK))

    def test_goal_without_iron_never_guards_iron(self):
        # Same hand hit on iron ore, but the goal (RM) does not need iron: not an anomaly.
        self.assertIsNone(Guard(self.kb, CTX, frozenset({'log'})).pre_dispatch(BASE, ATTACK))

    def test_no_attack_no_event(self):
        self.assertIsNone(Guard(self.kb, CTX, frozenset({'iron_ore'})).pre_dispatch(BASE, {}))

    def test_server_mining_backup_catches_aim_mismatch(self):
        guard = Guard(self.kb, CTX, frozenset({'iron_ore'}))
        mining = replace(BASE, hit_block='stone', mining_block='iron_ore', mining_pos=(0, 65, 3))
        event = guard.observe(mining)  # first server report already holds: the last reversible moment
        self.assertEqual((event['kind'], event['phase'], event['hold'], event['mining_ticks']), ('guard_unknown', 'server_mining', True, 0))
        self.assertEqual(guard.observe(replace(mining, tick=1006))['mining_ticks'], 6)
        self.assertIsNone(guard.observe(replace(mining, tick=1007, mining_block='air', mining_pos=None)))

    def test_disputed_only_is_unknown(self):
        self.iron_requires_stone(); claim, = self.kb.query('iron_ore', 'harvest_requires', context=CTX)
        self.kb.set_status(claim.id, 'disputed', 't', run_id='t')
        self.assertEqual(Guard(self.kb, CTX, frozenset({'iron_ore'})).pre_dispatch(BASE, ATTACK)['kind'], 'guard_unknown')

    def test_persistent_violation_holds(self):
        self.iron_requires_stone(); guard = Guard(self.kb, CTX, frozenset({'iron_ore'}), violation_hold_s=10.)
        held = replace(BASE, selected_item='wooden_pickaxe')
        events = [guard.pre_dispatch(replace(held, t=i * .5), ATTACK) for i in range(22)]  # continuous for 10.5 s
        self.assertFalse(events[0]['hold'])
        self.assertEqual((events[-1]['kind'], events[-1]['hold']), ('guard_violation_persistent', True))

    def test_violation_with_a_gap_is_not_persistent(self):
        # Real run A3 (2026-11-04): the timer survived while the policy looked elsewhere, so a brief return to the
        # iron ore 10 s after the first mask held at once. Persistence means continuous fixation.
        self.iron_requires_stone(); guard = Guard(self.kb, CTX, frozenset({'iron_ore'}), violation_hold_s=10.)
        held = replace(BASE, selected_item='wooden_pickaxe')
        guard.pre_dispatch(held, ATTACK)
        self.assertIsNone(guard.pre_dispatch(replace(held, t=5., hit_block='grass_block'), ATTACK))
        event = guard.pre_dispatch(replace(held, t=10.5), ATTACK)
        self.assertEqual((event['kind'], event['hold']), ('guard_violation', False))

    def test_gui_open_and_invalid_sample_ignored(self):
        guard = Guard(self.kb, CTX, frozenset({'iron_ore'}))
        self.assertIsNone(guard.pre_dispatch(replace(BASE, gui_open=True), ATTACK))
        self.assertIsNone(guard.pre_dispatch(replace(BASE, valid=False), ATTACK))

    def test_server_harvestability_never_changes_verdict(self):
        guard = Guard(self.kb, CTX, frozenset({'iron_ore'}))
        a = guard.pre_dispatch(replace(BASE, last_harvestable=True), ATTACK); b = guard.pre_dispatch(BASE, ATTACK)
        self.assertEqual(a, b)

    def test_set_guarded_after_replan(self):
        self.kb.add_claim('log', 'harvest_requires', dict(tool_class=None, min_tier='hand'), {}, WEB, run_id='t')
        guard = Guard(self.kb, CTX, frozenset({'iron_ore'})); guard.set_guarded(frozenset({'log', 'stone', 'iron_ore'}))
        self.assertIsNone(guard.pre_dispatch(replace(BASE, hit_block='oak_log'), ATTACK))


if __name__ == '__main__':
    unittest.main()
