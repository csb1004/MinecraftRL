"""Pre-dispatch guards on the sensor path (sensor-tier design 2026-11-19 §2.5, §6 checks 2 and 8; plan 2026-11-22 task 4)."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minecraftrl.pipeline.guard import Guard
from minecraftrl.pipeline.kb import Evidence, KnowledgeBase
from minecraftrl.pipeline.seen_map import SeenMap
from minecraftrl.pipeline.sensor import sensor_frame
from minecraftrl.pipeline.sensor_guard import SensorGuards, apply_masks, planned_uses
from minecraftrl.pipeline.steps import Step
from test_pipeline_sensor import info

CTX = {'game_version': '1.16.5'}
WEB = [Evidence('web', 'support', 'https://minecraft.wiki')]
SRC = Path(__file__).resolve().parents[1] / 'src/minecraftrl/pipeline'
IRON = Step('iron', 'mine_iron', 'policy', 'iron_ore', 3, block='iron_ore')


def frame(t=0., workshop=None, **sensor):
    value = info(**sensor)
    value['workshop'].update(workshop or {})
    return sensor_frame(value, t)


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.kb = KnowledgeBase(Path(self.temp.name) / 'kb.sqlite')
        self.kb.add_claim('iron_ore', 'harvest_requires', dict(tool_class='pickaxe', min_tier='stone'), {}, WEB, run_id='t')
        self.guard = Guard(self.kb, CTX, frozenset({'iron_ore'}))

    def tearDown(self):
        self.kb.close(); self.temp.cleanup()

    def guards(self, **kw):
        kw.setdefault('needs', frozenset({'iron_ore', 'stone_pickaxe'}))
        return SensorGuards(self.guard, kb=self.kb, context=CTX, **kw)


class AimGuardTests(Base):
    def crosshair(self, kind, held, **kw):
        cell = [0, 65, 3]
        return frame(workshop=dict(server_selected_item=held), sensor_ray_kind=['stone'] * 135 + [kind],
                     sensor_ray_x=[0] * 136, sensor_ray_y=[65] * 136, sensor_ray_z=[3] * 136, **kw), cell

    def test_crosshair_on_a_guarded_block_with_a_weak_tool_masks_attack(self):
        f, _ = self.crosshair('iron_ore', 'wooden_pickaxe')
        event, = self.guards().pre_dispatch(f, SeenMap(), {'attack': 1})
        self.assertEqual((event['kind'], event['phase'], event['mask'], event['hold']), ('guard_violation', 'crosshair', ['attack'], False))
        self.assertEqual(self.guards().pre_dispatch(self.crosshair('iron_ore', 'stone_pickaxe')[0], SeenMap(), {'attack': 1}), [])

    def test_camera_turn_toward_a_guarded_block_is_predicted_on_the_seen_map(self):
        f = frame(workshop=dict(server_selected_item='wooden_pickaxe'))   # crosshair on stone (not guarded), facing +z
        seen = SeenMap()
        for x in (0, -1): seen.mark((x, 65, 0), 'air', 1, 'ray')
        seen.mark((-2, 65, 0), 'iron_ore', 1, 'ray')
        turn = {'attack': 1, 'camera': np.array([0., 90.])}             # yaw +90: facing -x
        event, = self.guards().pre_dispatch(f, seen, turn)
        self.assertEqual((event['kind'], event['phase'], event['pos'], event['mask']), ('guard_violation', 'ray_prediction', (-2, 65, 0), ['attack']))
        self.assertEqual(self.guards().pre_dispatch(f, seen, {'attack': 1}), [])   # no turn: the crosshair (stone) decides

    def test_unknown_or_other_aim_blocks_attack_when_the_tool_is_short_for_any_guarded_block(self):
        f = frame(workshop=dict(server_selected_item='air'))
        event, = self.guards().pre_dispatch(f, SeenMap(), {'attack': 1, 'camera': np.array([0., 90.])})  # turn into the unseen
        self.assertEqual((event['kind'], event['block'], event['mask'], event['hold']), ('guard_aim_uncertain', 'unknown', ['attack'], False))
        other, _ = self.crosshair('other', 'air')
        self.assertEqual(self.guards().pre_dispatch(other, SeenMap(), {'attack': 1})[0]['kind'], 'guard_aim_uncertain')
        safe, _ = self.crosshair('other', 'stone_pickaxe')
        self.assertEqual(self.guards().pre_dispatch(safe, SeenMap(), {'attack': 1}), [])

    def test_no_attack_no_aim_event(self):
        f, _ = self.crosshair('iron_ore', 'air')
        self.assertEqual(self.guards().pre_dispatch(f, SeenMap(), {'use': 1}), [])

    def test_server_only_values_do_not_change_the_decision(self):
        # Leak check 8 (design §6): server harvestability and the server mining target never reach the guard.
        base = info(sensor_ray_kind=['stone'] * 135 + ['iron_ore']); base['workshop']['server_selected_item'] = 'wooden_pickaxe'
        changed = copy.deepcopy(base)
        changed['demo'].update(demo_mining_block='air', demo_mining_x=-999, demo_last_destroy_harvestable=0, demo_hit_block='dirt')
        a = self.guards().pre_dispatch(sensor_frame(base, 0.), SeenMap(), {'attack': 1})
        b = self.guards().pre_dispatch(sensor_frame(changed, 0.), SeenMap(), {'attack': 1})
        self.assertEqual(a, b); self.assertEqual(a[0]['kind'], 'guard_violation')


class DropGuardTests(Base):
    def test_dropping_a_needed_item_is_masked(self):
        event, = self.guards().pre_dispatch(frame(), SeenMap(), {'drop': 1})      # holds the stone pickaxe
        self.assertEqual((event['kind'], event['reason'], event['mask'], event['hold']), ('guard_drop_needed', 'needed_item', ['drop'], False))

    def test_dropping_an_unneeded_item_passes_unless_a_hotbar_key_is_pressed(self):
        dirt = frame(workshop=dict(server_selected_item='dirt'))
        self.assertEqual(self.guards().pre_dispatch(dirt, SeenMap(), {'drop': 1}), [])
        event, = self.guards().pre_dispatch(dirt, SeenMap(), {'drop': 1, 'hotbar.2': 1})
        self.assertEqual((event['reason'], event['mask']), ('hotbar_with_drop', ['drop']))

    def test_a_replanned_step_uses_the_new_needs(self):
        guards = self.guards(); dirt = frame(workshop=dict(server_selected_item='dirt'))
        self.assertEqual(guards.pre_dispatch(dirt, SeenMap(), {'drop': 1}), [])
        guards.start_step(needs={'dirt'})
        self.assertEqual(guards.pre_dispatch(dirt, SeenMap(), {'drop': 1})[0]['reason'], 'needed_item')

    def test_repeated_drops_hold(self):
        guards = self.guards(hold_s=1.)
        kinds = [guards.pre_dispatch(frame(t=0.1 * i), SeenMap(), {'drop': 1})[0] for i in range(12)]
        self.assertEqual((kinds[0]['kind'], kinds[-1]['kind'], kinds[-1]['hold']), ('guard_drop_needed', 'guard_drop_needed_persistent', True))


class WearGuardTests(Base):
    def test_too_few_uses_left_for_the_plan_masks_attack_and_holds(self):
        guards = self.guards(steps=[IRON]); worn = frame(sensor_tool_damage=129)        # 131 - 129 = 2 < 3 + 2
        event, = guards.pre_dispatch(worn, SeenMap(), {'attack': 1})
        self.assertEqual((event['kind'], event['hold'], event['mask'], event['remaining'], event['planned'], event['margin'],
                          event['correction']), ('guard_tool_wear', True, ['attack'], 2, 3, 2, 'craft_spare_tool'))
        self.assertEqual(guards.pre_dispatch(frame(sensor_tool_damage=3), SeenMap(), {'attack': 1}), [])
        self.assertEqual(guards.pre_dispatch(worn, SeenMap(), {'use': 1}), [])        # no attack: no wear check

    def test_planned_uses_count_steps_the_tool_can_do(self):
        dirt = Step('dirt', 'mine_dirt', 'policy', 'dirt', 5, block='dirt')
        self.assertEqual(planned_uses([IRON, dirt], self.kb, CTX, 'stone_pickaxe'), 3)
        self.assertEqual(planned_uses([IRON], self.kb, CTX, 'wooden_pickaxe'), 0)    # cannot do the iron step
        self.assertEqual(planned_uses([IRON], self.kb, CTX, 'stone_axe'), 0)


class HazardGuardTests(Base):
    FWD = {'forward': 1, 'sprint': 1, 'jump': 1}

    def seen(self, column, floor_y=None, landing='stone', ahead='air'):
        """A seen map with solid ground under the start (0, 63, 0) and behind it, and one column ahead at x=0, z=1."""
        seen = SeenMap()
        for z in (-1, 0): seen.mark((0, 63, z), 'stone', 1, 'voxel'); seen.mark((0, 64, z), 'air', 1, 'voxel')
        seen.mark((0, 64, 1), ahead, 1, 'voxel')
        for y in column: seen.mark((0, y, 1), 'air', 1, 'ray')
        if floor_y is not None: seen.mark((0, floor_y, 1), landing, 1, 'ray')
        return seen

    def test_damaging_fall_is_masked_outside_bridge_and_jump(self):
        seen = self.seen(range(60, 64), floor_y=59)                    # fall 4: 1 damage
        event, = self.guards().pre_dispatch(frame(), seen, self.FWD)
        self.assertEqual((event['kind'], event['reason'], event['key'], event['cell']), ('guard_hazard_move', 'damaging_fall', 'forward', (0, 64, 1)))
        self.assertEqual(sorted(event['mask']), ['forward', 'jump', 'sprint'])
        self.assertEqual(self.guards(skill_kind='bridge').pre_dispatch(frame(), seen, self.FWD), [])

    def test_lethal_unknown_and_lava_are_masked_for_every_skill(self):
        weak = frame(workshop=dict(server_health=7.))
        lethal = self.guards(skill_kind='bridge').pre_dispatch(weak, self.seen(range(59, 64), floor_y=58), self.FWD)
        self.assertEqual(lethal[0]['reason'], 'lethal_fall')           # fall 5: 2 damage, 7 -> 5 <= 6
        self.assertEqual(self.guards(skill_kind='jump').pre_dispatch(frame(), self.seen(range(60, 64)), self.FWD)[0]['reason'], 'unknown_floor')
        self.assertEqual(self.guards(skill_kind='jump').pre_dispatch(frame(), self.seen((), ahead='lava'), self.FWD)[0]['reason'], 'lava_or_fire')
        self.assertEqual(self.guards().pre_dispatch(frame(), self.seen(range(55, 64)), self.FWD)[0]['reason'], 'unknown_floor')

    def test_water_landing_and_safe_directions_pass(self):
        self.assertEqual(self.guards().pre_dispatch(frame(), self.seen(range(58, 64), floor_y=57, landing='water'), self.FWD), [])
        self.assertEqual(self.guards().pre_dispatch(frame(), self.seen(range(60, 64), floor_y=59), {'back': 1}), [])

    def test_airborne_or_still_is_not_judged(self):
        seen = self.seen(range(60, 64))
        self.assertEqual(self.guards().pre_dispatch(frame(workshop=dict(client_on_ground=0)), seen, self.FWD), [])
        self.assertEqual(self.guards().pre_dispatch(frame(), seen, {'jump': 1}), [])


class MaskAndImportTests(unittest.TestCase):
    def test_apply_masks_zeroes_only_the_named_keys(self):
        action = {'attack': np.array(1), 'forward': np.array(1), 'camera': np.array([1., 2.])}
        out, masked = apply_masks(action, [dict(mask=['attack']), dict(mask=['forward', 'attack'])])
        self.assertEqual((int(out['attack']), int(out['forward']), masked), (0, 0, ['attack', 'forward']))
        self.assertEqual(int(action['attack']), 1)                     # the policy's own action is not changed in place
        self.assertTrue(np.array_equal(out['camera'], [1., 2.]))

    def test_sensor_path_modules_never_import_fixture_or_server_record_code(self):
        # Leak check 2 (design §6).
        forbidden = ('fixture', 'stage_replay', 'teacher', 'place_task', 'build_task', 'bridge_task', 'jump_course',
                     'reacquire', 'monitor_replay', 'workshop_monitor')
        for name in ('sensor.py', 'seen_map.py', 'sensor_guard.py'):
            tree = ast.parse((SRC / name).read_text(encoding='utf-8'))
            modules = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            modules += [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            for module in modules:
                with self.subTest(file=name, module=module):
                    self.assertFalse(any(word in module for word in forbidden))


if __name__ == '__main__':
    unittest.main()
