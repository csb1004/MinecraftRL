import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from minecraftrl.pipeline.rm import Sample
from minecraftrl.pipeline.stage_replay import EYE as EYE_HEIGHT, StageView
from minecraftrl.reacquire_teacher import BUTTONS, aim_delta, in_view, policy_label, teacher_env_action

EYE = (0.5, 1.5, 0.5)
STEVE1 = importlib.util.find_spec('torch') is not None and (Path(__file__).resolve().parents[1] / 'external/steve1').exists()


def sample(*, hit='air', hit_pos=None, mining='air', mining_pos=None, gui=False, valid=True, yaw=0., pitch=0.):
    return Sample(t=1., tick=20, valid=valid, gui_open=gui, position=(EYE[0], EYE[1] - EYE_HEIGHT, EYE[2]), yaw=yaw,
                  pitch=pitch, action_keys=(), inventory={}, selected_item='stone_pickaxe', destroys=0,
                  last_destroyed='air', last_destroyed_pos=None, last_harvestable=False, mining_block=mining,
                  mining_pos=mining_pos, hit_block=hit, hit_pos=hit_pos)


def official_mapping():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'external/steve1'))
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.actions import ActionTransformer
    return CameraHierarchicalMapping(n_camera_bins=11), ActionTransformer(**ACTION_TRANSFORMER_KWARGS)


class AimDeltaTest(unittest.TestCase):
    def test_aim_delta_straight_ahead_is_zero(self):
        self.assertEqual(tuple(round(v, 6) for v in aim_delta(EYE, 0., 0., (0, 1, 3))), (0., 0.))

    def test_aim_delta_target_to_minus_x_turns_right_90(self):
        self.assertAlmostEqual(aim_delta(EYE, 0., 0., (-3, 1, 0))[1], 90., places=6)

    def test_aim_delta_wraps_yaw(self):
        self.assertAlmostEqual(aim_delta(EYE, 350., 0., (0, 1, 3))[1], 10., places=6)

    def test_aim_delta_behind_is_plus_180(self):
        self.assertAlmostEqual(aim_delta(EYE, 0., 0., (0, 1, -3))[1], 180., places=6)

    def test_aim_delta_block_below_looks_down(self):
        self.assertAlmostEqual(aim_delta(EYE, 0., 0., (0, 0, 1))[0], 45., places=6)

    def test_in_view_uses_screen_half_angles(self):
        self.assertTrue(in_view(30., 45.)); self.assertFalse(in_view(40., 0.)); self.assertFalse(in_view(0., 60.))


class TeacherActionTest(unittest.TestCase):
    def test_near_not_aimed_turns_without_attack(self):
        a = teacher_env_action(sample(hit='grass_block'), StageView('near', 3., (0, 1, -3), EYE), 'iron_ore')
        self.assertEqual(list(a['camera']), [0., 10.]); self.assertEqual(a['attack'], 0)
        self.assertEqual(sum(a[k] for k in ('forward', 'back', 'left', 'right', 'jump')), 0)

    def test_aimed_attacks_and_keeps_aim_on_the_hit_block(self):
        a = teacher_env_action(sample(hit='iron_ore', hit_pos=(0, 1, 3)), StageView('aimed', 3., (1, 1, 3), EYE), 'iron_ore')
        self.assertEqual(a['attack'], 1); self.assertEqual([round(v, 6) for v in a['camera']], [0., 0.])

    def test_mining_keeps_aim_on_the_mined_block(self):
        a = teacher_env_action(sample(hit='iron_ore', hit_pos=(0, 1, 3), mining='iron_ore', mining_pos=(0, 1, 3)),
                               StageView('mining', 3., (1, 1, 3), EYE), 'iron_ore')
        self.assertEqual(a['attack'], 1); self.assertEqual([round(v, 6) for v in a['camera']], [0., 0.])

    def test_hold_teacher_keeps_still_and_attacks_while_aimed(self):
        s = sample(hit='iron_ore', hit_pos=(1, 1, 3))          # on the block but off its centre
        view = StageView('aimed', 3., (1, 1, 3), EYE)
        centre = teacher_env_action(s, view, 'iron_ore')
        self.assertNotEqual(round(float(centre['camera'][1]), 6), 0.)
        hold = teacher_env_action(s, view, 'iron_ore', aim='hold')
        self.assertEqual(list(hold['camera']), [0., 0.]); self.assertEqual(hold['attack'], 1)
        mining = teacher_env_action(sample(hit='iron_ore', hit_pos=(1, 1, 3), mining='iron_ore', mining_pos=(1, 1, 3)),
                                    StageView('mining', 3., (1, 1, 3), EYE), 'iron_ore', aim='hold')
        self.assertEqual(list(mining['camera']), [0., 0.]); self.assertEqual(mining['attack'], 1)

    def test_hold_teacher_still_turns_when_near(self):
        view = StageView('near', 3., (0, 1, -3), EYE)
        self.assertEqual(list(teacher_env_action(sample(hit='grass_block'), view, 'iron_ore', aim='hold')['camera']),
                         [0., 10.])
        with self.assertRaises(ValueError): teacher_env_action(sample(), view, 'iron_ore', aim='other')

    def test_far_gui_and_invalid_rows_have_no_label(self):
        self.assertIsNone(teacher_env_action(sample(), StageView('far', 9., (0, 1, 9), EYE), 'iron_ore'))
        self.assertIsNone(teacher_env_action(sample(gui=True), StageView('near', 3., (0, 1, 3), EYE), 'iron_ore'))
        self.assertIsNone(teacher_env_action(sample(valid=False), StageView('near', 3., (0, 1, 3), EYE), 'iron_ore'))

    def test_every_button_is_present_and_only_attack_can_be_set(self):
        a = teacher_env_action(sample(hit='iron_ore', hit_pos=(0, 1, 3)), StageView('aimed', 3., (0, 1, 3), EYE), 'iron_ore')
        self.assertEqual(set(a), set(BUTTONS) | {'camera'})
        self.assertEqual([k for k in BUTTONS if a[k]], ['attack'])


@unittest.skipUnless(STEVE1, 'isolated WSL Torch/STEVE-1 environment required')
class OfficialLabelTest(unittest.TestCase):
    def test_button_list_matches_vpt(self):
        official_mapping()
        from steve1.VPT.lib.actions import Buttons
        self.assertEqual(list(BUTTONS), list(Buttons.ALL))

    def test_policy_label_roundtrips_turn_and_attack(self):
        mapper, transformer = official_mapping()
        action = {k: 0 for k in BUTTONS}; action.update(camera=np.array([0., 10.]), attack=1)
        label = policy_label(action, mapper, transformer)
        self.assertTrue(label['camera_on']); self.assertAlmostEqual(float(label['decoded_camera'][1]), 10., delta=1.)
        self.assertIsInstance(label['buttons'], int); self.assertIsInstance(label['camera'], int)

    def test_still_camera_label_has_camera_off(self):
        mapper, transformer = official_mapping()
        action = {k: 0 for k in BUTTONS}; action.update(camera=np.array([0., 0.]), attack=1)
        self.assertFalse(policy_label(action, mapper, transformer)['camera_on'])


if __name__ == '__main__':
    unittest.main()
