import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from minecraftrl.pipeline.rm import Sample
from minecraftrl.pipeline.stage_replay import EYE as EYE_HEIGHT, Geometry, StageView
from minecraftrl.reacquire_teacher import (BUTTONS, aim_delta, first_solid, in_view, policy_label, target_of,
                                           teacher_env_action)

EYE = (0.5, 1.5, 0.5)
STEVE1 = importlib.util.find_spec('torch') is not None and (Path(__file__).resolve().parents[1] / 'external/steve1').exists()


def sample(*, hit='air', hit_pos=None, mining='air', mining_pos=None, gui=False, valid=True, yaw=0., pitch=0.,
           destroyed='air', destroyed_pos=None, eye=EYE):
    return Sample(t=1., tick=20, valid=valid, gui_open=gui, position=(eye[0], eye[1] - EYE_HEIGHT, eye[2]), yaw=yaw,
                  pitch=pitch, action_keys=(), inventory={}, selected_item='stone_pickaxe', destroys=0,
                  last_destroyed=destroyed, last_destroyed_pos=destroyed_pos, last_harvestable=False, mining_block=mining,
                  mining_pos=mining_pos, hit_block=hit, hit_pos=hit_pos)


def pressed(action):
    return [k for k in BUTTONS if action[k]]


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


class MiningFeatureTest(unittest.TestCase):
    """Mining direction input design 2026-11-14 §1: the direction to the block the teacher aims at."""

    def test_names_and_values(self):
        import math
        from minecraftrl.reacquire_teacher import MINING_FEATURES, mining_features
        self.assertEqual(MINING_FEATURES, ('sin_yaw', 'cos_yaw', 'pitch', 'dist'))
        view = StageView('near', 3., (0, 1, -3), EYE)              # behind the player (yaw 0 looks to +z)
        f = dict(zip(MINING_FEATURES, mining_features(sample(), view, 'iron_ore')))
        dpitch, dyaw = aim_delta(EYE, 0., 0., (0, 1, -3))
        self.assertAlmostEqual(f['sin_yaw'], math.sin(math.radians(dyaw)))
        self.assertAlmostEqual(f['cos_yaw'], math.cos(math.radians(dyaw)))
        self.assertLess(f['cos_yaw'], -.9)
        self.assertAlmostEqual(f['pitch'], max(-1., min(1., dpitch / 60.)))
        self.assertAlmostEqual(f['dist'], math.dist(EYE, (.5, 1.5, -2.5)) / 9.)

    def test_the_aimed_block_wins_and_missing_targets_are_zero(self):
        from minecraftrl.reacquire_teacher import mining_features
        aimed = mining_features(sample(hit='iron_ore', hit_pos=(0, 1, 3)), StageView('aimed', 3., (1, 1, 3), EYE), 'iron_ore')
        self.assertAlmostEqual(aimed[0], 0.)                        # straight ahead: no yaw to the aimed block
        self.assertEqual(mining_features(sample(), StageView('far', None, None, EYE), 'iron_ore'), [0.] * 4)
        self.assertEqual(mining_features(sample(gui=True), StageView('gui', 3., (0, 1, 3), EYE), 'iron_ore'), [0.] * 4)
        self.assertEqual(mining_features(sample(), None, 'iron_ore'), [0.] * 4)
        self.assertEqual(mining_features(None, StageView('near', 3., (0, 1, 3), EYE), 'iron_ore'), [0.] * 4)
        far = mining_features(sample(), StageView('far', 30., (0, 1, 30), EYE), 'iron_ore')
        self.assertEqual(far[3], 1.)                                # distance clipped to 1


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

    def test_collect_rows_have_no_label_for_the_centre_and_hold_teachers(self):
        s = sample(destroyed='iron_ore', destroyed_pos=(0, 1, 3))
        view = StageView('collect', 4., (4, 1, 3), EYE)
        self.assertIsNone(teacher_env_action(s, view, 'iron_ore'))
        self.assertIsNone(teacher_env_action(s, view, 'iron_ore', aim='hold'))


class ApproachTeacherTest(unittest.TestCase):
    """Design 2026-11-09 §3: walk closer than 2 blocks before mining, then walk onto the drop."""

    def test_near_far_target_in_front_walks_and_turns_without_attack(self):
        a = teacher_env_action(sample(hit='stone'), StageView('near', 3., (-1, 1, 3), EYE), 'iron_ore', aim='approach')
        self.assertEqual(pressed(a), ['forward']); self.assertEqual(list(a['camera']), [0., 10.])   # 18 degrees left

    def test_near_target_within_two_blocks_turns_without_walking(self):
        a = teacher_env_action(sample(hit='stone'), StageView('near', 2., (1, 1, 1), EYE), 'iron_ore', aim='approach')
        self.assertEqual(pressed(a), []); self.assertNotEqual(float(a['camera'][1]), 0.)

    def test_near_far_target_behind_turns_without_walking(self):
        a = teacher_env_action(sample(hit='grass_block'), StageView('near', 3., (0, 1, -3), EYE), 'iron_ore', aim='approach')
        self.assertEqual(pressed(a), []); self.assertEqual(list(a['camera']), [0., 10.])

    def test_aimed_from_afar_holds_still_attacks_and_walks(self):
        a = teacher_env_action(sample(hit='iron_ore', hit_pos=(1, 1, 3)), StageView('aimed', 3., (1, 1, 3), EYE), 'iron_ore',
                               aim='approach')
        self.assertEqual(pressed(a), ['attack', 'forward']); self.assertEqual(list(a['camera']), [0., 0.])

    def test_mining_within_two_blocks_holds_still_and_attacks_without_walking(self):
        s = sample(hit='iron_ore', hit_pos=(0, 1, 2), mining='iron_ore', mining_pos=(0, 1, 2))
        a = teacher_env_action(s, StageView('mining', 2., (0, 1, 2), EYE), 'iron_ore', aim='approach')
        self.assertEqual(pressed(a), ['attack']); self.assertEqual(list(a['camera']), [0., 0.])

    def test_collect_walks_to_the_broken_target_and_turns_toward_it(self):
        s = sample(destroyed='iron_ore', destroyed_pos=(1, 1, 3))
        view = StageView('collect', 4., (4, 1, 3), EYE)      # the nearest remaining target is elsewhere
        a = teacher_env_action(s, view, 'iron_ore', aim='approach')
        self.assertEqual(pressed(a), ['forward'])
        self.assertEqual([round(float(v), 6) for v in a['camera']], [0., -10.])            # 18 degrees right
        self.assertEqual(target_of(s, view, 'iron_ore'), (1, 1, 3))

    def test_collect_on_the_drop_or_with_the_drop_behind_does_not_walk(self):
        here = teacher_env_action(sample(destroyed='iron_ore', destroyed_pos=(0, 0, 0)), StageView('collect', 4., (4, 1, 3), EYE),
                                  'iron_ore', aim='approach')
        self.assertEqual(pressed(here), [])
        behind = teacher_env_action(sample(destroyed='iron_ore', destroyed_pos=(0, 1, -3)),
                                    StageView('collect', 4., (4, 1, 3), EYE), 'iron_ore', aim='approach')
        self.assertEqual(pressed(behind), []); self.assertEqual(list(behind['camera']), [0., 10.])

    def test_collect_after_breaking_another_block_has_no_label(self):
        s = sample(destroyed='stone', destroyed_pos=(0, 1, 3))
        self.assertIsNone(teacher_env_action(s, StageView('collect', 4., (4, 1, 3), EYE), 'iron_ore', aim='approach'))
        self.assertIsNone(target_of(s, StageView('collect', 4., (4, 1, 3), EYE), 'iron_ore'))

    def test_far_rows_still_have_no_label(self):
        self.assertIsNone(teacher_env_action(sample(), StageView('far', 9., (0, 1, 9), EYE), 'iron_ore', aim='approach'))


class PressTeacherTest(unittest.TestCase):
    """Design 2026-11-10 section 3: one label for the whole mining stage, so the student can hold until the block breaks."""

    def off_target(self, pos):
        # the server still reports mining the target while the crosshair already slipped onto the wall
        return sample(hit='stone', hit_pos=(0, 1, 4), mining='iron_ore', mining_pos=pos)

    def test_mining_with_the_crosshair_off_the_target_still_holds_attack_and_the_view(self):
        view = StageView('mining', 2., (0, 1, 2), EYE)
        a = teacher_env_action(self.off_target((0, 1, 2)), view, 'iron_ore', aim='press')
        self.assertEqual(pressed(a), ['attack']); self.assertEqual(list(a['camera']), [0., 0.])
        old = teacher_env_action(self.off_target((0, 1, 2)), view, 'iron_ore', aim='approach')
        self.assertEqual(pressed(old), [])                      # the approach teacher let go and turned

    def test_mining_from_afar_holds_and_walks(self):
        a = teacher_env_action(self.off_target((1, 1, 3)), StageView('mining', 3., (1, 1, 3), EYE), 'iron_ore', aim='press')
        self.assertEqual(pressed(a), ['attack', 'forward']); self.assertEqual(list(a['camera']), [0., 0.])

    def test_other_stages_keep_the_approach_labels(self):
        cases = [(sample(hit='stone'), StageView('near', 3., (-1, 1, 3), EYE)),
                 (sample(hit='grass_block'), StageView('near', 3., (0, 1, -3), EYE)),
                 (sample(hit='iron_ore', hit_pos=(1, 1, 3)), StageView('aimed', 3., (1, 1, 3), EYE)),
                 (sample(hit='iron_ore', hit_pos=(0, 1, 2), mining='iron_ore', mining_pos=(0, 1, 2)), StageView('mining', 2., (0, 1, 2), EYE)),
                 (sample(destroyed='iron_ore', destroyed_pos=(1, 1, 3)), StageView('collect', 4., (4, 1, 3), EYE))]
        for s, view in cases:
            with self.subTest(stage=view.stage):
                old = teacher_env_action(s, view, 'iron_ore', aim='approach')
                new = teacher_env_action(s, view, 'iron_ore', aim='press')
                self.assertEqual(pressed(new), pressed(old)); self.assertEqual(list(new['camera']), list(old['camera']))

    def test_far_rows_have_no_label_and_older_teachers_are_unchanged(self):
        self.assertIsNone(teacher_env_action(sample(), StageView('far', 9., (0, 1, 9), EYE), 'iron_ore', aim='press'))
        view = StageView('mining', 2., (0, 1, 2), EYE)
        for aim in ('centre', 'hold'):
            with self.subTest(aim=aim):
                self.assertEqual(teacher_env_action(self.off_target((0, 1, 2)), view, 'iron_ore', aim=aim)['attack'], 0)


TARGET = (0, 1, 3)
FRONT_EYE = (0.5, 1.62, 0.5)          # straight in front of the target, 3 blocks away
WALL_END = (-1.5, 1.62, 3.5)          # beside the end of the wall, in line with it: the neighbour stone hides the target
SLANTED = (-0.88, 1.62, 2.7)          # in front of the wall but to the side: the centre is hidden, the front face is not


def wall(extra=None, barrier=True):
    """A stone wall at z = 3 with the target in its middle and one free row behind it (then a barrier), on a floor."""
    blocks = {(x, y, 3): 'stone' for x in (-1, 0, 1) for y in (0, 1, 2)}
    blocks[TARGET] = 'iron_ore'
    if barrier: blocks.update({(x, y, 5): 'barrier' for x in range(-3, 4) for y in (0, 1, 2)})
    blocks.update({(x, -1, z): 'grass_block' for x in range(-4, 5) for z in range(-1, 7)})
    blocks[(0, 1, 2)] = 'air'                                  # the air fill of a fixture is recorded as blocks too
    blocks.update(extra or {})
    return Geometry(blocks)


def near_state(eye, yaw):
    return sample(hit='stone', hit_pos=(-1, 1, 3), yaw=yaw, eye=eye), StageView('near', 2., TARGET, eye)


def same_label(test, a, b):
    test.assertEqual(pressed(a), pressed(b)); test.assertEqual(list(a['camera']), list(b['camera']))


class LineOfSightTest(unittest.TestCase):
    def test_first_solid_block_on_a_segment(self):
        g = wall()
        self.assertEqual(first_solid(g, FRONT_EYE, (0.5, 1.5, 3.5)), TARGET)             # air entries do not block
        self.assertEqual(first_solid(g, WALL_END, (0.5, 1.5, 3.5)), (-1, 1, 3))           # the neighbour stone
        self.assertIsNone(first_solid(g, FRONT_EYE, (0.5, 1.62, 2.5)))                    # ends before the wall
        self.assertEqual(first_solid(g, (0.5, 1.62, 0.5), (0.5, -2., 0.5)), (0, -1, 0))   # straight down: the floor


class SightTeacherTest(unittest.TestCase):
    """Design 2026-11-10 (line of sight) section 3: a near label that still works when the target is hidden."""

    def act(self, eye, yaw, aim='sight', geometry=None):
        s, view = near_state(eye, yaw)
        return teacher_env_action(s, view, 'iron_ore', aim=aim, geometry=wall() if geometry is None else geometry)

    def test_visible_centre_keeps_the_press_label(self):
        a = self.act(FRONT_EYE, 20.)
        same_label(self, a, self.act(FRONT_EYE, 20., aim='press'))
        self.assertEqual(pressed(a), ['forward']); self.assertEqual(list(a['camera'])[1], -10.)   # 3 blocks away, ahead

    def test_hidden_target_walks_to_the_spot_in_front_of_its_open_face(self):
        a = self.act(WALL_END, -100.)            # looking along the wall at the hidden target
        self.assertEqual(pressed(a), ['forward'])
        self.assertEqual(float(a['camera'][1]), -10.)         # the spot (0.5, 1.5) is 35 degrees to the right: clamped
        self.assertAlmostEqual(float(a['camera'][0]), aim_delta(WALL_END, -100., 0., TARGET)[0], places=6)   # target height
        old = self.act(WALL_END, -100., aim='press')
        self.assertEqual(pressed(old), []); self.assertEqual(float(old['camera'][1]), 10.)   # turned at the hidden centre

    def test_hidden_target_turns_before_it_walks(self):
        a = self.act(WALL_END, 90.)              # the standing spot is 135 degrees away
        self.assertEqual(pressed(a), []); self.assertEqual(float(a['camera'][1]), 10.)

    def test_a_spot_needs_room_and_the_nearest_open_side_is_used(self):
        blocked_front = {(0, 0, 1): 'stone'}
        a = self.act(WALL_END, -60., geometry=wall(blocked_front, barrier=False))    # only the back spot (0.5, 5.5) is free
        self.assertEqual(pressed(a), ['forward']); self.assertEqual(float(a['camera'][1]), 10.)   # 15 degrees to the left
        b = self.act(WALL_END, -60., geometry=wall(blocked_front))                    # the barrier fills the back spot
        same_label(self, b, self.act(WALL_END, -60., aim='press'))                    # no spot: the old centre label

    def test_visible_face_is_aimed_at_when_the_centre_is_hidden(self):
        a = self.act(SLANTED, -70.)
        self.assertEqual(pressed(a), [])                                    # 1.6 blocks away: no walking
        self.assertAlmostEqual(float(a['camera'][1]), -5.8, delta=.3)       # just inside the front face, to the right
        old = self.act(SLANTED, -70., aim='press')
        self.assertEqual(float(old['camera'][1]), 10.)                      # the hidden centre is to the left

    def test_other_teachers_ignore_the_geometry_and_sight_needs_it(self):
        s, view = near_state(WALL_END, -100.)
        for aim in ('centre', 'hold', 'approach', 'press'):
            with self.subTest(aim=aim):
                same_label(self, teacher_env_action(s, view, 'iron_ore', aim=aim, geometry=wall()),
                           teacher_env_action(s, view, 'iron_ore', aim=aim))
        same_label(self, teacher_env_action(s, view, 'iron_ore', aim='sight'),          # no geometry: the press label
                   teacher_env_action(s, view, 'iron_ore', aim='press'))

    def test_aimed_mining_and_collect_keep_the_press_labels(self):
        cases = [(sample(hit='iron_ore', hit_pos=TARGET, eye=FRONT_EYE), StageView('aimed', 3., TARGET, FRONT_EYE)),
                 (sample(hit='stone', hit_pos=(-1, 1, 3), mining='iron_ore', mining_pos=TARGET, eye=WALL_END),
                  StageView('mining', 2., TARGET, WALL_END)),
                 (sample(destroyed='iron_ore', destroyed_pos=TARGET, eye=WALL_END), StageView('collect', 4., (4, 1, 3), WALL_END))]
        for s, view in cases:
            with self.subTest(stage=view.stage):
                same_label(self, teacher_env_action(s, view, 'iron_ore', aim='sight', geometry=wall()),
                           teacher_env_action(s, view, 'iron_ore', aim='press'))


class AimFeatureTest(unittest.TestCase):
    """Mining aim point design 2026-11-14 §1: the direction to where the sight teacher turns the view."""

    def near(self, eye, yaw, geometry=None):
        s, view = near_state(eye, yaw)
        return s, view, wall() if geometry is None else geometry

    def test_names(self):
        from minecraftrl.reacquire_teacher import AIM_FEATURES
        self.assertEqual(AIM_FEATURES, ('aim_sin_yaw', 'aim_cos_yaw', 'aim_pitch', 'aim_dist'))

    def test_near_teacher_camera_is_the_aim_direction_clipped(self):
        from minecraftrl.reacquire_teacher import MAX_TURN, aim_view
        cases = [(FRONT_EYE, 20., None), (SLANTED, -70., None), (WALL_END, -100., None), (WALL_END, 90., None),
                 (WALL_END, -60., wall({(0, 0, 1): 'stone'}, barrier=False))]
        for eye, yaw, geometry in cases:
            with self.subTest(eye=eye, yaw=yaw):
                s, view, g = self.near(eye, yaw, geometry)
                dpitch, dyaw, _ = aim_view(s, view, 'iron_ore', g)
                label = teacher_env_action(s, view, 'iron_ore', aim='sight', geometry=g)['camera']
                clip = lambda v: max(-MAX_TURN, min(MAX_TURN, v))   # noqa: E731
                self.assertAlmostEqual(float(label[0]), clip(dpitch), places=6)
                self.assertAlmostEqual(float(label[1]), clip(dyaw), places=6)

    def test_hidden_centre_points_at_the_visible_face_not_the_centre(self):
        import math
        from minecraftrl.reacquire_teacher import aim_features, mining_features
        s, view, g = self.near(SLANTED, -70.)
        aim, centre = aim_features(s, view, 'iron_ore', g), mining_features(s, view, 'iron_ore')
        self.assertAlmostEqual(math.degrees(math.atan2(aim[0], aim[1])), -5.8, delta=.3)    # the face, to the right
        self.assertGreater(math.degrees(math.atan2(centre[0], centre[1])), 10.)            # the hidden centre, left

    def test_walk_plan_points_the_yaw_at_the_spot_and_measures_the_walk(self):
        import math
        from minecraftrl.reacquire_teacher import aim_features
        s, view, g = self.near(WALL_END, -100.)
        f = aim_features(s, view, 'iron_ore', g)
        self.assertAlmostEqual(math.degrees(math.atan2(f[0], f[1])), -35., delta=1.)        # the spot (0.5, 1.5)
        self.assertAlmostEqual(f[2], aim_delta(WALL_END, -100., 0., TARGET)[0] / 60., places=6)
        self.assertAlmostEqual(f[3], math.hypot(.5 - WALL_END[0], 1.5 - WALL_END[2]) / 9., places=6)

    def test_visible_centre_other_stages_and_no_geometry_keep_the_centre_features(self):
        from minecraftrl.reacquire_teacher import aim_features, mining_features
        s, view, g = self.near(FRONT_EYE, 20.)
        self.assertEqual(aim_features(s, view, 'iron_ore', g), mining_features(s, view, 'iron_ore'))
        s, view, _ = self.near(SLANTED, -70.)
        self.assertEqual(aim_features(s, view, 'iron_ore', None), mining_features(s, view, 'iron_ore'))
        cases = [(sample(hit='iron_ore', hit_pos=TARGET, eye=FRONT_EYE), StageView('aimed', 3., TARGET, FRONT_EYE)),
                 (sample(hit='stone', hit_pos=(-1, 1, 3), mining='iron_ore', mining_pos=TARGET, eye=WALL_END),
                  StageView('mining', 2., TARGET, WALL_END)),
                 (sample(destroyed='iron_ore', destroyed_pos=TARGET, eye=WALL_END), StageView('collect', 4., (4, 1, 3), WALL_END))]
        for s, view in cases:
            with self.subTest(stage=view.stage):
                self.assertEqual(aim_features(s, view, 'iron_ore', wall()), mining_features(s, view, 'iron_ore'))

    def test_missing_inputs_are_zero(self):
        from minecraftrl.reacquire_teacher import aim_features
        self.assertEqual(aim_features(sample(), StageView('far', None, None, EYE), 'iron_ore', wall()), [0.] * 4)
        self.assertEqual(aim_features(sample(gui=True), StageView('gui', 3., (0, 1, 3), EYE), 'iron_ore', wall()), [0.] * 4)
        self.assertEqual(aim_features(sample(), None, 'iron_ore', wall()), [0.] * 4)
        self.assertEqual(aim_features(None, StageView('near', 3., (0, 1, 3), EYE), 'iron_ore', wall()), [0.] * 4)

    def test_feature_values_picks_the_set_by_name(self):
        from minecraftrl.reacquire_teacher import AIM_FEATURES, MINING_FEATURES, aim_features, feature_values, mining_features
        s, view, g = self.near(SLANTED, -70.)
        self.assertEqual(feature_values(MINING_FEATURES, s, view, 'iron_ore', g), mining_features(s, view, 'iron_ore'))
        self.assertEqual(feature_values(list(AIM_FEATURES), s, view, 'iron_ore', g), aim_features(s, view, 'iron_ore', g))
        with self.assertRaises(ValueError): feature_values(('a', 'b'), s, view, 'iron_ore', g)


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

    def test_policy_label_roundtrips_walk_and_attack_with_a_still_camera(self):
        mapper, transformer = official_mapping()
        action = {k: 0 for k in BUTTONS}; action.update(camera=np.array([0., 0.]), attack=1, forward=1)
        label = policy_label(action, mapper, transformer)
        self.assertFalse(label['camera_on'])
        decoded = transformer.policy2env(mapper.to_factored(dict(buttons=np.array([[label['buttons']]]),
                                                                 camera=np.array([[label['camera']]]))))
        self.assertEqual((int(np.asarray(decoded['forward']).reshape(-1)[0]), int(np.asarray(decoded['attack']).reshape(-1)[0])),
                         (1, 1))

    def test_still_camera_label_has_camera_off(self):
        mapper, transformer = official_mapping()
        action = {k: 0 for k in BUTTONS}; action.update(camera=np.array([0., 0.]), attack=1)
        self.assertFalse(policy_label(action, mapper, transformer)['camera_on'])


if __name__ == '__main__':
    unittest.main()
