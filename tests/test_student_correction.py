import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from minecraftrl.student_correction import StudentCorrection, recovery_waypoints


def world(mirror=False):
    return dict(seed=104743, kind='recovery', experimental_split='train', chunk=[0, 0],
                curriculum=dict(wall_end=11, mirror_z=mirror))


def gui(tick=0):
    return dict(server_available=1, error=0, server_tick=tick, held_key_count=0, screen=0,
        server_x=6.7, server_y=201., server_z=3.5, yaw=-90., pitch=0., eye_x=6.7, eye_y=202.6,
        eye_z=3.5, hit_x=7, hit_y=202, hit_z=3, server_inventory={'oak_log': 0},
        wood_v3_kind=1, wood_v3_distance=5., wood_v3_reachable=0,
        wood_v3_target_x=11, wood_v3_target_y=202, wood_v3_target_z=3)


class Student:
    def __init__(self):
        self.rows = []
        self.calls = 0
    def reset(self, goal): pass
    def set_reward_context(self, context): self.context = list(context)
    def act(self, frame):
        self.calls += 1
        return {'attack': 1, 'camera': [0., 0.]}
    def commit(self, action, reward, terminal): self.rows.append(action)


class StudentCorrectionTests(unittest.TestCase):
    def make_policy(self):
        raw = SimpleNamespace(action_space=SimpleNamespace(noop=lambda: {'attack': 0, 'camera': [0., 0.]}))
        p = StudentCorrection(Student(), raw, world(), Path('test'))
        p.environment = SimpleNamespace(gui=gui(), reward=SimpleNamespace(completed=False))
        p.set_reward_context([0., 1., 0., 0., 0., .2])
        p.reset('collect wood')
        return p

    def step(self, p, *, held=0, terminal=False):
        action = p.act(None)
        after = deepcopy(p.environment.gui)
        after['server_tick'] += 1
        after['held_key_count'] = held
        p.environment.gui = after
        p.commit(action, 0., terminal)

    def stall(self, p):
        for _ in range(200): self.step(p)
        self.assertEqual(p.owner, 'handoff_neutral')

    def test_real_detector_handoff_keeps_complete_context_and_expert_mask(self):
        p = self.make_policy()
        self.stall(p)
        self.assertEqual(p.trigger['step'], 200)
        self.assertFalse(any(r['loss_eligible'] for r in p.rows))
        self.step(p)
        self.assertTrue(p.handoffs[0]['verified'])
        self.assertEqual(p.owner, 'teacher')
        self.step(p)
        self.assertEqual(p.student.calls, 200)
        self.assertEqual(len(p.student.rows), 200)
        self.assertEqual([r['owner'] for r in p.rows[-2:]], ['handoff_neutral', 'teacher'])
        self.assertTrue(p.rows[-1]['loss_eligible'])
        for i, row in enumerate(p.rows):
            self.assertEqual(row['frame'], f'frames/{i:06d}.png')
            if i: self.assertEqual(row['server_before'], p.rows[i-1]['server_after'])

    def test_neutral_failure_records_transition_but_never_starts_teacher(self):
        p = self.make_policy(); self.stall(p)
        with self.assertRaisesRegex(ValueError, 'neutral keys'): self.step(p, held=1)
        self.assertFalse(p.handoffs[0]['teacher_started'])
        self.assertFalse(p.rows[-1]['loss_eligible'])

    def test_stop_at_handoff_does_not_start_teacher(self):
        p = self.make_policy(); self.stall(p)
        action = p.act(None)
        p.environment.gui = gui(201)
        p.is_stopped = lambda: True
        p.commit(action, 0., False)
        self.assertFalse(p.handoffs[0]['teacher_started'])
        with self.assertRaises(InterruptedError): p.act(None)

    def test_student_success_never_hands_off(self):
        p = self.make_policy()
        p.environment.reward.completed = True
        self.step(p, terminal=True)
        self.assertTrue(p.student_success)
        self.assertEqual(p.handoffs, [])
        with self.assertRaises(InterruptedError): p.act(None)

    def test_wrong_action_is_not_committed(self):
        p = self.make_policy(); p.act(None)
        with self.assertRaises(ValueError): p.commit({'forward': 1}, 0., False)
        self.assertEqual(p.rows, [])
        self.assertEqual(p.student.rows, [])

    def test_routes_use_actual_pose_and_skip_already_crossed_wall(self):
        g = gui(); g['server_z'] = 1.3
        self.assertEqual(recovery_waypoints(world(), g)[0], (5.5, 1.3))
        g['server_z'] = 14.7
        self.assertAlmostEqual(recovery_waypoints(world(True), g)[0][1], 1.3)
        g.update(server_x=9.8, server_z=7.)
        self.assertEqual(recovery_waypoints(world(), g), [(9.5, 7.), (9.5, 3.5)])
        g.update(server_x=7.5, server_z=12.5)
        self.assertEqual(len(recovery_waypoints(world(), g)), 2)
        g['server_z'] = 5.
        with self.assertRaises(ValueError): recovery_waypoints(world(), g)
        g['server_x'] = float('nan')
        with self.assertRaises(ValueError): recovery_waypoints(world(), g)

    def test_open_gui_cannot_be_silently_driven_as_movement(self):
        p = self.make_policy(); self.stall(p)
        p.environment.gui['screen'] = 1
        with self.assertRaisesRegex(ValueError, 'open GUI'): self.step(p)
        self.assertFalse(p.handoffs[0]['teacher_started'])


if __name__ == '__main__': unittest.main()
