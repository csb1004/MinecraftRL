import unittest
from minecraftrl.navigation_refinement import TrialState
from minecraftrl.common_navigation import navigation_trial_arguments


def pose(x=0.,tick=1):
    return dict(server_x=x,server_y=201.,server_z=0.,yaw=0.,pitch=0.,
        server_tick=tick,server_available=1,error=0,screen=0,held_key_count=0)


class SettledNavigationTests(unittest.TestCase):
    def test_initial_radius_is_not_a_zero_action_success(self):
        state=TrialState([[0.,201.,0.]],'evaluate',arrival_hold_ticks=12)
        state.start(pose());self.assertFalse(state.finished);self.assertEqual(state.receipts,[])
        for tick in range(1,12):
            reward=state.advance(pose(tick=tick),pose(tick=tick+1))
            self.assertEqual(reward['task'],0.);self.assertFalse(state.finished)
        reward=state.advance(pose(tick=12),pose(tick=13))
        self.assertTrue(state.finished);self.assertEqual(reward['task'],1.)
        self.assertEqual(state.receipts[0]['settled_ticks'],12)
        self.assertEqual(state.receipts[0]['actions'],12)
        self.assertTrue(state.receipts[0]['student_success'])

    def test_handoff_cannot_bank_student_hold_ticks_as_teacher_success(self):
        state=TrialState([[0.,201.,0.]],'correction',student_ticks=8,arrival_hold_ticks=12)
        state.start(pose())
        for tick in range(1,9):state.advance(pose(tick=tick),pose(tick=tick+1))
        self.assertEqual(state.owner,'neutral');self.assertEqual(state.arrival.count,8)
        self.assertIsNone(state.advance(pose(tick=9),pose(tick=10)))
        self.assertEqual(state.owner,'teacher');self.assertEqual(state.arrival.count,0)
        for tick in range(10,21):
            state.advance(pose(tick=tick),pose(tick=tick+1));self.assertFalse(state.finished)
        state.advance(pose(tick=21),pose(tick=22))
        self.assertTrue(state.finished);self.assertFalse(state.receipts[0]['student_success'])
        self.assertEqual(state.receipts[0]['settled_ticks'],12)

    def test_new_nearby_goal_requires_its_own_evidence(self):
        state=TrialState([[0.,201.,0.],[.2,201.,0.]],'evaluate',arrival_hold_ticks=3)
        state.start(pose())
        for tick in range(1,4):state.advance(pose(tick=tick),pose(tick=tick+1))
        self.assertEqual(state.index,1);self.assertFalse(state.finished)
        self.assertEqual(state.arrival.count,0)
        for tick in range(4,7):state.advance(pose(tick=tick),pose(tick=tick+1))
        self.assertTrue(state.finished)
        self.assertEqual([r['actions'] for r in state.receipts],[3,3])

    def test_timeout_does_not_count_partial_hold(self):
        state=TrialState([[0.,201.,0.]],'evaluate',arrival_hold_ticks=12)
        state.start(pose(x=1.))
        for tick in range(1,89):state.advance(pose(x=1.,tick=tick),pose(x=1.,tick=tick+1))
        state.advance(pose(x=1.,tick=89),pose(tick=90))
        for tick in range(90,101):reward=state.advance(pose(tick=tick),pose(tick=tick+1))
        self.assertTrue(state.finished);self.assertEqual(reward['task'],-1.)
        self.assertEqual(state.receipts[0]['settled_ticks'],11)
        self.assertFalse(state.receipts[0]['student_success'])

    def test_explicit_trial_flags_preserve_legacy_defaults(self):
        self.assertEqual(navigation_trial_arguments({},'evaluate'),[])
        plan=dict(arrival_hold_ticks=12,student_ticks=60)
        self.assertEqual(navigation_trial_arguments(plan,'correction'),
            ['--arrival-hold-ticks','12','--student-ticks','60'])
        self.assertEqual(navigation_trial_arguments(plan,'evaluate'),['--arrival-hold-ticks','12'])
        for bad in (dict(arrival_hold_ticks=21),dict(student_ticks=61)):
            with self.assertRaises(ValueError):navigation_trial_arguments(bad,'correction')
