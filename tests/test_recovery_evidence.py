import importlib.util
import unittest
from copy import deepcopy

from minecraftrl.stall_detection import StallDetector
from minecraftrl.stall_replay import wood_sample


def obs(tick=0, **changes):
    state = dict(server_available=1,error=0,server_tick=tick,server_x=0.,server_y=64.,server_z=0.,
        yaw=0.,server_inventory={'oak_log':0},wood_v3_kind=1,wood_v3_reachable=0,
        wood_v3_distance=5.,wood_v3_target_x=5,wood_v3_target_y=64,wood_v3_target_z=0,
        wood_server_mining=0)
    state.update(changes); return state


class RecoveryEvidenceTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('minecraftrl.recovery_evidence'),
            'Evidence projection for planner is missing')
        from minecraftrl import recovery_evidence
        return recovery_evidence

    def stall(self):
        detector = StallDetector()
        for tick in range(201):
            state = obs(tick)
            decision = detector.update(wood_sample(state,{'attack':[1]},'test'))
        self.assertEqual(decision['event'],'stalled')
        return state,decision

    def test_stall_is_replan_evidence_not_impossibility_or_dispatch_permission(self):
        m = self.module(); state,decision = self.stall(); original = deepcopy((state,decision))
        result = m.assess_stall(state,decision,baseline_count=0)
        self.assertTrue(result['replan_candidate'])
        self.assertFalse(result['task_impossible']); self.assertFalse(result['execution_authorized'])
        self.assertEqual(result['rm_effect'],'preserve')
        self.assertEqual((state,decision),original)

    def test_mining_flag_is_not_damage_progress(self):
        m = self.module(); state,decision = self.stall()
        state.update(wood_server_mining=1,wood_v3_reachable=1)
        result = m.assess_stall(state,decision,baseline_count=0)
        self.assertFalse(result['facts']['mining_damage_observed'])
        self.assertIn('mining_damage_unmeasured',result['uncertainties'])
        self.assertEqual(result['classification'],'stagnation_suspected')

    def test_goal_or_stop_suppresses_replan_even_with_stale_stall(self):
        m = self.module(); state,decision = self.stall()
        for changes,stopped,expected in (({'server_inventory':{'oak_log':1}},False,'goal_completed'),
                                        ({},True,'stopped')):
            result = m.assess_stall(dict(state,**changes),decision,baseline_count=0,stopped=stopped)
            self.assertFalse(result['replan_candidate']); self.assertEqual(result['classification'],expected)

    def test_observation_gap_never_becomes_policy_failure(self):
        m = self.module(); state,decision = self.stall(); state['server_available'] = 0
        result = m.assess_stall(state,decision,baseline_count=0)
        self.assertEqual(result['classification'],'observation_unreliable')
        self.assertFalse(result['replan_candidate'])

    def test_motion_and_rotation_do_not_make_recovery_success(self):
        m = self.module(); before = obs()
        after = [obs(i,server_z=i*.1,yaw=float(i)) for i in range(1,201)]
        result = m.assess_attempt(before,after,reason='tick_budget',baseline_count=0,budget_ticks=200)
        self.assertEqual(result['classification'],'recovery_budget_exhausted')
        self.assertTrue(result['replan_candidate']); self.assertFalse(result['goal_completed'])
        self.assertGreater(result['max_horizontal_displacement'],.5)

    def test_reachable_target_is_handoff_evidence_not_goal_completion(self):
        m = self.module(); result = m.assess_attempt(obs(),[obs(1,wood_v3_reachable=1,wood_v3_distance=2.)],
            reason='wood_interaction_ready',baseline_count=0,budget_ticks=200)
        self.assertEqual(result['classification'],'interaction_ready')
        self.assertFalse(result['goal_completed']); self.assertFalse(result['replan_candidate'])

    def test_same_target_approach_progress_is_separate_from_acquisition(self):
        m = self.module(); before = obs()
        for after,expected in ((obs(1,wood_v3_distance=4.),True),
                               (obs(1,wood_v3_distance=1.,wood_v3_target_x=7),False)):
            result = m.assess_attempt(before,[after],reason='control_budget',baseline_count=0,budget_ticks=200)
            self.assertEqual(result['same_target_approach_progress'],expected)
            self.assertFalse(result['goal_completed']); self.assertFalse(result['replan_candidate'])

    def test_invalid_ticks_and_empty_attempt_cannot_count_as_recovery_failure(self):
        m = self.module()
        for after in ([],[obs(2)],[obs(1),obs(1)]):
            result = m.assess_attempt(obs(),after,reason='control_budget',baseline_count=0,budget_ticks=200)
            self.assertEqual(result['classification'],'observation_unreliable')
            self.assertFalse(result['replan_candidate'])

    def test_finding_a_target_is_reported_even_without_reaching_or_collecting_it(self):
        m = self.module()
        before = obs(wood_v3_kind=0,wood_v3_distance=0.)
        after = [obs(i) for i in range(1,201)]
        result = m.assess_attempt(before,after,reason='tick_budget',baseline_count=0,budget_ticks=200)
        self.assertTrue(result['target_became_observed'])
        self.assertEqual(result['classification'],'recovery_incomplete_with_target_evidence')
        self.assertFalse(result['goal_completed'])
        self.assertTrue(result['replan_candidate'])

    def test_wall_clock_skill_budget_preserves_partial_evidence(self):
        result=self.module().assess_attempt(obs(),[obs(1,wood_v3_kind=0)],
            reason='planning_time_budget',baseline_count=0,budget_ticks=200)
        self.assertEqual(result['classification'],'recovery_budget_exhausted')
        self.assertTrue(result['replan_candidate'])
        self.assertFalse(result['current_target_tracked'])
        self.assertEqual(result['committed_ticks'],1)

    def test_fresh_target_boundary_rejects_an_old_lock(self):
        m=self.module()
        before=obs(wood_v3_kind=0)
        result=m.assess_attempt(before,[obs(1)],reason='wood_target_acquired',baseline_count=0,budget_ticks=200)
        self.assertEqual(result['classification'],'target_acquired')
        self.assertFalse(result['goal_completed'])
        self.assertTrue(result['target_acquired_on_last_transition'])
        # The sensor keeps the lock when looking away; kind=1 alone is not visibility.
        with self.assertRaises(ValueError):
            m.assess_attempt(before,[obs(1),obs(2)],reason='wood_target_acquired',baseline_count=0,budget_ticks=200)


if __name__ == '__main__': unittest.main()
