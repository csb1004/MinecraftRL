import unittest
from minecraftrl.interaction_progress import InteractionProgress
from minecraftrl.priority_schedule import PriorityLedger


class ProgressTests(unittest.TestCase):
    def tracker(self, **kw):
        return InteractionProgress(tick=0, interactions=7, inventory_count=0, **kw)

    def observe(self, state, tick, **kw):
        return state.observe(**dict(dict(tick=tick, interactions=7, inventory_count=0,
                                        known=False, reachable=False), **kw))

    def test_confirmed_interaction_survives_missing_hint_and_new_target(self):
        p=self.tracker()
        self.observe(p, 1, interactions=8)
        self.assertEqual(self.observe(p, 50, interactions=8)['stage'], 'collect')
        self.assertEqual(self.observe(p, 51, interactions=8, known=True, reachable=True)['stage'], 'collect')
        self.assertEqual(self.observe(p, 52, interactions=8, inventory_count=1, complete=True)['stage'], 'complete')

    def test_result_wait_is_bounded_and_never_claims_success(self):
        p=self.tracker(recovery_ticks=5)
        self.observe(p, 1, interactions=8)
        row=self.observe(p, 6, interactions=8)
        self.assertEqual(row['blocked'], 'result_not_acquired')
        self.assertFalse(row['completed'])
        with self.assertRaises(RuntimeError):self.observe(p, 7, interactions=8)

    def test_flickering_target_does_not_reset_progress_budget(self):
        p=self.tracker(stall_ticks=5)
        self.observe(p, 1, known=True)
        self.observe(p, 2)
        self.observe(p, 3, known=True)
        self.assertEqual(self.observe(p, 6, known=True)['blocked'], 'no_confirmed_progress')

    def test_unknown_and_stale_observation_preserve_memory(self):
        p=self.tracker(); before=p.snapshot()
        for kwargs in (dict(tick=0), dict(tick=1, known=None), dict(tick=1, interactions=6)):
            with self.assertRaises(ValueError):self.observe(p, **kwargs)
            self.assertEqual(p.snapshot(), before)

    def test_old_harvest_is_not_new_pending_result(self):
        p=self.tracker()
        self.assertEqual(self.observe(p, 1)['stage'], 'search')

    def test_partial_acquisition_starts_next_cycle_without_duplicate_credit(self):
        p=self.tracker()
        self.observe(p, 1, interactions=8)
        self.assertEqual(self.observe(p, 2, interactions=8, inventory_count=1)['stage'], 'search')
        self.observe(p, 3, interactions=9, inventory_count=0)
        self.assertEqual(self.observe(p, 4, interactions=9, inventory_count=1)['stage'], 'collect')

    def test_root_completion_overrides_missing_hint(self):
        p=self.tracker()
        self.assertTrue(self.observe(p, 1, complete=True, known=None)['completed'])


class CompletionSelectionTests(unittest.TestCase):
    def test_priority_changes_cannot_preempt_active_goal(self):
        p=PriorityLedger();p.register('wood', priority=50);p.register('dirt', priority=10)
        self.assertEqual(p.select(until_complete=True)['goal_id'], 'wood')
        p.advance(100, serving='wood', service_s=100);p.reprioritize('dirt',100)
        self.assertEqual(p.select(until_complete=True)['goal_id'], 'wood')
        p.set_status('wood','completed')
        self.assertEqual(p.select(until_complete=True)['goal_id'], 'dirt')

    def test_equal_priority_is_registration_order_not_name(self):
        p=PriorityLedger();p.register('z_first');p.register('a_second')
        self.assertEqual(p.select(until_complete=True)['goal_id'], 'z_first')

    def test_blocked_or_cancelled_prerequisite_does_not_unlock_successor(self):
        for status in ('blocked','cancelled','paused'):
            p=PriorityLedger();p.register('wood');p.register('stone', priority=100, depends_on=['wood'])
            p.set_status('wood',status)
            self.assertIsNone(p.select(until_complete=True)['goal_id'])


if __name__=='__main__':unittest.main()
