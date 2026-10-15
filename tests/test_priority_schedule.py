import unittest

from minecraftrl import priority_schedule


class PriorityScheduleTests(unittest.TestCase):
    def ledger(self):
        self.assertTrue(hasattr(priority_schedule, 'PriorityLedger'), 'persistent aging scheduler missing')
        return priority_schedule.PriorityLedger()

    def test_old_low_priority_gets_service_despite_new_high_priority(self):
        ledger = self.ledger()
        ledger.register('old', priority=0)
        ledger.register('busy', priority=100)
        self.assertEqual(ledger.select()['goal_id'], 'busy')
        ledger.advance(60, serving='busy', service_s=60)
        ledger.register('new', priority=100)
        selected = ledger.select()
        self.assertEqual(selected['goal_id'], 'old')
        self.assertEqual(selected['reason'], 'maximum_wait')

    def test_partial_service_and_reprioritization_do_not_erase_age_or_wait(self):
        ledger = self.ledger()
        ledger.register('old', priority=0)
        ledger.advance(61)
        ledger.select()
        ledger.advance(1, serving='old', service_s=1)
        ledger.reprioritize('old', 100)
        self.assertEqual(ledger.records['old']['age_s'], 62)
        self.assertEqual(ledger.records['old']['wait_s'], 61)
        ledger.advance(9, serving='old', service_s=9)
        self.assertEqual(ledger.records['old']['age_s'], 71)
        self.assertEqual(ledger.records['old']['wait_s'], 0)

    def test_urgent_parent_lends_priority_and_service_to_prerequisite(self):
        ledger = self.ledger()
        ledger.register('inspect', priority=0)
        ledger.register('wood', priority=90, depends_on=['inspect'])
        ledger.register('dirt', priority=80)
        selected = ledger.select()
        self.assertEqual(selected['goal_id'], 'inspect')
        self.assertEqual(selected['inherited_from'], 'wood')
        ledger.advance(10, serving='inspect', service_s=10)
        self.assertEqual(ledger.records['wood']['service_s'], 10)
        ledger.set_status('inspect', 'completed')
        self.assertEqual(ledger.select()['goal_id'], 'wood')

    def test_pause_cancel_and_blocked_goals_cannot_force_execution(self):
        ledger = self.ledger()
        for name in ('paused', 'cancelled', 'blocked'):
            ledger.register(name, priority=100)
            ledger.set_status(name, name)
        ledger.register('ready', priority=0)
        ledger.advance(600)
        self.assertEqual(ledger.records['paused']['age_s'], 0)
        self.assertEqual(ledger.records['blocked']['age_s'], 600)
        self.assertEqual(ledger.select()['goal_id'], 'ready')

    def test_selected_task_keeps_bounded_slice_and_short_selection_is_not_service(self):
        ledger = self.ledger()
        ledger.register('first', priority=10)
        ledger.select()
        ledger.advance(1, serving='first', service_s=1)
        ledger.register('second', priority=100)
        self.assertEqual(ledger.select()['goal_id'], 'first')
        ledger.advance(9, serving='first', service_s=9)
        self.assertEqual(ledger.select()['goal_id'], 'second')
        self.assertEqual(ledger.records['second']['wait_s'], 9)

    def test_checkpoint_preserves_clocks_without_resuming_or_counting_offline_time(self):
        ledger = self.ledger()
        ledger.register('goal', priority=50, deadline_s=60)
        ledger.advance(42)
        ledger.select()
        saved = ledger.snapshot()
        restored = priority_schedule.PriorityLedger.restore(saved)
        self.assertEqual(restored.records, ledger.records)
        self.assertIsNone(restored.active)
        self.assertEqual(restored.records['goal']['age_s'], 42)
        self.assertGreater(restored.scores()['goal']['deadline_boost'], 0)

    def test_invalid_clock_and_missing_dependency_do_not_mutate_state(self):
        ledger = self.ledger()
        ledger.register('goal', priority=20)
        saved = ledger.snapshot()
        for delta in (-1, float('nan'), True):
            with self.assertRaises(ValueError): ledger.advance(delta)
        with self.assertRaises(ValueError): ledger.register('bad', depends_on=['absent'])
        with self.assertRaises(ValueError): ledger.advance(1, serving='goal', service_s=2)
        self.assertEqual(ledger.snapshot(), saved)

    def test_replan_dependency_cycle_is_rejected_atomically(self):
        ledger=self.ledger()
        ledger.register('a'); ledger.register('b',depends_on=['a'])
        saved=ledger.snapshot()
        with self.assertRaises(ValueError): ledger.dependencies({'a':['b']})
        self.assertEqual(ledger.snapshot(),saved)

    def test_completed_parent_does_not_lend_urgency_and_terminal_cannot_be_rearmed(self):
        ledger=self.ledger()
        ledger.register('old',priority=100); ledger.advance(100)
        ledger.set_status('old','completed'); ledger.register('next',priority=0)
        self.assertEqual(ledger.select()['goal_id'],'next')
        with self.assertRaises(ValueError): ledger.set_status('old','pending')

    def test_overdue_goal_cannot_start_at_unsafe_boundary(self):
        ledger=self.ledger(); ledger.register('old'); ledger.advance(100)
        self.assertIsNone(ledger.select(safe_boundary=False)['goal_id'])

    def test_inheritance_preserves_relative_priorities_of_ready_prerequisites(self):
        ledger=self.ledger();ledger.register('a_low',priority=0);ledger.register('z_high',priority=80)
        ledger.register('parent',priority=100,depends_on=['a_low','z_high'])
        self.assertEqual(ledger.select()['goal_id'],'z_high')
