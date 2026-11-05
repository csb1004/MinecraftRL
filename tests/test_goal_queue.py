import unittest
from minecraftrl.goal_queue import GoalQueue, validate_program


def goal(key, *, items=None, skill='wood', priority=50, depends_on=None, count=1, requires=None):
    return dict(id=key,objective='collect '+key,items=items or ['oak_log'],skill=skill,
                priority=priority,depends_on=depends_on or [],count=count,requires=requires or [])


class QueueTests(unittest.TestCase):
    def queue(self, *goals):
        return GoalQueue(dict(schema='goal_queue_v1',goals=list(goals)),dict(wood=['oak_log'],dirt=['dirt']))

    def observe(self, q, tick, **inv):
        return q.observe(dict(server_available=1,error=0,server_tick=tick,server_inventory=inv))

    def test_successor_only_starts_after_actual_completion_and_has_fresh_baseline(self):
        q=self.queue(goal('wood'),goal('second', depends_on=['wood']))
        self.observe(q,0);self.assertEqual(q.select()['id'],'wood')
        self.assertEqual(self.observe(q,1,oak_log=1),1.)
        self.assertEqual(q.select()['id'],'second')
        self.assertEqual(self.observe(q,2,oak_log=1),0.)
        self.assertFalse(q.completed)
        self.assertEqual(self.observe(q,3,oak_log=2),1.)
        self.assertTrue(q.completed);self.assertIsNone(q.select())

    def test_waiting_inventory_does_not_count_as_post_activation_acquisition(self):
        q=self.queue(goal('wood'),goal('dirt',items=['dirt'],skill='dirt',depends_on=['wood']))
        self.observe(q,0,dirt=2);q.select();self.observe(q,1,oak_log=1,dirt=3);q.select()
        self.assertEqual(q.records['dirt']['baseline'],3)
        self.assertEqual(self.observe(q,2,oak_log=1,dirt=3),0.)

    def test_unsupported_stone_goal_is_retained_and_not_executed(self):
        q=self.queue(goal('wood'),goal('stone',items=['cobblestone'],skill='stone',depends_on=['wood']))
        self.observe(q,0);q.select();self.observe(q,1,oak_log=1)
        self.assertIsNone(q.select());self.assertFalse(q.completed)
        self.assertEqual(q.records['stone']['reason'],'unsupported_skill')
        self.assertEqual(len(q.snapshot()['program']['goals']),2)

    def test_missing_tools_block_supported_skill_until_observed(self):
        q=self.queue(goal('wood',requires=[dict(any_of=['wooden_pickaxe','stone_pickaxe'],count=1)]))
        self.observe(q,0);self.assertIsNone(q.select())
        self.observe(q,1,stone_pickaxe=1);self.assertEqual(q.select()['id'],'wood')

    def test_stall_and_cancel_do_not_unlock_dependency(self):
        for cancel in (False,True):
            q=self.queue(goal('wood'),goal('later',depends_on=['wood']))
            self.observe(q,0);q.select()
            if cancel:q.cancel('wood',1)
            else:q.block('no_confirmed_progress',dict(ticks=600))
            self.assertIsNone(q.select());self.assertFalse(q.completed)

    def test_reprioritization_retains_active_goal_and_history(self):
        q=self.queue(goal('first'),goal('second',priority=10))
        self.observe(q,0);q.select();q.reprioritize('second',100,1)
        self.assertEqual(q.select()['id'],'first')
        self.assertEqual(q.records['second']['revision'],2)
        with self.assertRaises(ValueError):q.reprioritize('second',0,1)
        self.observe(q,1,oak_log=1)
        selected=q.select()
        self.assertEqual(selected['id'],'second')
        self.assertEqual(selected['priority'],100)
        self.assertEqual(q.program['goals'][1]['priority'],10)  # Original proposal stays immutable.
        self.assertEqual(q.snapshot()['priorities']['records']['second']['priority'],100)

    def test_stale_or_client_only_inventory_cannot_complete(self):
        q=self.queue(goal('wood'));self.observe(q,0);q.select()
        before=q.snapshot()
        with self.assertRaises(ValueError):self.observe(q,0,oak_log=1)
        with self.assertRaises(ValueError):q.observe(dict(server_tick=1,inventory=dict(oak_log=1)))
        self.assertEqual(q.snapshot(),before)

    def test_dependency_cycle_or_unregistered_parent_rejected(self):
        with self.assertRaises(ValueError):self.queue(goal('wood',depends_on=['wood']))
        with self.assertRaises(ValueError):self.queue(goal('wood',depends_on=['missing']))


if __name__=='__main__':unittest.main()
