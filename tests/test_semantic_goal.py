import unittest
from minecraftrl.goal_queue import GoalQueue
from minecraftrl.semantic_goal import context, contract, CONTEXT_FIELDS, ResultMonitor
from test_goal_queue import goal


class SemanticGoalTests(unittest.TestCase):
    def queue(self, count=3, skill='wood_memory'):
        q=GoalQueue(dict(schema='goal_queue_v1',goals=[goal('arbitrary_name',count=count,skill=skill)]),{skill:['oak_log']})
        q.observe(dict(server_available=1,error=0,server_tick=1,server_inventory={}))
        q.select()
        return q

    def test_semantics_preserve_target_quantity_and_new_acquisition_baseline(self):
        q=self.queue();q.observe(dict(server_available=1,error=0,server_tick=2,server_inventory={'oak_log':1}))
        v,c=context(q,'mine',None,{'target_known':True,'target_reachable':False})
        self.assertEqual(c['target'],['oak_log']);self.assertEqual(c['quantity'],3)
        self.assertEqual(c['result'],'inventory_increase_from_activation')
        self.assertAlmostEqual(v[CONTEXT_FIELDS.index('progress_fraction')],1/3)
        self.assertNotIn('arbitrary_name',str(v))

    def test_unknown_is_distinct_from_false_and_dirt_does_not_inherit_wood_observer(self):
        q=self.queue(skill='dirt_fixed')
        unknown,c=context(q,'collect',None,dict(target_known=None,target_reachable=None))
        negative,_=context(q,'collect',None,dict(target_known=False,target_reachable=False))
        self.assertEqual(c['stage'],'interact')
        self.assertEqual(unknown[CONTEXT_FIELDS.index('known_valid')],0.)
        self.assertEqual(negative[CONTEXT_FIELDS.index('known_valid')],1.)
        self.assertEqual(unknown[CONTEXT_FIELDS.index('pending_result')],0.)

    def test_repetition_is_not_failure_and_progress_resets_monitor(self):
        q=self.queue();m=ResultMonitor();m.observe(q)
        for tick in range(2,130):
            q.observe(dict(server_available=1,error=0,server_tick=tick,server_inventory={}))
            out=m.observe(q,{'attack':[1]})
        self.assertGreater(out['exact_action_repeat'],120)
        self.assertFalse(out['repeat_is_failure'])
        q.observe(dict(server_available=1,error=0,server_tick=130,server_inventory={'oak_log':1}))
        self.assertEqual(m.observe(q)['no_inventory_progress_ticks'],0)
        with self.assertRaises(ValueError):m.observe(q)

    def test_no_context_for_unbound_or_unsupported_goal(self):
        q=self.queue();q.block('unsupported',{})
        with self.assertRaises(ValueError):contract(q)

    def test_blocked_goal_preserves_final_failure_evidence(self):
        q=self.queue();m=ResultMonitor();m.observe(q)
        q.observe(dict(server_available=1,error=0,server_tick=601,server_inventory={}))
        q.block('no_confirmed_progress',{})
        result=m.observe(q,{'attack':[1]})
        self.assertIsNone(q.active)
        self.assertEqual(result['goal'],'arbitrary_name')
        self.assertEqual(result['no_inventory_progress_ticks'],600)

    def test_completed_goal_records_final_inventory_progress(self):
        q=self.queue(count=1);m=ResultMonitor();m.observe(q)
        q.observe(dict(server_available=1,error=0,server_tick=2,server_inventory={'oak_log':1}))
        result=m.observe(q,{'attack':[1]})
        self.assertTrue(q.completed)
        self.assertEqual(result['goal'],'arbitrary_name')
        self.assertEqual(result['no_inventory_progress_ticks'],0)
        self.assertEqual(m.peak,1)

    def test_monitor_resets_for_a_different_active_goal(self):
        first=goal('first',count=1,skill='wood_memory');second=goal('second',count=1,skill='wood_memory')
        second['depends_on']=['first']
        q=GoalQueue(dict(schema='goal_queue_v1',goals=[first,second]),{'wood_memory':['oak_log']})
        q.observe(dict(server_available=1,error=0,server_tick=1,server_inventory={}))
        self.assertEqual(q.select()['id'],'first')
        m=ResultMonitor();m.observe(q)
        q.observe(dict(server_available=1,error=0,server_tick=2,server_inventory={'oak_log':1}))
        self.assertEqual(q.select()['id'],'second')
        result=m.observe(q,{'attack':[1]})
        self.assertEqual(result['goal'],'second')
        self.assertEqual(result['no_inventory_progress_ticks'],0)
        self.assertEqual(m.peak,0)


if __name__=='__main__':unittest.main()
