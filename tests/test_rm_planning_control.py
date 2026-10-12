import unittest
from copy import deepcopy
from unittest.mock import patch

import test_planning_recovery
from minecraftrl.rm_planning_control import RMPlanningController, FixedRMPlanProvider


class RMControlTests(unittest.TestCase):
    run_case=test_planning_recovery.PlanningRecoveryTests.run_case

    def case(self,*,unsupported=False,provider_factory=None,**kwargs):
        def factory(out,gate):return FixedRMPlanProvider(.001,unsupported=unsupported)
        with patch('minecraftrl.planning_recovery.PlanningRecoveryController',RMPlanningController):
            return self.run_case(provider_factory=provider_factory or factory,continue_after_return=True,**kwargs)

    def test_failed_window_requests_new_plan_with_executed_outcome(self):
        p,a,g,r,events,provider=self.case()
        self.assertNotIn('error',r)
        self.assertEqual(provider.calls,2)
        self.assertEqual(len(provider.packets[1]['memory']['attempts']),1)
        self.assertEqual(provider.packets[1]['memory']['attempts'][0]['outcome'],'failed')
        self.assertEqual(provider.packets[1]['previous_plans'][0]['nodes'][0]['status'],'failed')
        self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(len(p.scheduler.state['plans']),2)
        self.assertTrue(any(e.get('reason')=='missing_capability' for e in p.planner_events))
        self.assertEqual(sum(x['mode']=='intervention' for x in p.rows),200)
        self.assertIs(p.environment.reward,p.root_reward_object)
        self.assertEqual(g.state['held_key_count'],0)

    def test_unsupported_subplan_dispatches_no_recovery_action(self):
        p,a,g,r,events,provider=self.case(unsupported=True)
        self.assertEqual(provider.calls,1);self.assertEqual(p.switches,[])
        self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(p.scheduler.state['plans'][0]['nodes'][0]['support'],'unsupported')
        self.assertFalse(p.environment.reward.completed)

    def test_supported_plan_handoff_then_authoritative_root_completion(self):
        def change(g):
            if not g.policy.switches:g.state['wood_v3_kind']=0
            elif g.policy.mode=='intervention':g.state['wood_v3_kind']=1
            elif g.policy.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r)
        self.assertEqual(r['status'],'target_reached')
        self.assertEqual(provider.calls,1)
        self.assertTrue(p.scheduler.state['root']['completed'])
        self.assertEqual(p.scheduler.state['root']['progress'],1)
        self.assertEqual(p.scheduler.state['plans'][0]['nodes'][0]['status'],'completed')
        self.assertEqual([s['mode'] for s in p.switches],['intervention','wood'])
        self.assertTrue(all(row['rm_plan']==saved['rm_plan'] for row,saved in zip(p.rows,a.rows)))
        self.assertEqual(a.resets,1)

    def test_fresh_target_then_restall_uses_next_planner_and_preserves_root(self):
        def change(g):
            if not g.policy.switches:g.state['wood_v3_kind']=0
            elif g.policy.mode=='intervention':g.state['wood_v3_kind']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r)
        self.assertEqual(provider.calls,2)
        self.assertEqual(provider.packets[1]['memory']['attempts'][0]['outcome'],'partial')
        self.assertEqual(len(provider.packets[1]['memory']['failed_attempts']),2)
        self.assertEqual(provider.packets[0]['root_id'],provider.packets[1]['root_id'])
        self.assertEqual(p.scheduler.state['root']['progress'],0)
        self.assertEqual(len(p.switches),2)

    def test_plan_priority_and_text_cannot_bypass_same_failed_method(self):
        class Repeating(FixedRMPlanProvider):
            def poll(self):
                from minecraftrl.rm_planner import fixture_plan
                result=super().poll()
                if result is not None:
                    result=fixture_plan(self.packet)
                    result['steps'][0].update(priority=99,objective='Another explanation of same method')
                return result
        p,a,g,r,events,provider=self.case(provider_factory=lambda *a:Repeating(.001))
        self.assertNotIn('error',r)
        self.assertEqual(provider.calls,2);self.assertEqual(len(p.switches),1)
        self.assertTrue(any(e.get('reason')=='same_failed_attempt' for e in p.planner_events))

    def test_stop_in_wait_or_reply_never_selects_a_node(self):
        for kwargs in (dict(stop_wait=True),dict(stop_reply=True)):
            with self.subTest(kwargs=kwargs):
                p,a,g,r,events,provider=self.case(**kwargs)
                self.assertNotIn('error',r)
                self.assertEqual(r['status'],'cancelled');self.assertEqual(p.switches,[])
                self.assertEqual(g.state['held_key_count'],0)
                self.assertEqual(p.scheduler.state['plans'],[])

    def test_root_goal_can_complete_during_neutral_without_installing_plan(self):
        def change(g):
            if g.policy.phase=='refresh':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertEqual(r['status'],'target_reached')
        self.assertTrue(p.scheduler.state['root']['completed'])
        self.assertEqual(p.scheduler.state['plans'],[]);self.assertEqual(p.switches,[])

    def test_stop_during_recovery_inference_has_no_post_stop_dispatch(self):
        p,a,g,r,events,provider=self.case(stop_recovery_inference=True)
        self.assertNotIn('error',r)
        self.assertEqual(r['status'],'cancelled')
        self.assertEqual(sum(x['mode']=='intervention' for x in p.rows),0)
        self.assertEqual(g.state['held_key_count'],0);self.assertEqual(provider.calls,1)

    def test_smaller_planned_tick_budget_is_used_by_policy(self):
        class Short(FixedRMPlanProvider):
            def poll(self):
                value=super().poll()
                if value is not None:value['steps'][0]['budget_ticks']=7
                return value
        p,a,g,r,events,provider=self.case(provider_factory=lambda *a:Short(.001))
        self.assertNotIn('error',r)
        self.assertEqual(sum(x['mode']=='intervention' for x in p.rows),7)
        self.assertEqual(p.attempts[0]['committed_transitions'],7)

    def test_changed_observed_resources_allow_new_method_review_up_to_three_requests(self):
        def change(g):
            if not g.policy.switches:g.state['wood_v3_kind']=0
            elif g.policy.mode=='intervention':g.state['wood_v3_kind']=1
            elif g.policy.phase=='resumed':g.state['server_inventory']['dirt']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r)
        self.assertEqual(provider.calls,3)
        self.assertEqual(len(p.scheduler.state['plans']),3)
        self.assertEqual(len(p.guard.history.snapshot()['attempts']),2)
        self.assertFalse(p.scheduler.state['root']['completed'])
        self.assertEqual(sum(s['mode']=='intervention' for s in p.switches),2)


if __name__=='__main__':unittest.main()
