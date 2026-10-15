import json,unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import test_planning_recovery
from test_subgoal_control import catalogue
from minecraftrl.rm_navigation import RMNavigationController,FixedNavigationProvider
from minecraftrl.waypoint_skill import NAMES
from minecraftrl.rm_planner import validate_plan,payload


class Bound(RMNavigationController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,catalogue=catalogue(),navigation_checkpoint='fixture',max_requests=2,**kwargs)
    def attach_navigation(self):
        self.nav_head=SimpleNamespace(goal=None);self.navigation_sha256='c'*64


class RMNavigationTests(unittest.TestCase):
    run_case=test_planning_recovery.PlanningRecoveryTests.run_case

    def case(self,**kwargs):
        factory=kwargs.pop('provider_factory',lambda *a:FixedNavigationProvider(.001,max_calls=2))
        with patch('minecraftrl.planning_recovery.PlanningRecoveryController',Bound):
            return self.run_case(provider_factory=factory,continue_after_return=True,**kwargs)

    def test_registered_target_becomes_neural_input_then_approach_and_original_root(self):
        def change(g):
            p=g.policy
            if p.phase=='waypoint':
                target=p.navigation_cap['arguments']['target']
                g.state.update(zip(('server_x','server_y','server_z'),target))
            elif p.phase=='subgoal':g.state['wood_v3_reachable']=1
            elif p.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'target_reached')
        self.assertIs(p.environment.reward,p.root_reward_object);self.assertEqual(a.resets,1)
        rows=[v for v in p.rows if v.get('waypoint_goal') is not None]
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['action']['attack'],1)
        self.assertEqual(len(rows[0]['waypoint_goal']),6);self.assertIsNone(p.nav_head.goal)
        receipt=p.scheduler.state['plans'][0]['nodes'][0]['receipt']
        self.assertEqual(receipt['outcome'],'completed');self.assertFalse(receipt['root_completed'])
        self.assertTrue(p.scheduler.state['root']['completed'])

    def test_failed_attempt_enters_next_packet_and_repeat_alias_is_blocked(self):
        p,a,g,r,events,provider=self.case()
        self.assertNotIn('error',r);self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(provider.calls,2)
        self.assertEqual(len(p.guard.history.data['requests']),2)
        second=provider.packets[1]
        receipt=second['previous_plans'][0]['nodes'][0]['receipt']
        self.assertEqual(receipt['type'],'neural_waypoint');self.assertEqual(receipt['outcome'],'failed')
        self.assertIn('diagnosis',receipt['assessment'])
        self.assertTrue(any(c['skill_id'] in NAMES for c in second['memory']['blocked_methods']))
        self.assertFalse(p.scheduler.state['root']['completed'])
        self.assertEqual(p.scheduler.state['plans'][0]['nodes'][0]['receipt'],receipt)

    def test_stop_during_wait_reply_inference_and_execution(self):
        for kwargs in (dict(stop_wait=True),dict(stop_reply=True),dict(stop_recovery_inference=True),dict(stop_after=2)):
            with self.subTest(kwargs=kwargs):
                p,a,g,r,events,provider=self.case(**kwargs)
                self.assertNotIn('error',r);self.assertEqual(r['status'],'cancelled')
                self.assertEqual(g.state['held_key_count'],0);self.assertEqual(provider.calls,1)
                nav=[v for v in p.rows if v.get('waypoint_goal') is not None]
                self.assertEqual(len(nav),2 if 'stop_after' in kwargs else 0)
                receipts=[n['receipt'] for plan in p.scheduler.state['plans'] for n in plan['nodes'] if n['receipt']]
                self.assertTrue(all(v['outcome'] in ('interrupted','unrun') for v in receipts))

    def test_payload_carries_bound_candidates_and_does_not_accept_forged_targets(self):
        p,a,g,r,events,provider=self.case(stop_reply=True)
        self.assertNotIn('error',r);packet=provider.packet
        self.assertTrue(packet['spatial_memory']['waypoint_execution_supported'])
        self.assertEqual(len([c for c in packet['execution_registry'] if c in NAMES]),4)
        wire=json.loads(payload(packet)['input'][1]['content'])
        self.assertEqual(wire['waypoint_context'],packet['waypoint_context'])
        plan=provider.make_plan(packet)
        bad=deepcopy(packet);name=plan['steps'][0]['skill_id']
        bad['execution_registry'][name]['arguments']['target'][0]+=100
        with self.assertRaises(ValueError):validate_plan(plan,bad)
        bad=deepcopy(plan);bad['steps'][0]['completion']='goal_inventory'
        with self.assertRaises(ValueError):validate_plan(bad,packet)

    def test_arrival_alone_cannot_resume_unreachable_root(self):
        class NoApproach(FixedNavigationProvider):
            def make_plan(self,packet):
                value=super().make_plan(packet)
                value['steps']=[n for n in value['steps'] if n['skill_id'] not in ('locate_wood','approach_wood')]
                value['steps'][-1]['depends_on']=[value['steps'][0]['id']]
                return value
        def change(g):
            if g.policy.phase=='waypoint':
                g.state.update(zip(('server_x','server_y','server_z'),g.policy.navigation_cap['arguments']['target']))
                g.state['wood_v3_target_x']+=1
        p,a,g,r,events,provider=self.case(change=change,provider_factory=lambda *a:NoApproach(.001))
        self.assertNotIn('error',r);self.assertEqual(r['status'],'planning_paused')
        self.assertFalse(p.scheduler.state['root']['completed'])
        self.assertTrue(any(e.get('reason')=='missing_fresh_handoff' for e in p.planner_events))

    def test_refresh_position_change_rejects_old_coordinate_choice(self):
        def change(g):
            if g.policy.phase=='refresh':g.state['server_z']+=.6
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'planning_paused')
        self.assertFalse(any(v.get('waypoint_goal') is not None for v in p.rows))
        self.assertTrue(any(e.get('reason')=='stale_waypoint_pose' for e in p.planner_events))

    def test_all_blocked_candidates_pause_without_fabricating_a_route(self):
        with patch('minecraftrl.adaptive_route.AdaptiveRoute.blocked',return_value=True):
            p,a,g,r,events,provider=self.case()
        self.assertNotIn('error',r);self.assertEqual(r['status'],'planning_paused')
        self.assertFalse(any(v.get('waypoint_goal') is not None for v in p.rows))

    def test_successful_prefix_survives_approach_failure_and_new_plan(self):
        def change(g):
            if g.policy.phase=='waypoint':
                g.state.update(zip(('server_x','server_y','server_z'),g.policy.navigation_cap['arguments']['target']))
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r);self.assertEqual(provider.calls,2)
        old=provider.packets[1]['previous_plans'][0]['nodes']
        self.assertEqual(old[0]['status'],'completed')
        self.assertEqual(next(n['status'] for n in old if n['id']=='approach'),'failed')
        self.assertEqual(p.scheduler.state['plans'][0]['nodes'][0]['receipt'],old[0]['receipt'])
        self.assertEqual(p.scheduler.state['plans'][1]['nodes'][0]['status'],'completed')

    def test_already_satisfied_waypoint_is_zero_action_and_saved_in_receipt_file(self):
        class ArrivalBeforeWaypoint(FixedNavigationProvider):
            def make_plan(self,packet):
                value=super().make_plan(packet);leave=deepcopy(value['steps'][0])
                leave.update(id='leave',skill_id='leave_stalled_area',completion='displacement_3_blocks')
                value['steps'][0]['depends_on']=['leave'];value['steps'].insert(0,leave)
                return value
        def change(g):
            if g.policy.phase=='subgoal' and g.policy.active_plan_node['skill_id']=='leave_stalled_area':
                plan=g.policy.scheduler.state['plans'][-1]
                move=next(n for n in plan['proposal']['steps'] if n['id']=='move')
                target=plan['registry'][move['skill_id']]['arguments']['target']
                g.state.update(zip(('server_x','server_y','server_z'),target))
            elif g.policy.phase=='subgoal':g.state['wood_v3_reachable']=1
            elif g.policy.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change,provider_factory=lambda *a:ArrivalBeforeWaypoint(.001))
        self.assertNotIn('error',r);self.assertEqual(r['status'],'target_reached')
        receipts=p.navigation_receipts();self.assertEqual(len(receipts),1)
        self.assertTrue(receipts[0]['already_satisfied']);self.assertEqual(receipts[0]['committed_steps'],0)
        saved=json.loads((p.output/'navigation-receipts.json').read_text())
        self.assertEqual(saved,receipts)

    def test_unknown_target_after_movement_is_located_before_approach(self):
        def change(g):
            p=g.policy
            if p.phase=='collect':g.state.update(wood_v3_kind=0,wood_v3_reachable=0)
            elif p.phase=='waypoint':
                g.state.update(zip(('server_x','server_y','server_z'),p.navigation_cap['arguments']['target']))
            elif p.phase=='subgoal' and p.active_plan_node['skill_id']=='locate_wood':g.state['wood_v3_kind']=1
            elif p.phase=='subgoal' and p.active_plan_node['skill_id']=='approach_wood':g.state['wood_v3_reachable']=1
            elif p.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'target_reached')
        subgoals=[r['subgoal_context']['skill_id'] for r in p.rows if r.get('subgoal_context',{}).get('phase')=='subgoal']
        self.assertEqual(subgoals,['locate_wood','approach_wood'])

    def test_replanning_uses_execution_order_not_model_node_listing_order(self):
        class ReversedPlan(FixedNavigationProvider):
            def make_plan(self,packet):
                value=super().make_plan(packet);value['steps'].reverse();return value
        def change(g):
            if g.policy.phase=='waypoint':
                g.state.update(zip(('server_x','server_y','server_z'),g.policy.navigation_cap['arguments']['target']))
        p,a,g,r,events,provider=self.case(change=change,provider_factory=lambda *a:ReversedPlan(.001,max_calls=2))
        self.assertNotIn('error',r);self.assertEqual(provider.calls,2)
        latest=provider.packets[1]['evidence']
        self.assertEqual(latest['node_id'],'approach');self.assertEqual(latest['outcome'],'failed')


if __name__=='__main__':unittest.main()
