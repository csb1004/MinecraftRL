import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import test_planning_recovery

from minecraftrl.subgoal_control import NeuralSubgoalController, FixedSubgoalProvider
from minecraftrl.subgoal_skills import SKILLS


def catalogue():
    result={}
    for i,name in enumerate(SKILLS):
        value=np.full((1,512),i+1,dtype=np.float32)
        result[name]=dict(embedding=value,sha256=hashlib.sha256(value.tobytes()).hexdigest(),provenance={'fixture':True})
    return result


class Bound(NeuralSubgoalController):
    def __init__(self,*args,**kwargs):super().__init__(*args,catalogue=catalogue(),**kwargs)


class SubgoalControlTests(unittest.TestCase):
    run_case=test_planning_recovery.PlanningRecoveryTests.run_case

    def case(self,provider_factory=None,**kwargs):
        with patch('minecraftrl.planning_recovery.PlanningRecoveryController',Bound):
            return self.run_case(provider_factory=provider_factory or (lambda *a:FixedSubgoalProvider(.001)),
                                 continue_after_return=True,**kwargs)

    def test_spatial_history_reaches_real_packet_without_motor_capability_promotion(self):
        from copy import deepcopy
        from minecraftrl.rm_planner import validate_packet,payload
        import json
        p,a,g,r,events,provider=self.case(stop_reply=True)
        self.assertNotIn('error',r)
        packet=provider.packet;space=packet['spatial_memory']
        self.assertEqual(space['world_id'],packet['runtime_binding']['world_id'])
        self.assertEqual(space['last_tick'],packet['runtime_binding']['tick'])
        self.assertEqual(space['source'],'privileged_server')
        self.assertEqual(space['samples'],packet['runtime_binding']['sequence']+1)
        self.assertFalse(space['waypoint_execution_supported'])
        self.assertNotIn('navigate_to_waypoint',packet['execution_registry'])
        transmitted=json.loads(payload(packet)['input'][1]['content'])
        self.assertEqual(transmitted['spatial_memory'],space)
        self.assertTrue((p.output/'spatial-memory.json').exists())
        bad=deepcopy(packet);bad['spatial_memory']['world_id']='different'
        with self.assertRaises(ValueError):validate_packet(bad)

    def test_actual_option_switches_and_server_completion_preserve_root(self):
        def change(g):
            p=g.policy
            if p.phase=='subgoal':
                skill=p.active_plan_node['skill_id']
                if skill=='leave_stalled_area':g.state['server_z']+=1.
                elif skill=='approach_wood':g.state['wood_v3_reachable']=1
            elif p.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r)
        self.assertEqual(r['status'],'target_reached')
        self.assertEqual(provider.calls,1)
        self.assertTrue(p.scheduler.state['root']['completed'])
        self.assertIs(p.environment.reward,p.root_reward_object)
        self.assertEqual([s['embedding_sha256'] for s in p.switches],
                         [catalogue()['leave_stalled_area']['sha256'],catalogue()['approach_wood']['sha256'],p.embed_hashes['wood']])
        self.assertEqual(a.resets,1)
        self.assertTrue(all(x['subgoal_context']==y['subgoal_context'] for x,y in zip(p.rows,a.rows)))
        self.assertEqual(g.state['held_key_count'],0)

    def test_failed_skills_are_in_next_packet_and_cannot_repeat(self):
        p,a,g,r,events,provider=self.case()
        self.assertNotIn('error',r)
        self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(provider.calls,3)
        receipts=[n['receipt'] for plan in p.scheduler.state['plans'] for n in plan['nodes'] if n['receipt']]
        self.assertEqual([r['method']['skill_id'] for r in receipts],['leave_stalled_area','approach_wood'])
        self.assertTrue(all(r['outcome']=='failed' and r['committed_steps']==200 for r in receipts))
        blocked={c['skill_id'] for c in provider.packets[-1]['memory']['blocked_methods']}
        self.assertTrue({'leave_stalled_area','approach_wood','explore_then_resume'}<=blocked)
        self.assertEqual(provider.packets[1]['evidence']['event'],'subgoal_ended')
        self.assertEqual(len(p.switches),2)
        self.assertFalse(p.scheduler.state['root']['completed'])

    def test_unknown_target_executes_locate_before_approach(self):
        def change(g):
            p=g.policy
            if not p.switches:g.state['wood_v3_kind']=0
            if p.phase=='subgoal':
                name=p.active_plan_node['skill_id']
                if name=='leave_stalled_area':g.state['server_x']+=1
                if name=='locate_wood':g.state['wood_v3_kind']=1
                if name=='approach_wood':g.state['wood_v3_reachable']=1
            if p.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r)
        self.assertEqual(r['status'],'target_reached')
        self.assertEqual([e['node']['skill_id'] for e in p.planner_events if e['event']=='subgoal_started'],list(SKILLS))

    def test_satisfied_approach_is_skipped_without_fake_actions(self):
        def change(g):
            if g.policy.phase=='subgoal':
                g.state['server_x']+=1;g.state['wood_v3_reachable']=1
            if g.policy.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r)
        receipts=[n['receipt'] for n in p.scheduler.state['plans'][0]['nodes']]
        self.assertTrue(receipts[1]['already_satisfied'])
        self.assertEqual(receipts[1]['committed_steps'],0)
        self.assertEqual(len(p.switches),2)

    def test_smaller_planned_budget_and_stop_boundaries(self):
        class Short(FixedSubgoalProvider):
            def poll(self):
                value=super().poll()
                if value:
                    for n in value['steps']:
                        if n['id']!='root':n['budget_ticks']=7
                return value
        p,a,g,r,events,provider=self.case(provider_factory=lambda *a:Short(.001))
        self.assertNotIn('error',r)
        self.assertEqual(sum(row['mode']=='intervention' for row in p.rows),14)
        for kwargs in (dict(stop_wait=True),dict(stop_reply=True),dict(stop_recovery_inference=True),dict(stop_after=2)):
            with self.subTest(kwargs=kwargs):
                p,a,g,r,events,provider=self.case(**kwargs)
                self.assertNotIn('error',r)
                self.assertEqual(r['status'],'cancelled');self.assertEqual(g.state['held_key_count'],0)
                self.assertEqual(sum(row['mode']=='intervention' for row in p.rows),2 if 'stop_after' in kwargs else 0)

    def test_changed_completion_or_missing_state_is_rejected(self):
        from minecraftrl.rm_planner import validate_plan
        from minecraftrl.subgoal_planner import subgoal_fixture
        from copy import deepcopy
        p,a,g,r,events,provider=self.case(stop_reply=True)
        packet=provider.packet;plan=subgoal_fixture(packet)
        broken=deepcopy(plan);broken['steps'][0]['completion']='goal_inventory'
        with self.assertRaises(ValueError):validate_plan(broken,packet)
        broken=deepcopy(packet);broken['runtime_binding']['navigation']['tick']+=1
        with self.assertRaises(ValueError):validate_plan(plan,broken)

    def test_provider_can_omit_exploration_and_choose_approach_as_first_condition(self):
        class Direct(FixedSubgoalProvider):
            def poll(self):
                value=super().poll()
                if value:
                    value['steps']=[n for n in value['steps'] if n['skill_id']!='leave_stalled_area']
                    value['steps'][0]['depends_on']=[]
                return value
        def change(g):
            if g.policy.phase=='subgoal':g.state['wood_v3_reachable']=1
            if g.policy.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change,provider_factory=lambda *a:Direct(.001))
        self.assertNotIn('error',r)
        self.assertEqual(r['status'],'target_reached')
        self.assertEqual(p.switches[0]['embedding_sha256'],catalogue()['approach_wood']['sha256'])
        self.assertEqual(sum(row['mode']=='intervention' for row in p.rows),1)

    def test_root_policy_can_open_and_close_inventory_before_planning(self):
        def change(g):
            if len(g.sent)==70:g.state['screen']=1
            if len(g.sent)==72:g.state['screen']=0
            if len(g.sent)==74:g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r)
        self.assertEqual(r['status'],'target_reached')
        self.assertTrue(any(row['subgoal_context']['abstract_state']['phase']=='gui_open' for row in p.rows))

    def test_legacy_exploration_restores_its_own_embedding_after_new_option(self):
        class Mixed(FixedSubgoalProvider):
            def poll(self):
                from minecraftrl.rm_planner import fixture_plan
                value=super().poll()
                if not value:return value
                if self.calls==1:
                    value['steps']=[n for n in value['steps'] if n['skill_id']!='leave_stalled_area']
                    value['steps'][0].update(depends_on=[],budget_ticks=3)
                else:value=fixture_plan(self.packet,unsupported=self.calls>2)
                return value
        p,a,g,r,events,provider=self.case(provider_factory=lambda *a:Mixed(.001))
        self.assertNotIn('error',r)
        self.assertEqual([s['embedding_sha256'] for s in p.switches],
            [catalogue()['approach_wood']['sha256'],catalogue()['leave_stalled_area']['sha256']])
        self.assertEqual(p.condition_metadata['intervention']['embedding_sha256']['intervention'],
                         catalogue()['leave_stalled_area']['sha256'])

    def test_displacement_does_not_authorize_unreachable_root(self):
        class NoApproach(FixedSubgoalProvider):
            def poll(self):
                value=super().poll()
                if value:
                    value['steps']=[n for n in value['steps'] if n['skill_id']!='approach_wood']
                    value['steps'][-1]['depends_on']=['leave_stalled_area']
                return value
        def change(g):
            if g.policy.phase=='subgoal':
                g.state['server_z']+=1.
                g.state['wood_v3_target_x']+=1  # New target avoids the separate same-method block.
        p,a,g,r,events,provider=self.case(change=change,provider_factory=lambda *a:NoApproach(.001))
        self.assertNotIn('error',r)
        self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(len(p.switches),1)
        self.assertTrue(any(e.get('reason')=='missing_fresh_handoff' for e in p.planner_events))
        self.assertFalse(p.scheduler.state['root']['completed'])

    def test_expiry_before_first_skill_action_is_unrun_not_failed(self):
        class Expired(Bound):
            def prepare_control(self,**kwargs):
                if self.phase=='subgoal' and self.subgoal_start is not None:
                    original=kwargs['elapsed'];kwargs['elapsed']=lambda:original()+31.
                    self.elapsed=kwargs['elapsed']
                return super().prepare_control(**kwargs)
        with patch('minecraftrl.planning_recovery.PlanningRecoveryController',Expired):
            p,a,g,r,events,provider=self.run_case(provider_factory=lambda *a:FixedSubgoalProvider(.001),
                                                continue_after_return=True)
        self.assertNotIn('error',r)
        self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(provider.calls,1)
        receipt=p.scheduler.state['plans'][0]['nodes'][0]['receipt']
        self.assertEqual(receipt['outcome'],'unrun');self.assertEqual(receipt['committed_steps'],0)
        self.assertEqual(len(p.switches),0)


if __name__=='__main__':unittest.main()
