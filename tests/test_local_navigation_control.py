from copy import deepcopy
import unittest
from unittest.mock import patch

import test_planning_recovery
import test_rm_navigation
from test_local_navigation import observation


class LocalGame(test_planning_recovery.Game):
    def __init__(self,change=None):
        super().__init__(change)
        self.state.update(observation(target=True,tick=self.state['server_tick']))
    def step(self,action):
        obs,r,done,info=super().step(action)
        fresh=observation(self.state['server_x'],self.state['server_z'],self.state['server_tick'],target=True)
        self.state['local_navigation']=fresh['local_navigation']
        info['workshop']=deepcopy(self.state)
        return obs,r,done,info


class LocalBound(test_rm_navigation.Bound):
    def __init__(self,*a,**kw):super().__init__(*a,navigation_strategy='local_passability',**kw)


class AirborneGame(LocalGame):
    def __init__(self,change=None):
        super().__init__(change);self.state['server_y']=11.25


class LocalControlTests(unittest.TestCase):
    run_case=test_planning_recovery.PlanningRecoveryTests.run_case
    def case(self,**kwargs):
        with patch('test_planning_recovery.Game',LocalGame),patch('test_rm_navigation.Bound',LocalBound):
            return test_rm_navigation.RMNavigationTests.case(self,**kwargs)

    def test_route_condition_reaches_neural_input_and_preserves_root(self):
        def change(g):
            p=g.policy
            if p.phase=='waypoint':target=p.navigation_cap['arguments']['target']
            elif p.phase=='subgoal':target=p.local_approach_target
            else:target=None
            if target:
                g.state.update(zip(('server_x','server_y','server_z'),target))
                if target[0]>=4:g.state['wood_v3_reachable']=1
            if p.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'target_reached')
        local=[row for row in p.rows if 'local_route' in row]
        self.assertTrue(local)
        for row in local:
            self.assertIsNotNone(row['waypoint_goal'])
            self.assertEqual(row['embedding_sha256'],p.embed_hashes['wood'])
            self.assertEqual(row['subgoal_context']['skill_id'],'approach_wood')
        cap=provider.packet['execution_registry']['approach_wood']
        self.assertEqual(cap['arguments']['navigation_actor_sha256'],p.navigation_sha256)
        self.assertIs(p.environment.reward,p.root_reward_object)
        self.assertEqual(g.state['held_key_count'],0)

    def test_stop_during_neural_inference_dispatches_no_navigation(self):
        p,a,g,r,events,provider=self.case(stop_recovery_inference=True)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'cancelled')
        self.assertFalse(any('waypoint_goal' in row for row in p.rows))
        self.assertEqual(g.state['held_key_count'],0)

    def test_unknown_tree_search_uses_reachable_waypoints_until_observed(self):
        searched=[]
        def change(g):
            p=g.policy
            if p.phase=='collect':g.state.update(wood_v3_kind=0,wood_v3_reachable=0)
            if p.phase=='waypoint':target=p.navigation_cap['arguments']['target']
            elif p.phase=='subgoal':
                target=p.local_approach_target
                if p.active_plan_node['skill_id']=='locate_wood':
                    searched.append(target)
                    if len(searched)==2:g.state.update(wood_v3_kind=1,wood_v3_reachable=1)
            else:target=None
            if target:g.state.update(zip(('server_x','server_y','server_z'),target))
            if p.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'target_reached')
        rows=[x for x in p.rows if x.get('local_route',{}).get('stage')=='locate_wood']
        self.assertEqual(len(rows),2);self.assertTrue(all(x['waypoint_goal'] is not None for x in rows))
        self.assertTrue(all(x['action_temperature']==1. for x in rows))
        self.assertEqual(provider.packet['execution_registry']['locate_wood']['arguments']['navigation_actor_sha256'],p.navigation_sha256)

    def test_airborne_replan_keeps_supported_candidates_and_stop_boundary(self):
        with patch('test_planning_recovery.Game',AirborneGame),patch('test_rm_navigation.Bound',LocalBound):
            p,a,g,r,events,provider=test_rm_navigation.RMNavigationTests.case(self,stop_reply=True)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'cancelled')
        self.assertTrue(any(c['admissible'] for c in provider.packet['waypoint_context']['candidates']))
        self.assertEqual(provider.packet['runtime_binding']['navigation']['position'][1],11.25)
        self.assertTrue(all(c['target'][1]==10 for c in provider.packet['waypoint_context']['candidates'] if c['admissible']))
        self.assertFalse(any('waypoint_goal' in row for row in p.rows))


if __name__=='__main__':unittest.main()
