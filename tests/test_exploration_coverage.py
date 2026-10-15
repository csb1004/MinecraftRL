import unittest
from copy import deepcopy
from unittest.mock import patch
import test_rm_navigation
import test_planning_recovery

from minecraftrl.spatial_memory import SpatialMemory
from minecraftrl.exploration_coverage import score_candidates,coverage_delta


class CoverageTests(unittest.TestCase):
    def memory(self):
        return SpatialMemory('world','episode',source='privileged_server',dimension='overworld')

    def candidates(self):
        return [dict(target=t,cost=i*.01,admissible=True,failed=0)
                for i,t in enumerate(([3.5,201,.5],[.5,201,3.5]))]

    def test_unknown_path_preferred_to_repeated_path_with_equal_unknown_endpoints(self):
        m=self.memory()
        for i,x in enumerate((.5,2.5,.5,2.5,.5)):m.observe([x,201,.5],i+1)
        candidates=self.candidates();candidates[0]['target']=[4.5,201,.5]
        ranked,evidence=score_candidates(m,candidates)
        self.assertLess(ranked[1]['cost'],ranked[0]['cost'])
        self.assertEqual(evidence['traversability'],'unknown')
        self.assertGreater(evidence['candidates'][0]['repeated_crossings'],0)
        self.assertEqual(candidates[0]['cost'],0)  # Caller data remains immutable.

    def test_return_and_resource_do_not_penalize_familiar_routes(self):
        m=self.memory()
        for i,x in enumerate((.5,2.5,.5,2.5,.5)):m.observe([x,201,.5],i+1)
        m.set_landmark('home')
        for purpose in ('return','resource'):
            ranked,e=score_candidates(m,self.candidates(),purpose=purpose)
            self.assertEqual([x['cost'] for x in ranked],[0.,.01])
            self.assertEqual(e['purpose'],purpose)
        self.assertIn('home',m.landmarks)

    def test_blocked_candidate_stays_blocked_and_no_observation_is_invented(self):
        m=self.memory();m.observe([.5,201,.5],1)
        candidates=self.candidates();candidates[0]['admissible']=False
        before=deepcopy(m.summary());ranked,e=score_candidates(m,candidates)
        self.assertFalse(ranked[0]['admissible']);self.assertEqual(before,m.summary())
        self.assertTrue(all(x['estimated_unvisited_cells']>0 for x in e['candidates']))

    def test_invalid_or_discontinuous_observation_refuses_ranking(self):
        m=self.memory()
        with self.assertRaises(ValueError):score_candidates(m,self.candidates())
        m.observe([.5,201,.5],1);m.break_segment()
        with self.assertRaises(ValueError):score_candidates(m,self.candidates())

    def test_actual_coverage_is_distinct_from_arrival_and_revisits(self):
        m=self.memory();m.observe([.5,201,.5],1);before=m.summary()
        m.observe([2.5,201,.5],2);m.observe([.5,201,.5],3)
        receipt=coverage_delta(before,m.summary())
        self.assertEqual(receipt['new_cells'],1);self.assertEqual(receipt['new_directed_crossings'],2)
        self.assertFalse(receipt['root_success_claimed'])
        wrong=deepcopy(m.summary());wrong['world_id']='other'
        with self.assertRaises(ValueError):coverage_delta(before,wrong)


class CoverageBound(test_rm_navigation.Bound):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,navigation_strategy='coverage',**kwargs)


class CoverageIntegrationTests(unittest.TestCase):
    run_case=test_planning_recovery.PlanningRecoveryTests.run_case

    def case(self,**kwargs):
        with patch('minecraftrl.planning_recovery.PlanningRecoveryController',CoverageBound):
            return self.run_case(provider_factory=lambda *a:test_rm_navigation.FixedNavigationProvider(.001,max_calls=2),
                continue_after_return=True,**kwargs)

    def test_exploration_scores_reach_planner_and_actual_coverage_reaches_next_request(self):
        p,a,g,r,events,provider=self.case()
        self.assertNotIn('error',r);self.assertEqual(provider.calls,2)
        self.assertEqual(p.condition_metadata['planning']['navigation_strategy'],'coverage')
        first=provider.packets[0]['waypoint_context']
        self.assertEqual(first['exploration']['purpose'],'explore')
        self.assertEqual([c['cost'] for c in first['candidates']],
                         [c['cost'] for c in first['exploration']['candidates']])
        receipt=provider.packets[1]['previous_plans'][0]['nodes'][0]['receipt']
        self.assertEqual(receipt['exploration']['new_cells'],0)
        self.assertFalse(receipt['exploration']['root_success_claimed'])
        self.assertIs(p.environment.reward,p.root_reward_object);self.assertEqual(a.resets,1)

    def test_stop_before_navigation_dispatch_has_no_exploration_motion(self):
        p,a,g,r,events,provider=self.case(stop_recovery_inference=True)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'cancelled')
        self.assertEqual(g.state['held_key_count'],0)
        self.assertEqual([r for r in p.rows if 'waypoint_goal' in r],[])

    def test_actual_arrival_coverage_stays_separate_from_root_success(self):
        def change(g):
            p=g.policy
            if p.phase=='waypoint':g.state.update(zip(('server_x','server_y','server_z'),p.navigation_cap['arguments']['target']))
            elif p.phase=='subgoal':g.state['wood_v3_reachable']=1
            elif p.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'target_reached')
        receipt=p.navigation_receipts()[0]
        self.assertGreater(receipt['exploration']['new_cells'],0)
        self.assertFalse(receipt['root_completed']);self.assertTrue(p.scheduler.state['root']['completed'])

    def test_cli_coverage_requires_executable_rm_navigation(self):
        from minecraftrl.cli import parser,validate_waypoint_args
        from pathlib import Path
        args=parser().parse_args(['episode','--policy','steve1','--rl-navigation-strategy','coverage'])
        with self.assertRaises(SystemExit):validate_waypoint_args(args)
        args.rl_rm_navigation_actor=Path('results/nav.pt');validate_waypoint_args(args)


if __name__=='__main__':unittest.main()
