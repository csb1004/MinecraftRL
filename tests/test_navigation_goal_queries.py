import copy
import importlib.util
import math
import unittest
from minecraftrl.navigation_goal_queries import goal_variation_requests,supervision_goal
from minecraftrl.waypoints import goal_features
from test_navigation_supervision import sample


def fixture():
    data,observed=sample()
    g=observed[0]['server_before'];g['server_on_ground']=1
    goal=goal_features(g,observed[0]['target'])+[.03,.07,.01,.02]
    data['rows'][0]['motion_goal']=goal;observed[0]['goal']=list(goal)
    return data,observed


class GoalVariationContracts(unittest.TestCase):
    def test_same_real_state_has_distinct_near_stop_and_far_forward_targets(self):
        data,observed=fixture();before=copy.deepcopy((data,observed))
        requests,receipt=goal_variation_requests(data,observed)
        near=next(r for r in requests if r['bearing']==0 and r['distance']==.1)
        far=next(r for r in requests if r['bearing']==0 and r['distance']==1.5)
        self.assertEqual(near['changes'],{})
        self.assertEqual(far['changes']['forward'],1)
        self.assertEqual(near['motion_goal'][6:9],data['rows'][0]['motion_goal'][6:9])
        self.assertEqual(near['motion_goal'][-1],1.)
        self.assertEqual((data,observed),before)
        self.assertEqual(receipt['selected_rows'],[0])

    def test_all_counterfactuals_stay_in_the_known_convex_domain(self):
        data,observed=fixture();requests,_=goal_variation_requests(data,observed)
        rays={}
        for row in requests:
            x,y,z=row['target']
            self.assertTrue(1.<x<6.7 and .5<z<15.5 and y==201.)
            rays.setdefault(row['bearing'],[]).append(row['distance'])
            self.assertAlmostEqual(row['motion_goal'][2]*16,row['distance'])
            self.assertAlmostEqual(math.degrees(math.atan2(*row['motion_goal'][:2])),row['bearing'])
        self.assertTrue(all(any(d>=.6 for d in distances) for distances in rays.values()))

    def test_goal_feet_never_overlap_actual_wall_columns_near_perimeter(self):
        for x,z in ((1.31,1.31),(6.69,14.69),(4.,14.69)):
            data,observed=fixture();g=observed[0]['server_before'];g.update(server_x=x,server_z=z)
            goal=goal_features(g,observed[0]['target'])+[0.,0.,0.,1.]
            data['rows'][0]['motion_goal']=goal;observed[0]['goal']=list(goal)
            requests,_=goal_variation_requests(data,observed)
            self.assertTrue(requests)
            for request in requests:
                tx,_,tz=request['target']
                for bx in range(math.floor(tx-.3+1e-7),math.floor(tx+.3-1e-7)+1):
                    for bz in range(math.floor(tz-.3+1e-7),math.floor(tz+.3-1e-7)+1):
                        self.assertNotIn(bx,(0,7,15));self.assertNotIn(bz,(0,15))
                        self.assertTrue(0<=bx<16 and 0<=bz<16)

    def test_missing_ground_and_development_and_offline_rows_are_rejected(self):
        data,observed=fixture();del observed[0]['server_before']['server_on_ground']
        requests,receipt=goal_variation_requests(data,observed)
        self.assertFalse(requests);self.assertEqual(receipt['excluded']['unverified_ground'],1)
        for change in ('development','offline'):
            data,observed=fixture()
            if change=='development':data['split']='development'
            else:data['rows'][0]['bc_motion_goal']=[0.]*10
            with self.assertRaises(ValueError):goal_variation_requests(data,observed)

    def test_geometry_is_bound_to_original_observation_and_future_is_unused(self):
        data,observed=fixture();expected=goal_variation_requests(data,observed)
        observed[0]['server_after']={'server_x':999.};data['rows'][0]['navigation_reward']={'success':True}
        self.assertEqual(goal_variation_requests(data,observed),expected)
        data['rows'][0]['motion_goal'][2]=observed[0]['goal'][2]=.9
        with self.assertRaises(ValueError):goal_variation_requests(data,observed)

    def test_selection_is_bounded_unique_and_reproducible(self):
        data,observed=fixture();data['rows']=[copy.deepcopy(data['rows'][0]) for _ in range(40)]
        observed=[copy.deepcopy(observed[0]) for _ in range(40)]
        first=goal_variation_requests(data,observed);second=goal_variation_requests(data,observed)
        self.assertEqual(first,second)
        selected=first[1]['selected_rows'];self.assertEqual(len(selected),12)
        self.assertEqual(len(set(selected)),12);self.assertEqual((selected[0],selected[-1]),(0,39))

    def test_supervised_goal_requires_a_separate_label_and_finite_shape(self):
        row=dict(motion_goal=[0.]*10)
        self.assertIs(supervision_goal(row),row['motion_goal'])
        for goal in ([0.]*9,[float('nan')]*10):
            with self.assertRaises(ValueError):supervision_goal(dict(row,bc_motion_goal=goal,bc_action={}))
        with self.assertRaises(ValueError):supervision_goal(dict(row,bc_motion_goal=[1.]*10))

    def test_client_queries_use_current_pose_and_require_client_ground_evidence(self):
        data,observed=fixture();data['config']['pose_source']='client';g=observed[0]['server_before']
        g.update(client_available=1,client_tick=7,client_x=4.4,client_y=201.,client_z=4.,client_on_ground=1)
        goal=goal_features(g,observed[0]['target'],pose_source='client')+[.03,.07,.01,.02]
        data['rows'][0]['motion_goal']=goal;observed[0]['goal']=list(goal)
        requests,receipt=goal_variation_requests(data,observed)
        near=next(r for r in requests if r['bearing']==0 and r['distance']==.1)
        self.assertAlmostEqual(near['target'][0],4.4)
        self.assertAlmostEqual(near['target'][2],3.9)
        self.assertEqual(receipt['pose_source'],'client')
        del g['client_on_ground']
        self.assertEqual(goal_variation_requests(data,observed)[0],[])

    def test_offline_supervision_cannot_be_reintroduced_as_on_policy_data(self):
        from minecraftrl.navigation_refinement import validate_dataset
        for key in ('bc_action','bc_motion_goal'):
            data,_=fixture();data['rows'][0][key]=[]
            with self.assertRaises(ValueError):validate_dataset(data,'ppo')


@unittest.skipIf(importlib.util.find_spec('minerl') is None,'official WSL mapper required')
class GoalVariationMappingTests(unittest.TestCase):
    def test_mapping_keeps_original_action_logp_features_and_goal_intact(self):
        import sys,torch
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/steve1'))
        from steve1.VPT.lib.actions import ActionTransformer
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
        from minecraftrl.navigation_goal_queries import label_goal_variations
        data,observed=fixture();original=data['rows'][0]
        original.update(action=dict(buttons=torch.zeros(1,1,1,dtype=torch.long),camera=torch.zeros(1,1,1,dtype=torch.long)),
            old_logp=-7.,features=torch.randn(2,1,2054))
        old_goal=list(original['motion_goal'])
        mapper=CameraHierarchicalMapping(n_camera_bins=11);transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
        labeled,receipt=label_goal_variations(data,observed,mapper,transformer)
        for row in labeled:
            self.assertIs(row['action'],original['action']);self.assertIs(row['features'],original['features'])
            self.assertEqual(row['old_logp'],-7.);self.assertEqual(row['motion_goal'],old_goal)
        self.assertNotIn('bc_action',original);self.assertNotIn('bc_motion_goal',original)
        self.assertEqual(receipt['executed_counterfactual_goals'],0)
        self.assertTrue(receipt['official_mapping_roundtrip_verified'])


if __name__=='__main__':unittest.main()
