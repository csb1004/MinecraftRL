import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_rm_navigation
from minecraftrl.navigation_refinement import motion_goal


class RefinedBound(test_rm_navigation.Bound):
    def attach_navigation(self):
        super().attach_navigation();self.navigation_refined=True


class RefinedRMTests(unittest.TestCase):
    def case(self,**kwargs):
        with patch('test_rm_navigation.Bound',RefinedBound):
            return test_rm_navigation.RMNavigationTests.case(self,**kwargs)
    run_case=test_rm_navigation.RMNavigationTests.run_case

    def test_motion_and_remaining_budget_reach_actual_rm_action_then_root_clears(self):
        counts={}
        def change(g):
            p=g.policy
            if p.phase=='waypoint':
                key=p.scheduler.state['revision'];counts[key]=counts.get(key,0)+1
                if counts[key]==1:g.state['server_z']+=.1;g.state['yaw']+=3.
                else:g.state.update(zip(('server_x','server_y','server_z'),p.navigation_cap['arguments']['target']))
            elif p.phase=='subgoal':g.state['wood_v3_reachable']=1
            elif p.phase=='resumed':g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.case(change=change)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'target_reached')
        rows=[v for v in p.rows if 'waypoint_goal' in v];self.assertEqual(len(rows),2)
        for i,row in enumerate(rows):
            index=p.rows.index(row);previous=p.rows[index-1]['server_before'] if index else None
            self.assertEqual(row['waypoint_goal'],motion_goal(row['server_before'],row['waypoint_target'],previous,remaining=100-i))
        self.assertEqual(rows[1]['waypoint_goal'][-1],.99)
        self.assertNotEqual(rows[1]['waypoint_goal'][6:9],[0.,0.,0.])
        self.assertIsNone(p.nav_head.goal);self.assertIs(p.environment.reward,p.root_reward_object)

    def test_motion_budget_restarts_after_failed_plan_without_resetting_memory(self):
        p,a,g,r,events,provider=self.case()
        self.assertNotIn('error',r);self.assertEqual(provider.calls,2)
        first={}
        for row in p.rows:
            if 'waypoint_goal' in row:first.setdefault(row['rm_plan']['revision'],row['waypoint_goal'])
        self.assertEqual(len(first),2)
        self.assertTrue(all(len(v)==10 and v[-1]==1. for v in first.values()))
        self.assertEqual(a.resets,1)

    def test_refined_stop_does_not_dispatch_or_invent_navigation_failure(self):
        p,a,g,r,events,provider=self.case(stop_recovery_inference=True)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'cancelled')
        self.assertFalse(any('waypoint_goal' in v for v in p.rows))
        self.assertEqual(g.state['held_key_count'],0)
        self.assertTrue(all(v['outcome'] in ('unrun','interrupted') for v in p.navigation_receipts()))

    def test_rm_navigation_uses_checkpoint_client_pose_with_server_success_evidence(self):
        class ClientBound(RefinedBound):
            def attach_navigation(self):
                super().attach_navigation();self.navigation_pose_source='client'
        original=test_rm_navigation.test_planning_recovery.gui
        def observation():
            g=original();g.update(client_available=1,client_tick=g['server_tick']+50)
            g.update({'client_'+k:g['server_'+k]+(.2 if k=='z' else 0.) for k in 'xyz'})
            return g
        def change(g):
            if g.policy.phase=='waypoint':
                g.state.update(zip(('server_x','server_y','server_z'),g.policy.navigation_cap['arguments']['target']))
            elif g.policy.phase=='subgoal':g.state['wood_v3_reachable']=1
            elif g.policy.phase=='resumed':g.state['server_inventory']['oak_log']=1
            g.state['client_tick']=g.state['server_tick']+50
            g.state.update({'client_'+k:g.state['server_'+k]+(.2 if k=='z' else 0.) for k in 'xyz'})
        with patch('test_rm_navigation.Bound',ClientBound),patch('test_planning_recovery.gui',observation):
            p,a,g,r,events,provider=test_rm_navigation.RMNavigationTests.case(self,change=change)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'target_reached')
        rows=[v for v in p.rows if 'waypoint_goal' in v];self.assertEqual(len(rows),1)
        row=rows[0];index=p.rows.index(row);previous=p.rows[index-1]['server_before'] if index else None
        self.assertEqual(row['waypoint_goal'],motion_goal(row['server_before'],row['waypoint_target'],previous,pose_source='client'))
        self.assertNotEqual(row['waypoint_goal'],motion_goal(row['server_before'],row['waypoint_target'],previous))


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch unavailable')
class CheckpointBindingTests(unittest.TestCase):
    def test_refined_checkpoint_binds_parent_and_routes_root_without_motion(self):
        from minecraftrl.rm_navigation import RMNavigationController
        from minecraftrl.navigation_model import refinement_head,rescale_distance_input
        from minecraftrl.waypoint_actor import navigation_head
        from minecraftrl.world_store import digest
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__();self.buttons=torch.nn.Linear(2054,7);self.camera=torch.nn.Linear(2054,5)
            def forward(self,x):return {k:getattr(self,k)(x) for k in ('buttons','camera')}
        root=Head();head=refinement_head(navigation_head(root),'mlp');x=torch.randn(2,1,2054)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);parent=folder/'parent';parent.write_bytes(b'parent')
            embed=folder/'embed';embed.write_bytes(b'embed');path=folder/'navigation.pt'
            saved=dict(schema='navigation_refinement_v1',kind='mlp',motion_dim=10,
                parent_sha256=digest(parent),embedding_sha256=digest(embed),actor=head.state_dict())
            torch.save(saved,path)
            p=object.__new__(RMNavigationController);p.navigation_path=path
            p.student=SimpleNamespace(torch=torch,checkpoint=parent,embed_path=embed,
                agent=SimpleNamespace(policy=SimpleNamespace(pi_head=root)))
            p.attach_navigation();self.assertTrue(p.navigation_refined)
            for k,v in root(x).items():self.assertTrue(torch.equal(v,p.nav_head(x)[k]))
            p.nav_head.goal=[0.,1.,.2,0.,0.,1.,0.,0.,0.,1.]
            self.assertEqual(p.nav_head(x)['buttons'].shape,(2,1,7))
            p.student.agent.policy.pi_head=root
            saved.update(pose_source='client',coordinate_source='privileged_client');torch.save(saved,path)
            p.attach_navigation();self.assertEqual(p.navigation_pose_source,'client')
            p.student.agent.policy.pi_head=root
            rescale_distance_input(head,16)
            saved.update(schema='navigation_refinement_v2',residual_distance_scale=16,actor=head.state_dict())
            torch.save(saved,path);p.attach_navigation()
            p.nav_head.goal=[0.,1.,.2,0.,0.,1.,0.,0.,0.,1.]
            from minecraftrl.navigation_model import append_motion
            expected=head(append_motion(x,p.nav_head.goal))
            self.assertTrue(all(torch.equal(v,p.nav_head(x)[k]) for k,v in expected.items()))
            from minecraftrl.navigation_model import set_root_context_mode
            p.student.agent.policy.pi_head=root
            set_root_context_mode(head,'ignored')
            saved.update(schema='navigation_refinement_v3',root_context_mode='ignored',actor=head.state_dict())
            torch.save(saved,path);p.attach_navigation()
            self.assertEqual(p.navigation_root_context_mode,'ignored')
            for k,v in root(x).items():self.assertTrue(torch.equal(v,p.nav_head(x)[k]))
            p.nav_head.goal=[0.,1.,.2,0.,0.,1.,0.,0.,0.,1.]
            changed=x.clone();changed[...,2048:]=100
            expected=head(append_motion(x,p.nav_head.goal))
            self.assertTrue(all(torch.equal(v,p.nav_head(changed)[k]) for k,v in expected.items()))
            p.student.agent.policy.pi_head=root
            saved['coordinate_source']='privileged_server';torch.save(saved,path)
            with self.assertRaises(ValueError):p.attach_navigation()
            saved['coordinate_source']='privileged_client'
            saved['parent_sha256']='0'*64;torch.save(saved,path)
            with self.assertRaises(ValueError):p.attach_navigation()


if __name__=='__main__':unittest.main()
