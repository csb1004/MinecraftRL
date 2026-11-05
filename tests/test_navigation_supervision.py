import copy
import importlib.util
import unittest
from minecraftrl.navigation_supervision import expert_requests


def sample(owner='student'):
    action={'camera': [0., 0.], 'forward': 1}
    goal=[0., 1., .2, 0., 0., 1., 0., .1, 0., 1.]
    row=dict(owner=owner,index=0,motion_goal=goal,env_action=action,interrupted=False)
    actual=dict(owner=owner,index=0,goal=goal,action=action,interrupted=False,target=[4.,201.,6.],
        server_before=dict(server_x=4.,server_y=201.,server_z=4.,yaw=180.,pitch=-13.,
            server_available=1,error=0,screen=0))
    data=dict(schema='navigation_trials_v1',split='train',eligible=True,rows=[row],
        world=dict(experimental_split='train',kind='recovery',chunk=[0,0],curriculum=dict(mirror_z=False)),
        config=dict(points=[[4.,6.]]))
    return data,[actual]


class ExpertQueryContracts(unittest.TestCase):
    def test_student_failure_gets_turn_label_without_rewriting_executed_action(self):
        data,observed=sample();before=copy.deepcopy((data,observed))
        requests,excluded=expert_requests(data,observed)
        self.assertEqual(excluded,{})
        self.assertEqual(requests[0]['changes']['forward'],0)
        self.assertEqual(abs(requests[0]['changes']['camera'][1]),10)
        self.assertEqual((data,observed),before)

    def test_development_even_with_training_flag_cannot_supply_labels(self):
        for scope in ('split','world'):
            data,observed=sample()
            if scope=='split':data['split']='development'
            else:data['world']['experimental_split']='development'
            with self.assertRaises(ValueError):expert_requests(data,observed)

    def test_expert_does_not_consult_future_state_or_outcome(self):
        data,observed=sample();expected=expert_requests(data,observed)
        observed[0].update(server_after={'server_x':999},root_reward=1000)
        data['rows'][0]['navigation_reward']={'success':True}
        self.assertEqual(expert_requests(data,observed),expected)

    def test_expert_abstains_at_wall_or_height_or_gui_and_neutral_handoff(self):
        for key,value in (('server_x',8.),('server_y',200.),('screen',1)):
            data,observed=sample();observed[0]['server_before'][key]=value
            requests,excluded=expert_requests(data,observed)
            self.assertEqual(requests,[]);self.assertEqual(excluded,{'outside_expert_domain':1})
        data,observed=sample('neutral')
        self.assertEqual(expert_requests(data,observed),([],{'neutral_handoff':1}))

    def test_labels_cannot_be_shifted_to_different_observation_or_target(self):
        for key,value in (('owner','teacher'),('target',[3.,201.,6.]),('goal',[1.]*10)):
            data,observed=sample();observed[0][key]=value
            with self.assertRaises(ValueError):expert_requests(data,observed)
        data,observed=sample();observed[0]['server_before']['yaw']=float('nan')
        with self.assertRaises(ValueError):expert_requests(data,observed)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None or importlib.util.find_spec('minerl') is None,
    'isolated WSL Torch/MineRL environment required')
class OfficialExpertMappingTests(unittest.TestCase):
    def test_actual_action_and_logp_preserved_teacher_parity_fail_closed(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/steve1'))
        from steve1.VPT.lib.actions import ActionTransformer
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
        from minecraftrl.navigation_supervision import label_training_states
        mapper=CameraHierarchicalMapping(n_camera_bins=11)
        transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
        data,observed=sample();original=dict(buttons=torch.tensor([[[0]]]),camera=torch.tensor([[[0]]]))
        data['rows'][0].update(action=original,old_logp=-3.)
        rows,receipt=label_training_states(data,observed,mapper,transformer)
        self.assertIs(rows[0]['action'],original);self.assertEqual(rows[0]['old_logp'],-3.)
        self.assertNotIn('bc_action',data['rows'][0])
        self.assertTrue(receipt['official_mapping_roundtrip_verified'])
        data['rows'][0]['owner']=observed[0]['owner']='teacher'
        with self.assertRaises(ValueError):label_training_states(data,observed,mapper,transformer)
        data['rows'][0]['action']=rows[0]['bc_action']
        _,receipt=label_training_states(data,observed,mapper,transformer)
        self.assertTrue(receipt['teacher_parity_verified'])


if __name__=='__main__':unittest.main()
