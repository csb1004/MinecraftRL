import unittest
from math import pi


def pose(x=0., z=0., yaw=0., tick=1):
    return dict(server_x=x, server_y=201., server_z=z, yaw=yaw, pitch=0.,
                server_tick=tick, server_available=1, error=0, screen=0)


class NavigationContractTests(unittest.TestCase):
    def test_short_correction_window_releases_input_before_teacher(self):
        from minecraftrl.navigation_refinement import TrialState
        s=TrialState([[0.,201.,3.]],'correction',student_ticks=8);s.start(pose())
        for i in range(8):s.advance(pose(tick=i+1),pose(tick=i+2))
        self.assertEqual(s.owner,'neutral');self.assertEqual(s.receipts,[])
        released=pose(tick=10);released['held_key_count']=0
        s.advance(pose(tick=9),released);self.assertEqual(s.owner,'teacher')
        with self.assertRaises(ValueError):TrialState([[0.,201.,3.]],'evaluate',student_ticks=8)

    def test_bc_may_mix_behavior_heads_only_with_same_frozen_feature_lineage(self):
        from minecraftrl.navigation_refinement import validate_bc_lineage
        data=dict(schema='navigation_trials_v1',split='train',eligible=True,rows=[dict(owner='teacher')])
        binding=dict(parent_sha256='a'*64,embedding_sha256='b'*64,model_sha256='c'*64)
        validate_bc_lineage(data,binding,binding)
        with self.assertRaises(ValueError):validate_bc_lineage(data,dict(binding,parent_sha256='d'*64),binding)
        with self.assertRaises(ValueError):validate_bc_lineage(dict(data,split='development'),binding,binding)
        with self.assertRaises(ValueError):validate_bc_lineage(data,{}, {})

    def test_stop_prevents_even_entering_navigation_inference(self):
        from minecraftrl.navigation_trial import NavigationTrial
        policy=object.__new__(NavigationTrial)
        policy.is_stopped=lambda:True
        with self.assertRaises(InterruptedError):policy.act(None)

    def test_cli_requires_isolated_navigation_inputs(self):
        from minecraftrl.cli import parser,validate_navigation_trial_args
        args=parser().parse_args(['episode','--policy','steve1','--rl-navigation-trial','results/trial.json'])
        with self.assertRaises(SystemExit):validate_navigation_trial_args(args)

    def test_student_failure_handoff_is_neutral_then_explicit_teacher(self):
        from minecraftrl.navigation_refinement import TrialState
        s=TrialState([[0.,201.,3.]],'correction');s.start(pose())
        for i in range(32):s.advance(pose(tick=i+1),pose(tick=i+2))
        self.assertEqual(s.owner,'neutral')
        g=pose(tick=34);g['held_key_count']=0
        s.advance(pose(tick=33),g)
        self.assertEqual(s.owner,'teacher')
        s.advance(g,pose(z=2.8,tick=35))
        self.assertTrue(s.finished)
        self.assertEqual(s.receipts[0]['owner'],'teacher')
        self.assertFalse(s.receipts[0]['student_success'])
        with self.assertRaises(RuntimeError):s.advance(pose(),pose(tick=2))

    def test_handoff_without_input_release_fails_closed(self):
        from minecraftrl.navigation_refinement import TrialState
        s=TrialState([[0.,201.,3.]],'correction');s.start(pose())
        for i in range(32):s.advance(pose(tick=i+1),pose(tick=i+2))
        g=pose(tick=34);g['held_key_count']=1
        with self.assertRaises(ValueError):s.advance(pose(tick=33),g)

    def test_task_budget_terminal_and_zero_action_skip_do_not_fake_success(self):
        from minecraftrl.navigation_refinement import TrialState
        s=TrialState([[0.,201.,0.],[0.,201.,3.]],'evaluate');s.start(pose())
        self.assertEqual(s.receipts[0]['outcome'],'already_satisfied')
        for i in range(100):r=s.advance(pose(tick=i+1),pose(tick=i+2))
        self.assertTrue(s.finished);self.assertEqual(r['task'],-1.)
        self.assertEqual(s.receipts[-1]['outcome'],'failed')

    def test_motion_is_previous_committed_tick_in_body_frame(self):
        from minecraftrl.navigation_refinement import motion_goal
        g=motion_goal(pose(z=.2,tick=2), [0.,201.,3.],pose(),remaining=75)
        self.assertEqual(len(g),10)
        self.assertAlmostEqual(g[6],0.)
        self.assertAlmostEqual(g[7],.2)
        self.assertAlmostEqual(g[9],.75)
        with self.assertRaises(ValueError):motion_goal(pose(tick=3),[0.,201.,3.],pose(),remaining=75)

    def test_reward_telescopes_and_true_terminal_zeroes_potential(self):
        from minecraftrl.navigation_refinement import navigation_reward
        points=[pose(z=z,tick=i) for i,z in enumerate((0.,.2,0.,.2,0.))]
        rows=[navigation_reward(a,b,[0.,201.,3.],terminal=False,success=False) for a,b in zip(points,points[1:])]
        shaped=sum(.99**i*r['shaping'] for i,r in enumerate(rows))
        self.assertAlmostEqual(shaped,.2*((.99**4-1)*(-3/4)))
        end=navigation_reward(points[-2],points[-1],[0.,201.,3.],terminal=True,success=False)
        self.assertEqual(end['potential_after'],0.)
        self.assertEqual(end['task'],-1.)
        with self.assertRaises(ValueError):navigation_reward(pose(),pose(tick=2),[0.,201.,3.],terminal=False,success=True)

    def test_train_contract_rejects_development_and_teacher_ppo(self):
        from minecraftrl.navigation_refinement import validate_dataset
        d=dict(schema='navigation_trials_v1',split='development',eligible=True,rows=[dict(owner='student')])
        with self.assertRaises(ValueError):validate_dataset(d,'ppo')
        d['split']='train';d['rows'][0]['owner']='teacher'
        with self.assertRaises(ValueError):validate_dataset(d,'ppo')

    def test_gae_does_not_propagate_reward_between_different_goals(self):
        from minecraftrl.navigation_refinement import navigation_returns
        rows=[dict(index=0,value=0.,navigation_reward=dict(total=1.,terminal=True)),
              dict(index=1,value=0.,navigation_reward=dict(total=-1.,terminal=True))]
        advantages,returns=navigation_returns(rows)
        self.assertEqual(advantages,[1.,-1.]);self.assertEqual(returns,[1.,-1.])
        rows[-1]['navigation_reward']['terminal']=False
        with self.assertRaises(ValueError):navigation_returns(rows)

    def test_correction_weights_raise_rare_turns_without_unbounded_single_label_weight(self):
        from minecraftrl.navigation_refinement import correction_weights
        goals=[[0.,1.,.2,0.,0.,1.,0.,0.,0.,1.]]*9+[[1.,0.,.02,0.,0.,1.,0.,0.,0.,1.]]
        weights=correction_weights(goals)
        self.assertGreater(weights[-1],weights[0])
        self.assertLessEqual(max(weights),3.)
        self.assertAlmostEqual(sum(weights),10.)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch unavailable')
class NavigationHeadTests(unittest.TestCase):
    def test_nonfinite_distribution_is_rejected_before_action_sampling(self):
        from types import SimpleNamespace
        from minecraftrl.navigation_trial import NavigationTrial
        from minecraftrl.navigation_refinement import TrialState
        class InvalidHead(torch.nn.Module):
            def forward(self,x):return dict(buttons=torch.full((2,1,2),float('nan')))
            def sample(self,*args,**kwargs):raise AssertionError('invalid distribution reached sampling')
        policy=object.__new__(NavigationTrial);policy.is_stopped=lambda:False;policy.ended=False
        policy.pending=None;policy.previous=None;policy.root_reward=object()
        policy.environment=SimpleNamespace(gui=pose(),reward=policy.root_reward)
        policy.state=TrialState([[0.,201.,3.]],'evaluate')
        policy.switch=InvalidHead()
        policy.student=SimpleNamespace(torch=torch,features=lambda screen:(torch.zeros(2,1,2054),None))
        with self.assertRaises(ValueError):policy.act(None)

    def test_new_head_initially_preserves_old_distribution_and_motion_changes_it(self):
        from minecraftrl.waypoint_actor import navigation_head,append_goal
        from minecraftrl.navigation_model import refinement_head,append_motion
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__();self.buttons=torch.nn.Linear(2054,7);self.camera=torch.nn.Linear(2054,5)
            def forward(self,x):return {k:torch.log_softmax(getattr(self,k)(x),-1) for k in ('buttons','camera')}
        torch.manual_seed(1);old=navigation_head(Head());x=torch.randn(2,1,2054);goal=[0.,1.,.2,0.,0.,1.,0.,.1,0.,1.]
        before={k:v.clone() for k,v in old.state_dict().items()}
        for kind in ('linear','mlp'):
            new=refinement_head(old,kind)
            expected=old(append_goal(x,goal[:6]));actual=new(append_motion(x,goal))
            for k in expected:self.assertTrue(torch.equal(expected[k],actual[k]))
            loss=-actual['buttons'][0,0,0]-actual['camera'][0,0,0];loss.backward()
            self.assertTrue(any(v.grad is not None and v.grad.abs().sum()>0 for v in new.parameters() if v.requires_grad))
        for k,v in old.state_dict().items():self.assertTrue(torch.equal(v,before[k]))


if __name__=='__main__':unittest.main()
