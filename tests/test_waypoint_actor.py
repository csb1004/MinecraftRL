import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch unavailable')
class WaypointActorTests(unittest.TestCase):
    def test_goal_focus_scales_only_navigation_visual_prior_and_freezes_it(self):
        from minecraftrl.waypoint_actor import navigation_head,focus_goal_head,GoalLinear
        parent=self.head();nav=navigation_head(parent)
        original={k:v.clone() for k,v in parent.state_dict().items()}
        prior={name:m.visual.weight.clone() for name,m in nav.named_modules() if isinstance(m,GoalLinear)}
        focus_goal_head(nav,.1)
        for name,m in nav.named_modules():
            if isinstance(m,GoalLinear):
                self.assertTrue(torch.allclose(m.visual.weight,prior[name]*.1))
                self.assertFalse(m.visual.weight.requires_grad)
                self.assertTrue(m.goal.weight.requires_grad)
        for k,v in parent.state_dict().items():self.assertTrue(torch.equal(v,original[k]))
        with self.assertRaises(ValueError):focus_goal_head(nav,float('nan'))

    def head(self):
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__();self.buttons=torch.nn.Linear(2054,7);self.camera=torch.nn.Linear(2054,5)
            def forward(self,x):return {k:torch.log_softmax(getattr(self,k)(x),-1) for k in ('buttons','camera')}
        return Head()

    def test_zero_init_matches_parent_and_root_switch_is_exact(self):
        from minecraftrl.waypoint_actor import navigation_head,append_goal,SwitchHead,GoalLinear
        torch.manual_seed(1);parent=self.head();nav=navigation_head(parent)
        x=torch.randn(2,3,2054);goal=[1.,0.,.5,0.,0.,1.]
        original={k:v.clone() for k,v in parent.state_dict().items()}
        first=parent(x);zero=nav(append_goal(x,goal))
        for k in first:self.assertTrue(torch.equal(first[k],zero[k]))
        for m in nav.modules():
            if isinstance(m,GoalLinear):m.goal.weight.data[0,0]=2
        switch=SwitchHead(parent,nav)
        switch.goal=goal;left=switch(x);switch.goal=[-1.,0.,.5,0.,0.,1.];right=switch(x)
        self.assertGreater((left['camera']-right['camera']).abs().max(),.1)
        switch.goal=None
        for k,v in parent(x).items():self.assertTrue(torch.equal(v,switch(x)[k]))
        for k,v in parent.state_dict().items():self.assertTrue(torch.equal(v,original[k]))

    def test_goal_gradients_and_per_sample_goals_are_used(self):
        from minecraftrl.waypoint_actor import navigation_head,append_goal,GoalLinear
        head=navigation_head(self.head());x=torch.randn(2,2,2054)
        goals=torch.tensor([[[1.,0.,.1,0.,0.,1.],[-1.,0.,.2,0.,0.,1.]]])
        pd=head(append_goal(x,goals));(-pd['camera'][0,0,0]-pd['buttons'][0,1,1]).backward()
        for m in head.modules():
            if isinstance(m,GoalLinear):self.assertGreater(float(m.goal.weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):append_goal(x,[float('nan')]*6)


if __name__=='__main__':unittest.main()
