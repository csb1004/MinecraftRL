"""Frozen STEVE-1 navigation prior plus comparable linear/nonlinear residuals."""
import copy
import torch
from .waypoint_actor import GoalLinear,navigation_head


class RefinedLinear(torch.nn.Module):
    def __init__(self,base,kind):
        super().__init__();self.base=base.requires_grad_(False);self.kind=kind
        if kind=='linear':self.residual=torch.nn.Linear(10,base.visual.out_features,bias=False)
        elif kind=='mlp':
            self.visual_projection=torch.nn.Sequential(torch.nn.Linear(2054,16),torch.nn.Tanh())
            self.residual=torch.nn.Sequential(torch.nn.Linear(26,64),torch.nn.Tanh(),torch.nn.Linear(64,base.visual.out_features))
        else:raise ValueError('unknown refinement head')
        self.to(base.visual.weight)
        last=self.residual if kind=='linear' else self.residual[-1]
        torch.nn.init.zeros_(last.weight)
        if last.bias is not None:torch.nn.init.zeros_(last.bias)

    def forward(self,x):
        goal=x[...,-10:];visual=x[...,:-10]
        inputs=goal if self.kind=='linear' else torch.cat([self.visual_projection(visual),goal],-1)
        return self.base(torch.cat([visual,goal[...,:6]],-1))+self.residual(inputs)


def refinement_head(prior,kind):
    head=copy.deepcopy(prior).requires_grad_(False);count=0
    def replace(module):
        nonlocal count
        for name,child in list(module.named_children()):
            if isinstance(child,GoalLinear):setattr(module,name,RefinedLinear(child,kind));count+=1
            else:replace(child)
    replace(head)
    if count!=2:raise ValueError('expected two navigation output heads')
    return head


def append_motion(features,goal):
    g=torch.as_tensor(goal,device=features.device,dtype=features.dtype)
    if g.ndim==1:g=g.view(1,1,10)
    if g.ndim!=3 or g.shape[-1]!=10 or not torch.isfinite(g).all():raise ValueError('invalid motion goal')
    return torch.cat([features,g.expand(features.shape[0],features.shape[1],10)],-1)


def load_navigation(parent,saved):
    old=navigation_head(parent)
    if saved['schema']=='waypoint_actor_v1':old.load_state_dict(saved['actor']);return old,False
    if saved['schema']!='navigation_refinement_v1':raise ValueError('unknown navigation checkpoint')
    head=refinement_head(old,saved['kind']);head.load_state_dict(saved['actor']);return head,True


def value_head():
    return torch.nn.Sequential(torch.nn.Linear(2064,64),torch.nn.Tanh(),torch.nn.Linear(64,1))


class NavigationSwitch(torch.nn.Module):
    def __init__(self,parent,head,refined):
        super().__init__();self.root=parent;self.navigation=head;self.refined=refined;self.goal=None
    def forward(self,features):
        from .waypoint_actor import append_goal
        if self.goal is None:return self.root(features)
        return self.navigation(append_motion(features,self.goal) if self.refined else append_goal(features,self.goal[:6]))
    def sample(self,*args,**kwargs):return self.root.sample(*args,**kwargs)
    def logprob(self,*args,**kwargs):return self.root.logprob(*args,**kwargs)
    def entropy(self,*args,**kwargs):return self.root.entropy(*args,**kwargs)
    def kl_divergence(self,*args,**kwargs):return self.root.kl_divergence(*args,**kwargs)
