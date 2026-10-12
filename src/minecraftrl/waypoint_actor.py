"""A separate neural navigation head; original root head and action vocabulary stay intact."""
import copy
import math
import torch
from .waypoints import GOAL_DIM


class GoalLinear(torch.nn.Module):
    def __init__(self,visual):
        super().__init__();self.visual=visual
        self.goal=torch.nn.Linear(GOAL_DIM,visual.out_features,bias=False).to(visual.weight)
        torch.nn.init.zeros_(self.goal.weight)

    def forward(self,x):
        return self.visual(x[...,:-GOAL_DIM].contiguous())+self.goal(x[...,-GOAL_DIM:].contiguous())


def navigation_head(parent):
    head=copy.deepcopy(parent);count=0
    def replace(module):
        nonlocal count
        for name,child in list(module.named_children()):
            if isinstance(child,torch.nn.Linear):
                if child.in_features!=2054:raise ValueError('unexpected navigation parent feature size')
                setattr(module,name,GoalLinear(child));count+=1
            else:replace(child)
    replace(head)
    if count!=2:raise ValueError('expected original button and camera heads')
    return head


def append_goal(features,goal):
    g=torch.as_tensor(goal,device=features.device,dtype=features.dtype)
    if g.ndim==1:g=g.view(1,1,GOAL_DIM)
    if g.ndim!=3 or g.shape[-1]!=GOAL_DIM or not torch.isfinite(g).all():raise ValueError('invalid goal tensor')
    return torch.cat([features,g.expand(features.shape[0],features.shape[1],GOAL_DIM)],-1)


def focus_goal_head(head,scale):
    if not math.isfinite(scale) or not 0<scale<=1:raise ValueError('invalid visual prior scale')
    for module in head.modules():
        if isinstance(module,GoalLinear):
            with torch.no_grad():
                module.visual.weight.mul_(scale)
                if module.visual.bias is not None:module.visual.bias.mul_(scale)
            module.visual.requires_grad_(False);module.goal.requires_grad_(True)


class SwitchHead(torch.nn.Module):
    def __init__(self,parent,navigation):
        super().__init__();self.root=parent;self.navigation=navigation;self.goal=None

    def forward(self,features):
        return self.root(features) if self.goal is None else self.navigation(append_goal(features,self.goal))

    def sample(self,*args,**kwargs):return self.root.sample(*args,**kwargs)
    def logprob(self,*args,**kwargs):return self.root.logprob(*args,**kwargs)
    def entropy(self,*args,**kwargs):return self.root.entropy(*args,**kwargs)
    def kl_divergence(self,*args,**kwargs):return self.root.kl_divergence(*args,**kwargs)
