"""Frozen STEVE-1 navigation prior plus comparable linear/nonlinear residuals."""
import copy
import math
import torch
from .waypoint_actor import GoalLinear,navigation_head


def checked_distance_scale(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 1 <= value <= 64:
        raise ValueError('residual distance scale must be finite and within 1..64')
    return float(value)


def checked_root_context_mode(value):
    if value not in ('observed','ignored'):
        raise ValueError('root context mode must be observed or ignored')
    return value


def checked_visual_adapter_width(value):
    if type(value) is not int or value not in (0,64):
        raise ValueError('visual adapter width must be 0 or 64')
    return value


class VisualResidualAdapter(torch.nn.Module):
    """Normalized visual path with zero initial output; no running statistics."""
    def __init__(self,out_features):
        super().__init__()
        # Training-only reparameterization; fold into weights before v4 export.
        self.training_spatial_scale=1
        # Explicit CPU factories and an isolated CPU generator preserve both CPU/CUDA
        # sampling state while constructing this additional experimental branch.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(17599).get_state())
            self.normalization=torch.nn.LayerNorm(2048,elementwise_affine=False,device='cpu')
            self.network=torch.nn.Sequential(torch.nn.Linear(2058,64,device='cpu'),torch.nn.GELU(),
                torch.nn.Linear(64,out_features,device='cpu'))
            torch.nn.init.zeros_(self.network[-1].weight);torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self,image,conditioned_goal):
        if self.training_spatial_scale!=1:
            conditioned_goal=torch.cat([conditioned_goal[...,:4]*self.training_spatial_scale,
                                       conditioned_goal[...,4:]],-1)
        return self.network(torch.cat([self.normalization(image),conditioned_goal],-1))


class RefinedLinear(torch.nn.Module):
    def __init__(self,base,kind,distance_scale=1,root_context_mode='observed',visual_adapter_width=0):
        super().__init__();self.base=base.requires_grad_(False);self.kind=kind
        self.distance_scale=checked_distance_scale(distance_scale)
        self.root_context_mode=checked_root_context_mode(root_context_mode)
        self.visual_adapter_width=checked_visual_adapter_width(visual_adapter_width)
        if self.visual_adapter_width and (kind!='mlp' or self.root_context_mode!='ignored'):
            raise ValueError('visual adapter requires ignored-root MLP navigation')
        if kind=='linear':self.residual=torch.nn.Linear(10,base.visual.out_features,bias=False)
        elif kind=='mlp':
            self.visual_projection=torch.nn.Sequential(torch.nn.Linear(2054,16),torch.nn.Tanh())
            self.residual=torch.nn.Sequential(torch.nn.Linear(26,64),torch.nn.Tanh(),torch.nn.Linear(64,base.visual.out_features))
        else:raise ValueError('unknown refinement head')
        self.to(base.visual.weight)
        last=self.residual if kind=='linear' else self.residual[-1]
        torch.nn.init.zeros_(last.weight)
        if last.bias is not None:torch.nn.init.zeros_(last.bias)
        if self.visual_adapter_width:
            self.visual_adapter=VisualResidualAdapter(base.visual.out_features).to(base.visual.weight)

    def forward(self,x):
        goal=x[...,-10:];visual=x[...,:-10]
        if self.root_context_mode=='ignored':goal=goal.contiguous()
        conditioned=goal
        if self.distance_scale!=1:
            conditioned=torch.cat([goal[...,:2],goal[...,2:3]*self.distance_scale,goal[...,3:]],-1)
        if self.root_context_mode=='ignored':
            if visual.shape[-1] not in (2048,2054):raise ValueError('invalid navigation visual feature size')
            # Structurally omit unobserved RM columns; no invented RM state is appended.
            image=visual[...,:2048].contiguous()
            linear=torch.nn.functional.linear
            prior=linear(image,self.base.visual.weight[:,:2048],self.base.visual.bias)+self.base.goal(goal[...,:6].contiguous())
            if self.kind=='mlp':
                layer=self.visual_projection[0]
                projected=self.visual_projection[1](linear(image,layer.weight[:,:2048],layer.bias))
                inputs=torch.cat([projected,conditioned],-1)
            else:inputs=conditioned
        else:
            if visual.shape[-1]!=2054:raise ValueError('observed root context requires six RM channels')
            inputs=conditioned if self.kind=='linear' else torch.cat([self.visual_projection(visual),conditioned],-1)
            prior=self.base(torch.cat([visual,goal[...,:6]],-1))
        result=prior+self.residual(inputs)
        if self.visual_adapter_width:result=result+self.visual_adapter(image,conditioned)
        return result


def refinement_head(prior,kind,distance_scale=1,root_context_mode='observed',visual_adapter_width=0):
    checked_visual_adapter_width(visual_adapter_width)
    head=copy.deepcopy(prior).requires_grad_(False);count=0
    def replace(module):
        nonlocal count
        for name,child in list(module.named_children()):
            if isinstance(child,GoalLinear):setattr(module,name,RefinedLinear(child,kind,distance_scale,root_context_mode,visual_adapter_width));count+=1
            else:replace(child)
    replace(head)
    if count!=2:raise ValueError('expected two navigation output heads')
    return head


def attach_visual_adapter(head):
    modules=[m for m in head.modules() if isinstance(m,RefinedLinear)]
    if len(modules)!=2 or any(m.kind!='mlp' or m.root_context_mode!='ignored' or m.visual_adapter_width for m in modules):
        raise ValueError('expected two ignored-root MLP heads without an existing visual adapter')
    for module in modules:
        module.visual_adapter=VisualResidualAdapter(module.base.visual.out_features).to(module.base.visual.weight)
        module.visual_adapter_width=64
    return dict(previous_width=0,visual_adapter_width=64,heads=2,initialization_seed=17599,
        additional_parameters=sum(p.numel() for m in modules for p in m.visual_adapter.parameters()),
        initial_output_zero=True,normalization='Per-observation LayerNorm2048 without affine parameters or running statistics',
        scope='Additional learned visual/goal residual; original state tensors and root actor preserved')


def configure_refinement_training(head,scope='all-residuals'):
    """Choose optimizer support without changing the policy function or its tensors."""
    modules=[m for m in head.modules() if isinstance(m,RefinedLinear)]
    if scope not in ('all-residuals','visual-adapter-only') or len(modules)!=2:
        raise ValueError('expected two refinement heads and a supported training scope')
    if scope=='visual-adapter-only' and any(m.visual_adapter_width!=64 for m in modules):
        raise ValueError('adapter-only training requires an attached adapter on both heads')
    head.requires_grad_(False)
    for module in modules:
        if scope=='all-residuals':
            module.residual.requires_grad_(True)
            if hasattr(module,'visual_projection'):module.visual_projection.requires_grad_(True)
        if module.visual_adapter_width:module.visual_adapter.requires_grad_(True)
    return dict(scope=scope,trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in head.parameters() if not p.requires_grad))


def set_root_context_mode(head,mode):
    mode=checked_root_context_mode(mode)
    modules=[m for m in head.modules() if isinstance(m,RefinedLinear)]
    if len(modules)!=2 or len({m.root_context_mode for m in modules})!=1:
        raise ValueError('expected two consistently configured navigation heads')
    if mode!='ignored' and any(m.visual_adapter_width for m in modules):
        raise ValueError('visual adapter requires ignored root context')
    previous=modules[0].root_context_mode
    for module in modules:module.root_context_mode=mode
    return dict(previous_mode=previous,root_context_mode=mode,heads=len(modules),
                changes_input_function=previous!=mode)


def rescale_distance_input(head,distance_scale):
    new=checked_distance_scale(distance_scale)
    modules=[m for m in head.modules() if isinstance(m,RefinedLinear)]
    if len(modules)!=2 or len({m.distance_scale for m in modules})!=1:
        raise ValueError('expected two consistently scaled refinement heads')
    old=modules[0].distance_scale
    # Reparameterize the same function; the trainer checks numeric output equality.
    # The frozen base prior still receives its original distance/16 feature.
    with torch.no_grad():
        for module in modules:
            layer=module.residual if module.kind=='linear' else module.residual[0]
            column=layer.in_features-10+2
            layer.weight[:,column].mul_(old/new)
            if module.visual_adapter_width:
                module.visual_adapter.network[0].weight[:,2050].mul_(old/new)
            module.distance_scale=new
    return dict(previous_scale=old,distance_scale=new,heads=len(modules))


def rescale_adapter_spatial_input(head,scale):
    """Reparameterize four spatial inputs without changing the policy function.

    This attribute is deliberately not a checkpoint field: callers must fold back
    to one before export. All existing v4 inference paths use the canonical scale.
    """
    if type(scale) is not int or scale not in (1,8):
        raise ValueError('training spatial scale must be integer 1 or 8')
    modules=[m for m in head.modules() if isinstance(m,RefinedLinear)]
    if len(modules)!=2 or any(m.visual_adapter_width!=64 for m in modules):
        raise ValueError('spatial preconditioning requires two attached visual adapters')
    previous={m.visual_adapter.training_spatial_scale for m in modules}
    if len(previous)!=1 or not previous.issubset({1,8}):
        raise ValueError('inconsistent training spatial scale')
    old=previous.pop()
    with torch.no_grad():
        for m in modules:
            m.visual_adapter.network[0].weight[:,2048:2052].mul_(old/scale)
            m.visual_adapter.training_spatial_scale=scale
    return dict(previous_scale=old,training_spatial_scale=scale,heads=2,
                columns=['bearing_sin','bearing_cos','distance','relative_height'])


def append_motion(features,goal):
    g=torch.as_tensor(goal,device=features.device,dtype=features.dtype)
    if g.ndim==1:g=g.view(1,1,10)
    if g.ndim!=3 or g.shape[-1]!=10 or not torch.isfinite(g).all():raise ValueError('invalid motion goal')
    return torch.cat([features,g.expand(features.shape[0],features.shape[1],10)],-1)


def load_navigation(parent,saved):
    old=navigation_head(parent)
    scale=checked_distance_scale(saved.get('residual_distance_scale',1))
    mode=checked_root_context_mode(saved.get('root_context_mode','observed'))
    width=checked_visual_adapter_width(saved.get('visual_adapter_width',0))
    if saved['schema']!='navigation_refinement_v4' and width:
        raise ValueError('visual adapter requires version 4 checkpoint')
    if saved['schema']=='navigation_refinement_v4' and (width!=64 or mode!='ignored' or saved.get('kind')!='mlp'):
        raise ValueError('version 4 requires explicit64-wide adapter and ignored-root MLP')
    if saved['schema'] not in ('navigation_refinement_v3','navigation_refinement_v4') and mode!='observed':
        raise ValueError('ignored root context requires version 3 checkpoint')
    if saved['schema']=='waypoint_actor_v1':
        if scale!=1:raise ValueError('distance preconditioning requires a refined actor')
        old.load_state_dict(saved['actor']);return old,False
    if saved['schema'] not in ('navigation_refinement_v1','navigation_refinement_v2','navigation_refinement_v3','navigation_refinement_v4'):
        raise ValueError('unknown navigation checkpoint')
    if saved['schema']=='navigation_refinement_v1' and scale!=1:
        raise ValueError('scaled residual requires version 2 checkpoint')
    if saved['schema'] in ('navigation_refinement_v2','navigation_refinement_v3','navigation_refinement_v4') and 'residual_distance_scale' not in saved:
        raise ValueError('versioned checkpoint requires explicit distance scale')
    if saved['schema'] in ('navigation_refinement_v3','navigation_refinement_v4') and 'root_context_mode' not in saved:
        raise ValueError('version 3 checkpoint requires explicit root context mode')
    head=refinement_head(old,saved['kind'],scale,mode,width);head.load_state_dict(saved['actor']);return head,True


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
