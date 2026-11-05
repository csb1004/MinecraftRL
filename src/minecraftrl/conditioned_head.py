"""Small residual visual/context interaction; no change to motor action vocabulary."""
import torch
from torch import nn
from .rl_policy import extend_context_head
from .semantic_goal import CONTEXT_DIM

VARIANTS = ('linear-off', 'linear-on', 'nonlinear-off', 'nonlinear-on')


class ConditionedHead(nn.Module):
    def __init__(self, base, width, variant, hidden=64, context_dim=CONTEXT_DIM):
        super().__init__()
        if variant not in VARIANTS: raise ValueError('unknown S2 architecture')
        self.base = base
        self.width = width
        self.variant = variant
        self.context_dim = context_dim
        extend_context_head(base, width, context_dim)
        self.adapter = None
        if variant.startswith('nonlinear'):
            # Local initialization cannot consume the policy sampling RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(110101)
                self.adapter = nn.Sequential(nn.Linear(width+context_dim, hidden), nn.Tanh(), nn.Linear(hidden, width))
                nn.init.zeros_(self.adapter[-1].weight)
                nn.init.zeros_(self.adapter[-1].bias)

    def conditioned(self, features):
        if features.shape[-1] != self.width+self.context_dim: raise ValueError('S2 feature shape mismatch')
        visual, ctx = features[..., :self.width], features[..., self.width:]
        if self.variant.endswith('-off'): ctx = torch.zeros_like(ctx)
        joined = torch.cat([visual, ctx], -1)
        if self.adapter is not None:
            visual = visual+self.adapter(joined)
        return torch.cat([visual, ctx], -1)

    def forward(self, features): return self.base(self.conditioned(features))
    def sample(self, *args, **kwargs): return self.base.sample(*args, **kwargs)
    def logprob(self, *args, **kwargs): return self.base.logprob(*args, **kwargs)


class ConditionedValue(nn.Module):
    def __init__(self, width, variant, context_dim=CONTEXT_DIM):
        super().__init__()
        self.width, self.variant = width, variant
        if variant.startswith('nonlinear'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(110102)
                self.net = nn.Sequential(nn.Linear(width+context_dim,64),nn.Tanh(),nn.Linear(64,1))
            final = self.net[-1]
        else:
            self.net = nn.Linear(width+context_dim,1)
            final = self.net
        nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)

    def forward(self, features):
        if self.variant.endswith('-off'):
            features = torch.cat([features[..., :self.width],torch.zeros_like(features[..., self.width:])],-1)
        return self.net(features)
