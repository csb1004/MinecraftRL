"""Two stage heads on one frozen action head (sensor repair 2026-11-23).

The building head (category block x use) and the dig head (block x attack) are separate ReacquireHead residuals over
the same official STEVE-1 action head. The session's hook sees one head whose stage list is COMBINED_STAGES (the
building head's stages, then 'dig_' + the dig head's stages); the stage name it sets picks the residual that runs, with
that head's own stage index and features. The anomaly monitor's feed decides the stage, so it decides the head.
"""
from pathlib import Path

import torch

from .reacquire_head import ReacquireHead, checkpoint_feature_gain, checkpoint_features, load_into, sha256_file

PREFIX = 'dig_'


class SwitchHead(torch.nn.Module):
    def __init__(self, build, dig, prefix=PREFIX):
        super().__init__()
        if len(build.features) != len(dig.features): raise ValueError('both heads must take as many feature values')
        self.build = build; self.dig = dig; self.prefix = prefix
        self.stages = tuple(build.stages) + tuple(prefix + s for s in dig.stages)
        self.features = tuple(build.features)       # names for the latent log; the values belong to the head that runs
        self.feature_gain = build.feature_gain; self.width = build.width; self.use_stage = build.use_stage
        self._stage = None; self._which = 'build'; self._active = True

    def _head(self): return self.dig if self._which == 'dig' else self.build

    @property
    def active(self): return self._active

    @active.setter
    def active(self, on):
        self._active = bool(on); self.build.active = self._active; self.dig.active = self._active

    @property
    def stage(self): return self._stage

    @stage.setter
    def stage(self, index):
        self._stage = index
        if index is None:
            self.build.stage = None; self.dig.stage = None
            return
        name = self.stages[int(index)]
        if name.startswith(self.prefix):
            self._which = 'dig'; self.dig.stage = self.dig.stages.index(name[len(self.prefix):]); self.build.stage = None
        else:
            self._which = 'build'; self.build.stage = self.build.stages.index(name); self.dig.stage = None

    @property
    def feature(self): return self._head().feature

    @feature.setter
    def feature(self, values):
        self._head().feature = values
        if values is None: self.build.feature = None; self.dig.feature = None

    @property
    def last_latent(self): return self._head().last_latent

    def forward(self, pi_h, mask=None, **kwargs): return self._head()(pi_h, mask=mask, **kwargs)

    def sample(self, *args, **kwargs): return self.build.base.sample(*args, **kwargs)
    def logprob(self, *args, **kwargs): return self.build.base.logprob(*args, **kwargs)
    def entropy(self, *args, **kwargs): return self.build.base.entropy(*args, **kwargs)
    def kl_divergence(self, *args, **kwargs): return self.build.base.kl_divergence(*args, **kwargs)


def install_switch(policy, *, use_stage, build_checkpoint, dig_checkpoint, build_stages, dig_stages):
    """reacquire_head.install for two checkpoints: both residuals wrap the same frozen official head. Returns the
    SwitchHead, inactive until the hook turns it on."""
    agent_policy = policy.agent.policy
    base = agent_policy.pi_head.requires_grad_(False)
    width = agent_policy.net.output_latent_size()
    weights = sha256_file(Path(policy.root) / 'weights/steve1/steve1.weights')
    made = []
    for checkpoint, stages in ((build_checkpoint, build_stages), (dig_checkpoint, dig_stages)):
        features = checkpoint_features(checkpoint)
        head = ReacquireHead(base, width, use_stage=use_stage, stages=stages, features=features,
                             feature_gain=checkpoint_feature_gain(checkpoint) if features else 1.)
        made.append(load_into(head, checkpoint, base_weights_sha256=weights))
    switch = SwitchHead(*made)
    device = next(base.parameters()).device
    switch.active = False
    agent_policy.pi_head = switch.to(device).requires_grad_(False).eval()
    return switch
