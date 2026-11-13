"""Stage-conditioned residual on the frozen STEVE-1 action head (design 2026-11-07 §5).

ReacquireHead replaces `policy.agent.policy.pi_head`. It adds a small zero-initialized residual to the policy latent
before the official head, so an untrained head returns exactly the official output. The stage input is a one-hot of
`stage_replay.STAGES` (privileged: computed from fixture positions), or zeros for the image-only arm. Classifier-free
guidance runs the head on both the conditional and the unconditional latent; both get the same stage.
"""
import hashlib
from pathlib import Path

import torch

from .pipeline.stage_replay import STAGES

STAGE_DIM = len(STAGES)
CHECKPOINT_SCHEMA = 'reacquire_head_v1'
INIT_SEED = 110201


def stage_onehot(index, use_stage, dim=STAGE_DIM):
    values = [0.] * dim
    if use_stage and index is not None and index >= 0: values[int(index)] = 1.
    return values


_onehot = stage_onehot   # ReacquireHead.forward's keyword argument shadows the name


class ReacquireHead(torch.nn.Module):
    def __init__(self, base, width, *, use_stage, hidden=64, stages=STAGES, features=(), feature_gain=1.):
        super().__init__()
        self.base = base; self.width = int(width); self.use_stage = bool(use_stage)
        # The stage list of the skill (default: the mining stages; the jump course has its own, design 2026-11-13).
        self.stages = tuple(stages); self.stage_dim = len(self.stages)
        # Privileged features next to the stage one-hot (jump direction input, design 2026-11-14); none by default, so
        # the adapter and every earlier checkpoint keep their shape.
        self.features = tuple(features); self.feature_dim = len(self.features)
        # Feature values are multiplied by this before the adapter: values of size 1 next to a 2048-wide latent were
        # barely used (offline gate 2026-11-14), so the jump heads use 10.
        self.feature_gain = float(feature_gain)
        self.stage = None          # index into self.stages the caller sets before each action; None = no stage
        self.feature = None        # feature values the caller sets before each action; None = zeros
        self.last_latent = None    # detached policy latent of the last forward, [B, T, width]
        # False = the official head alone. install() turns it off; the live hook turns it on only for policy steps of
        # the skill with a `reacquire` block, so other skills in the same game keep the official policy.
        self.active = True
        # Local initialization must not consume the policy's sampling RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(INIT_SEED)
            self.adapter = torch.nn.Sequential(torch.nn.Linear(self.width + self.stage_dim + self.feature_dim, hidden),
                                               torch.nn.Tanh(), torch.nn.Linear(hidden, self.width))
        torch.nn.init.zeros_(self.adapter[-1].weight); torch.nn.init.zeros_(self.adapter[-1].bias)

    def delta(self, pi_h, onehot, features=None):
        if not self.use_stage: onehot = torch.zeros_like(onehot)
        parts = [pi_h.float(), onehot.float()]
        if self.feature_dim:
            if features is None or not self.use_stage:
                features = torch.zeros(*pi_h.shape[:-1], self.feature_dim, device=pi_h.device)
            parts.append(features.float() * self.feature_gain)
        return self.adapter(torch.cat(parts, -1))

    def _row(self, values, pi_h, dim):
        row = torch.tensor(values, dtype=torch.float32, device=pi_h.device)
        return row.view(*(1,) * (pi_h.dim() - 1), dim).expand(*pi_h.shape[:-1], dim)

    def forward(self, pi_h, mask=None, stage_onehot=None, feature_values=None):
        """stage_onehot ([B, T, stage_dim]) and feature_values ([B, T, feature_dim]) override `self.stage` and
        `self.feature` row by row (training batches)."""
        self.last_latent = pi_h.detach()
        if not self.active: return self.base(pi_h, mask=mask)
        if stage_onehot is None:
            stage_onehot = self._row(_onehot(self.stage, self.use_stage, self.stage_dim), pi_h, self.stage_dim)
        if self.feature_dim and feature_values is None:
            values = [0.] * self.feature_dim if self.feature is None else [float(v) for v in self.feature]
            if len(values) != self.feature_dim: raise ValueError('expected %d feature values' % self.feature_dim)
            feature_values = self._row(values, pi_h, self.feature_dim)
        features = None if feature_values is None else feature_values.to(pi_h.device)
        latent = pi_h + self.delta(pi_h, stage_onehot.to(pi_h.device), features).to(pi_h.dtype)
        return self.base(latent, mask=mask)

    def sample(self, *args, **kwargs): return self.base.sample(*args, **kwargs)
    def logprob(self, *args, **kwargs): return self.base.logprob(*args, **kwargs)
    def entropy(self, *args, **kwargs): return self.base.entropy(*args, **kwargs)
    def kl_divergence(self, *args, **kwargs): return self.base.kl_divergence(*args, **kwargs)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def save_checkpoint(head, path, *, base_weights_sha256, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(features=list(head.features), feature_gain=head.feature_gain) if head.feature_dim else {}   # earlier: no keys
    torch.save(dict(schema=CHECKPOINT_SCHEMA, use_stage=head.use_stage, stages=list(head.stages), width=head.width,
                    adapter={k: v.detach().cpu() for k, v in head.adapter.state_dict().items()},
                    base_weights_sha256=base_weights_sha256, metadata=metadata, **extra), path)
    return sha256_file(path)


def checkpoint_features(path):
    """The feature list a checkpoint was trained with (empty for every checkpoint before 2026-11-14)."""
    return tuple(torch.load(Path(path), map_location='cpu').get('features') or ())


def checkpoint_feature_gain(path):
    """The feature gain a checkpoint was trained with (1 when it names none)."""
    return float(torch.load(Path(path), map_location='cpu').get('feature_gain', 1.))


def load_into(head, path, *, base_weights_sha256):
    saved = torch.load(Path(path), map_location='cpu')
    if (saved.get('schema') != CHECKPOINT_SCHEMA or saved.get('use_stage') != head.use_stage
            or saved.get('stages') != list(head.stages) or saved.get('width') != head.width
            or list(saved.get('features') or ()) != list(head.features)
            or (head.feature_dim and float(saved.get('feature_gain', 1.)) != head.feature_gain)
            or saved.get('base_weights_sha256') != base_weights_sha256):
        raise ValueError('incompatible reacquire checkpoint')
    head.adapter.load_state_dict(saved['adapter'], strict=True)
    return head


def install(policy, *, use_stage, checkpoint=None, stages=STAGES):
    """Wrap the official action head of a Steve1Policy; the base stays frozen. Returns the head, inactive until a
    reacquire step turns it on (ReacquireHook.activate). The head takes the checkpoint's feature list (none without one)."""
    agent_policy = policy.agent.policy
    base = agent_policy.pi_head.requires_grad_(False)
    features = checkpoint_features(checkpoint) if checkpoint is not None else ()
    gain = checkpoint_feature_gain(checkpoint) if features else 1.
    head = ReacquireHead(base, agent_policy.net.output_latent_size(), use_stage=use_stage, stages=stages, features=features,
                         feature_gain=gain)
    if checkpoint is not None:
        load_into(head, checkpoint, base_weights_sha256=sha256_file(Path(policy.root) / 'weights/steve1/steve1.weights'))
    device = next(base.parameters()).device
    head.active = False
    agent_policy.pi_head = head.to(device).requires_grad_(False).eval()
    return head
