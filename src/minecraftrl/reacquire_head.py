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


def stage_onehot(index, use_stage):
    values = [0.] * STAGE_DIM
    if use_stage and index is not None and index >= 0: values[int(index)] = 1.
    return values


_onehot = stage_onehot   # ReacquireHead.forward's keyword argument shadows the name


class ReacquireHead(torch.nn.Module):
    def __init__(self, base, width, *, use_stage, hidden=64):
        super().__init__()
        self.base = base; self.width = int(width); self.use_stage = bool(use_stage)
        self.stage = None          # STAGES index the caller sets before each action; None = no stage
        self.last_latent = None    # detached policy latent of the last forward, [B, T, width]
        # Local initialization must not consume the policy's sampling RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(INIT_SEED)
            self.adapter = torch.nn.Sequential(torch.nn.Linear(self.width + STAGE_DIM, hidden), torch.nn.Tanh(),
                                               torch.nn.Linear(hidden, self.width))
        torch.nn.init.zeros_(self.adapter[-1].weight); torch.nn.init.zeros_(self.adapter[-1].bias)

    def delta(self, pi_h, onehot):
        if not self.use_stage: onehot = torch.zeros_like(onehot)
        return self.adapter(torch.cat([pi_h.float(), onehot.float()], -1))

    def forward(self, pi_h, mask=None, stage_onehot=None):
        """stage_onehot ([B, T, STAGE_DIM]) overrides `self.stage` row by row (training batches)."""
        self.last_latent = pi_h.detach()
        if stage_onehot is None:
            row = torch.tensor(_onehot(self.stage, self.use_stage), device=pi_h.device)
            stage_onehot = row.view(*(1,) * (pi_h.dim() - 1), STAGE_DIM).expand(*pi_h.shape[:-1], STAGE_DIM)
        latent = pi_h + self.delta(pi_h, stage_onehot.to(pi_h.device)).to(pi_h.dtype)
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
    torch.save(dict(schema=CHECKPOINT_SCHEMA, use_stage=head.use_stage, stages=list(STAGES), width=head.width,
                    adapter={k: v.detach().cpu() for k, v in head.adapter.state_dict().items()},
                    base_weights_sha256=base_weights_sha256, metadata=metadata), path)
    return sha256_file(path)


def load_into(head, path, *, base_weights_sha256):
    saved = torch.load(Path(path), map_location='cpu')
    if (saved.get('schema') != CHECKPOINT_SCHEMA or saved.get('use_stage') != head.use_stage
            or saved.get('stages') != list(STAGES) or saved.get('width') != head.width
            or saved.get('base_weights_sha256') != base_weights_sha256):
        raise ValueError('incompatible reacquire checkpoint')
    head.adapter.load_state_dict(saved['adapter'], strict=True)
    return head


def install(policy, *, use_stage, checkpoint=None):
    """Wrap the official action head of a Steve1Policy; the base stays frozen. Returns the head."""
    agent_policy = policy.agent.policy
    base = agent_policy.pi_head.requires_grad_(False)
    head = ReacquireHead(base, agent_policy.net.output_latent_size(), use_stage=use_stage)
    if checkpoint is not None:
        load_into(head, checkpoint, base_weights_sha256=sha256_file(Path(policy.root) / 'weights/steve1/steve1.weights'))
    device = next(base.parameters()).device
    agent_policy.pi_head = head.to(device).requires_grad_(False).eval()
    return head
