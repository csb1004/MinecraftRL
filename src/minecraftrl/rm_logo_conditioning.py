"""Explicit LOGO-inspired diagnostic mode; official policy remains frozen."""
import hashlib
import io
from pathlib import Path

from .goal_contract import digest, read_json
from .logo_validators import SuccessMonitor
from .logo_wood import action_candidate, admit, context, prompts, stage
from .rm_conditioning import ConditioningEnvironment, ConditioningPolicy
from .rl_collect import RLEnvironment
from .runtime import write_json


def load_bank(path):
    import numpy as np
    path = Path(path).resolve()
    meta = read_json(path/'manifest.json')
    validators, review = admit(meta['proposal'])
    if (meta.get('schema') != 'logo_conditioning_v1' or meta.get('context') != context()
            or meta.get('review') != review or meta.get('validator_sha256') != validators.sha256
            or meta.get('candidate') != action_candidate() or meta.get('prompts') != prompts()
            or meta.get('seed') != 31031 or meta.get('capability_verified') is not False):
        raise ValueError('goal bank does not match the reviewed LOGO adapter')
    if set(meta.get('embeddings', {})) != set(prompts()) or set(meta.get('tokens', {})) != set(prompts()):
        raise ValueError('missing LOGO embeddings/token receipts')
    embeddings = {}
    for key in prompts():
        token = meta['tokens'][key]
        ids, count = token.get('ids', []), token.get('count_including_special')
        if (type(count) is not int or not 2 <= count <= 77 or len(ids) != 77
                or token.get('truncated') is not False or token.get('sha256') != digest(ids)):
            raise ValueError('invalid token receipt')
        raw = (path/(key+'.npy')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['embeddings'][key]['file_sha256']:
            raise ValueError('goal file hash mismatch')
        value = np.load(io.BytesIO(raw), allow_pickle=False)
        if value.shape != (1,512) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('invalid goal vector')
        if hashlib.sha256(value.tobytes()).hexdigest() != meta['embeddings'][key]['vector_sha256']:
            raise ValueError('goal vector hash mismatch')
        embeddings[key] = value
    return embeddings, meta


class LogoReward:
    def __init__(self, monitor, server):
        self.monitor, self.server = monitor, server

    @property
    def completed(self):
        return self.monitor.completed

    def observe(self, inventory):
        current = self.server()
        if inventory != current['server_inventory']:
            raise ValueError('reward input differs from authoritative server observation')
        return self.monitor.advance(current)


class LogoEnvironment(ConditioningEnvironment):
    def reset(self):
        obs = RLEnvironment.reset(self)
        self.monitor = SuccessMonitor(self.policy.validators, self.gui)
        self.reward = LogoReward(self.monitor, lambda: self.gui)
        self.update_stage()
        return self.authoritative(obs)

    def update_stage(self):
        self.policy.validator_values = dict(self.monitor.values)
        self.policy.observe_stage(stage(self.monitor.values, self.gui))

    def step(self, action):
        result = super().step(action)
        self.policy.rows[-1]['validators_after'] = dict(self.monitor.values)
        self.policy.rows[-1]['validator_tick_after'] = self.monitor.last_tick
        return result


class LogoConditioningPolicy(ConditioningPolicy):
    bank_loader = staticmethod(load_bank)
    environment_type = LogoEnvironment

    def __init__(self, *args, **kwargs):
        self.validator_values = {}
        super().__init__(*args, **kwargs)
        self.validators, self.validator_review = admit(self.bank_metadata['proposal'])

    def reset(self, goal=None):
        super().reset(goal)
        self.condition_metadata.update(stage_source='reviewed_LOGO_predicates_over_fixed_privileged_server',
            validator_sha256=self.validators.sha256, validator_review=self.validator_review,
            root_success='server_inventory_delta', method='LOGO-inspired adaptation; not a full LOGO reproduction')
        write_json(self.output/'rm-condition.json', self.condition_metadata)

    def goal_key(self):
        if self.selector.current == 'complete':
            raise InterruptedError('no policy condition after root success')
        if self.mode == 'logo-fixed':
            return 'root'
        # Three-observation confirmation, but no irreversible gather latch.
        # Known-but-unreachable targets still need a separate recovery design.
        return 'seek' if self.selector.current == 'search' else 'root'

    def row_context(self):
        return dict(validator_sha256=self.validators.sha256, validators_before=dict(self.validator_values))
