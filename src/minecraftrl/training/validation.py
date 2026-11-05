"""Admission checks shared by R1, E1 and E2 on-policy training."""
import math

import torch

from ..goal_spec import validate_trace
from ..recovery_learning import SCOPE, CONTEXT_FIELDS, VARIANTS
from ..rl_policy import state_fingerprint


def validate(data, summary, variant, parent_sha, profile=None):
    scope = SCOPE if profile is None else profile.SCOPE
    fields = CONTEXT_FIELDS if profile is None else profile.CONTEXT_FIELDS
    variants = VARIANTS if profile is None else profile.VARIANTS
    if (data.get('scope') != scope or data.get('split') != 'train' or not data.get('eligible')
            or data.get('checkpoint_sha256') != parent_sha
            or data.get('status') != summary.get('status')
            or data.get('status') not in ('target_reached', 'time_limit', 'environment_done')
            or summary.get('execution_valid') is not True or summary.get('worker_exit_code') != 0):
        raise ValueError('verified fresh on-policy training data required')
    if data.get('variant') not in variants or (parent_sha is not None and data['variant'] != variant):
        raise ValueError('candidate behavior variant mismatch')
    if data.get('context_fields') != list(fields) or data.get('recurrent_commit') != 'dispatch_only':
        raise ValueError('incompatible context or recurrent commit')
    if state_fingerprint(data['initial_heads']) != data['initial_heads_sha256']:
        raise ValueError('head fingerprint mismatch')
    rows = data['rows']
    dim = len(fields)
    if not rows or len(rows) != summary.get('steps'):
        raise ValueError('invalid committed action count')
    for i, row in enumerate(rows):
        if row['step'] != i + 1 or (row['terminal'] and i != len(rows) - 1):
            raise ValueError('invalid terminal ordering')
        if not all((math.isfinite(row[k]) for k in ('reward', 'value', 'old_logp'))):
            raise ValueError('nonfinite row')
        if row['reward'] not in (0.0, 1.0):
            raise ValueError('unsupported reward')
        expected = torch.tensor(row['phase_context'], dtype=torch.float32).view(1, 1, dim).expand(2, 1, dim)
        f = row['features']
        if (f.shape != (2, 1, data['feature_width']) or not torch.isfinite(f).all()
                or not torch.equal(f[..., -dim:], expected)):
            raise ValueError('context disconnected from actor')
    if bool(rows[-1]['terminal']) != bool(data['terminal']) or not math.isfinite(data['bootstrap']):
        raise ValueError('terminal/bootstrap mismatch')
    if data['terminal'] and data['bootstrap'] != 0:
        raise ValueError('terminal bootstrap must be zero')
    if not data['terminal'] and data.get('bootstrap_features') is None:
        raise ValueError('truncation requires final features')
    validate_trace(data)
    if profile is not None:
        (profile.validate if hasattr(profile, 'validate') else profile.validate_effect_trace)(data)
