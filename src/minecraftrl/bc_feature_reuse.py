"""Reuse complete, unchanged correction episodes from an audited temporal cache."""
from .bc_core import validate_plan
from .temporal_bc import episode_rows


def checked_reuse(data, plan, *, parent_sha256, model_sha256, embedding_sha256):
    import torch
    validate_plan(plan)
    identities = dict(parent_sha256=parent_sha256, model_sha256=model_sha256,
                      goal_embedding_sha256=embedding_sha256)
    if any(data.get(k) != value for k, value in identities.items()):
        raise ValueError('feature reuse requires identical parent, base model and embedding')
    for key in ('goal', 'data_source', 'frame_offset', 'rm_context', 'reward_mode', 'parent', 'masked_loss_required'):
        if data['plan'].get(key) != plan.get(key):
            raise ValueError('feature conditioning contract changed: '+key)
    if plan['data_source'] != 'student_correction' or any(
            k not in data for k in ('tail_initial', 'temporal_initial', 'temporal_initial_memory')):
        raise ValueError('complete masked temporal cache required')
    groups = episode_rows(data)
    previous = {s['id']: s for s in data['plan']['sessions']}
    reused = {}
    for session in plan['sessions']:
        name = session['id']
        if name not in previous:
            continue
        if session != previous[name]:
            raise ValueError('source session receipt changed: '+name)
        for row in groups[name]:
            x = row.get('tail_input')
            if not isinstance(x, torch.Tensor) or x.shape != row['temporal_input'].shape or not torch.isfinite(x).all():
                raise ValueError('invalid cached tail input')
        reused[name] = groups[name]
    return reused
