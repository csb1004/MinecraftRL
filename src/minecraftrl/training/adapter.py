"""One verified on-policy PPO update for a frozen-motor common recovery actor."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from ..conditioned_head import ConditionedValue
from ..policies import model_kwargs
from ..recovery_learning import SCOPE, CONTEXT_FIELDS, VARIANTS, RecoveryHead
from ..rl_core import gae
from ..rl_opt import ppo_loss
from ..rl_policy import cpu_tree, guided_minibatch, state_fingerprint
from ..runtime import write_json
from .provenance import source_manifest, verify_sources
from .validation import validate


def build_parser(profile=None):
    scope = SCOPE if profile is None else profile.SCOPE
    variants = VARIANTS if profile is None else profile.VARIANTS
    execution = scope == 'execution_contract_e2'
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'output', 'stop-file'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--variant', choices=variants, required=True)
    p.add_argument('--parent', type=Path)
    if execution:
        p.add_argument('--critic-warmup-epochs', type=int, default=0, choices=range(11))
        p.add_argument('--retention-runs', type=Path, nargs='*', default=[])
    return p


def train(args, profile=None):
    """Admit fresh rollouts, replay behavior, update small heads, save a candidate.

    R1/E1/E2 retain their original return and objective boundaries. This function
    owns training only: it never launches Minecraft or promotes the checkpoint.
    """
    scope = SCOPE if profile is None else profile.SCOPE
    fields = CONTEXT_FIELDS if profile is None else profile.CONTEXT_FIELDS
    head = (RecoveryHead if profile is None else
            profile.actor_type if hasattr(profile, 'actor_type') else profile.EffectHead)
    execution = scope == 'execution_contract_e2'
    archive = 'recovery-trajectory.pt' if profile is None else profile.ARCHIVE
    root = args.root.resolve()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(root / 'results'):
        raise ValueError('fresh project-local output required')

    def check_stop():
        if args.stop_file.exists():
            raise InterruptedError('STOP during recovery training')
    check_stop()
    source_code = source_manifest(root)
    torch.set_num_threads(2)
    # Admit the parent and fresh rollouts before allocating the CUDA actor.
    parent = None
    parent_sha = None
    if args.parent:
        if not args.parent.resolve().is_relative_to(root / 'results'):
            raise ValueError('external parent rejected')
        raw = args.parent.read_bytes()
        parent_sha = hashlib.sha256(raw).hexdigest()
        parent = torch.load(args.parent, map_location='cpu')
        if parent['scope'] != scope or parent['variant'] != args.variant:
            raise ValueError('parent identity mismatch')
    runs = []
    sources = []
    if len({p.resolve() for p in args.runs}) != len(args.runs):
        raise ValueError('duplicate input run')
    for path in args.runs:
        check_stop()
        if not path.resolve().is_relative_to(root / 'results'):
            raise ValueError('external run rejected')
        data = torch.load(path / archive, map_location='cpu')
        summary = json.loads((path / 'summary.json').read_text())
        validate(data, summary, args.variant, parent_sha, profile)
        runs.append(data)
        sources.append(dict(
            path=str(path.relative_to(root)),
            trajectory_sha256=hashlib.sha256((path / archive).read_bytes()).hexdigest(),
            summary_sha256=hashlib.sha256((path / 'summary.json').read_bytes()).hexdigest(),
            world_sha256=data['world_sha256'], sampling_seed=data['sampling_seed'],
            task=data['recovery_task']))
    first = runs[0]
    dim = len(fields)
    width = first['feature_width'] - dim
    for data in runs:
        for key in ('initial_heads_sha256', 'base_model_sha256', 'feature_width', 'guidance', 'bank_metadata', 'official_motor_sha256'):
            if data[key] != first[key]:
                raise ValueError('mixed behavior ' + key)
    if parent and state_fingerprint(parent['heads']) != first['initial_heads_sha256']:
        raise ValueError('rollout did not use parent heads')
    # Keep the official motor frozen; only residual and value heads can learn.
    sys.path.insert(0, str(root / 'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.action_head import make_action_head
    _, kwargs = model_kwargs(root / 'weights/steve1/2x.model')
    space = DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())
    actor = head(make_action_head(space, width, **kwargs), width, args.variant).cuda()
    value = ConditionedValue(width, args.variant, context_dim=dim).cuda()
    actor.load_state_dict(first['initial_heads']['actor'], strict=True)
    value.load_state_dict(first['initial_heads']['value'], strict=True)
    original_motor = cpu_tree(actor.base.state_dict())
    if state_fingerprint(original_motor) != first['official_motor_sha256']:
        raise ValueError('initial motor identity mismatch')
    reference = copy.deepcopy(actor)
    with torch.no_grad():
        reference.adapter[-1].weight.zero_()
        reference.adapter[-1].bias.zero_()
    reference.eval().requires_grad_(False)
    # Retention anchors are separate train data and never PPO samples.
    retention_features = None
    retention_sources = []
    if execution:
        from minecraftrl.execution_training import check_retention_source, calibrate_value, objective_weights
        seen = set()
        retention_rows = []
        for path in args.retention_runs:
            check_stop()
            identity = str(path.resolve())
            if identity in seen or not path.resolve().is_relative_to(root / 'results'):
                raise ValueError('duplicate or external retention source')
            seen.add(identity)
            data = torch.load(path / archive, map_location='cpu')
            summary = json.loads((path / 'summary.json').read_text())
            check_retention_source(data, summary, first, source=identity, ppo_sources={str(p.resolve()) for p in args.runs})
            validate(data, summary, args.variant, data.get('checkpoint_sha256'), profile)
            for key in ('feature_width', 'bank_metadata', 'official_motor_sha256'):
                if data[key] != first[key]:
                    raise ValueError('retention feature/teacher mismatch: ' + key)
            retention_rows.extend(data['rows'])
            retention_sources.append(dict(
                path=str(path.relative_to(root)), role='official_KL_only_not_PPO',
                trajectory_sha256=hashlib.sha256((path / archive).read_bytes()).hexdigest(),
                rows=len(data['rows'])))
        if retention_rows:
            retention_features = torch.cat([r['features'] for r in retention_rows], 1).cuda()
    rows = [r for run in runs for r in run['rows']]
    features = torch.cat([r['features'] for r in rows], 1).cuda()
    actions = {k: torch.cat([r['action'][k] for r in rows], 1).cuda() for k in rows[0]['action']}
    old = torch.tensor([r['old_logp'] for r in rows], device='cuda')
    weights = None
    balance = None
    if execution:
        raw_weights, balance = objective_weights(rows)
        weights = torch.tensor(raw_weights, device='cuda')
    config = dict(
        gamma=0.999, gae_lambda=1.0, actor_lr=0.0001, value_lr=0.001,
        epochs=2, minibatch=32, ppo_clip=0.1, target_kl=0.02, reference_kl=0.1,
        seed=111101, trainable='residual_adapter_and_value_only')
    if profile is not None:
        config['return_boundary'] = 'verified_effect_or_episode_terminal'
    if execution:
        config.update(
            return_boundary='episode_terminal_only', reward_profile=profile.REWARD_PROFILE,
            critic_warmup_epochs=args.critic_warmup_epochs,
            retention_kl=0.1 if retention_features is not None else 0.0, objective_balance=balance)
    # Reproduce the behavior that generated the data before any optimizer step.
    logp_error = value_error = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            f = features[:, start:start + 32].contiguous()
            pd = guided_minibatch(actor, f)
            logp = actor.logprob({k: v[:, start:start + 32] for k, v in actions.items()}, pd).flatten()
            logp_error = max(logp_error, float((logp - old[start:start + 32]).abs().max()))
            expected = torch.tensor([r['value'] for r in rows[start:start + 32]], device='cuda')
            value_error = max(value_error, float((value(f[0:1]).flatten() - expected).abs().max()))
    if max(logp_error, value_error) > 0.0001:
        raise ValueError(f'behavior replay gate failed: {logp_error}, {value_error}')
    # Each profile defines its own return boundary; E2 uses episode completion.
    advantage = []
    returns = []
    for run in runs:
        if profile is None:
            a, t = gae(
                [r['reward'] for r in run['rows']], [r['value'] for r in run['rows']],
                run['bootstrap'], terminal=run['terminal'], gamma=0.999, lam=1.0)
        else:
            a, t = (profile.returns if hasattr(profile, 'returns') else profile.skill_returns)(run)
        advantage += a
        returns += t
    advantage = torch.tensor(advantage, device='cuda')
    target = torch.tensor(returns, device='cuda')
    calibration = None
    if execution:
        actor_before = state_fingerprint(actor.state_dict())
        calibration = calibrate_value(
            value, features[0:1], target, epochs=args.critic_warmup_epochs,
            check_stop=check_stop, weights=weights)
        if actor_before != state_fingerprint(actor.state_dict()):
            raise ValueError('critic calibration changed actor')
        if args.critic_warmup_epochs:
            with torch.no_grad():
                advantage = target - value(features[0:1]).flatten()
    if advantage.std(unbiased=False) < 1e-08:
        raise ValueError('no usable advantage variation; preserve failed data and revise curriculum explicitly')
    if execution:
        mean = (advantage * weights).mean()
        variance = ((advantage - mean).square() * weights).mean()
        advantage = (advantage - mean) / (variance.sqrt() + 1e-08)
    else:
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-08)
    # Preserve update/RNG order and separate actor/value learning rates.
    params = [p for p in actor.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([dict(params=params, lr=0.0001), dict(params=value.parameters(), lr=0.001)])
    if parent and 'optimizer' in parent:
        optimizer.load_state_dict(parent['optimizer'])
    torch.manual_seed(111101)
    torch.cuda.manual_seed_all(111101)
    history = []
    early = False
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(2):
        order = torch.randperm(len(rows), device='cuda')
        for start in range(0, len(rows), 32):
            check_stop()
            idx = order[start:start + 32]
            f = features[:, idx].contiguous()
            pd = guided_minibatch(actor, f)
            logp = actor.logprob({k: v[:, idx] for k, v in actions.items()}, pd).flatten()
            ratio = logp - old[idx]
            kl_old = float((ratio.exp() - 1 - ratio).mean().detach())
            if kl_old > 0.02:
                early = True
                break
            with torch.no_grad():
                ref = guided_minibatch(reference, f)
            kl = sum(((ref[k].exp() * (ref[k] - pd[k])).sum(-1).mean() for k in pd))
            if execution:
                clipped = torch.minimum(ratio.exp() * advantage[idx], ratio.exp().clamp(0.9, 1.1) * advantage[idx])
                loss = -(clipped * weights[idx]).mean() + 0.1 * kl
            else:
                loss = ppo_loss(logp, old[idx], advantage[idx], 0.1) + 0.1 * kl
            anchor_kl = None
            if retention_features is not None:
                anchor_idx = (torch.arange(len(idx), device='cuda') + len(history) * 32) % retention_features.shape[1]
                anchor = retention_features[:, anchor_idx].contiguous()
                anchor_pd = guided_minibatch(actor, anchor)
                with torch.no_grad():
                    anchor_ref = guided_minibatch(reference, anchor)
                anchor_kl = sum(((anchor_ref[k].exp() * (anchor_ref[k] - anchor_pd[k])).sum(-1).mean() for k in anchor_pd))
                loss = loss + 0.1 * anchor_kl
            squared_value_error = (value(f[0:1]).flatten() - target[idx]).square()
            vloss = (squared_value_error * weights[idx]).mean() if execution else squared_value_error.mean()
            if not torch.isfinite(loss + vloss):
                raise ValueError('nonfinite learning loss')
            optimizer.zero_grad()
            (loss + vloss).backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            value_grad = torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(
                epoch=epoch, rows=len(idx), actor_loss=float(loss.detach()), value_loss=float(vloss.detach()),
                old_kl=kl_old, official_kl=float(kl.detach()),
                actor_grad=float(actor_grad), value_grad=float(value_grad)))
            if execution:
                history[-1]['retention_kl'] = None if anchor_kl is None else float(anchor_kl.detach())
        if early:
            break
    if not history:
        raise ValueError('no optimizer updates')
    if state_fingerprint(original_motor) != state_fingerprint(actor.base.state_dict()):
        raise ValueError('frozen motor changed')
    heads = cpu_tree(dict(actor=actor.state_dict(), value=value.state_dict()))
    if not all((torch.isfinite(v).all() for h in heads.values() for v in h.values())):
        raise ValueError('nonfinite checkpoint')
    if state_fingerprint(heads['actor']) == state_fingerprint(first['initial_heads']['actor']):
        raise ValueError('actor unchanged')
    verify_sources(root, source_code)
    check_stop()
    out.mkdir(parents=True)
    checkpoint = dict(
        scope=scope, variant=args.variant, heads=heads, optimizer=cpu_tree(optimizer.state_dict()),
        base_model_sha256=first['base_model_sha256'], context_fields=list(fields), guidance=6.0,
        official_motor_sha256=first['official_motor_sha256'], parent_sha256=parent_sha,
        round=1 if parent is None else parent['round'] + 1, config=config, sources=sources, promoted=False)
    checkpoint['source_code'] = source_code
    if execution:
        checkpoint.update(retention_sources=retention_sources, calibration=calibration)
    torch.save(checkpoint, out / 'checkpoint.pt')
    result = dict(
        scope=scope, variant=args.variant, round=checkpoint['round'], parent_sha256=parent_sha,
        rows=len(rows), optimized_row_visits=sum(h['rows'] for h in history), updates=len(history),
        early_kl_stop=early, replay_logp_max_error=logp_error, replay_value_max_error=value_error,
        frozen_motor_unchanged=True,
        adapter_delta_l2=sum(
            float((v - first['initial_heads']['actor'][k]).square().sum())
            for k, v in heads['actor'].items() if k.startswith('adapter.')) ** 0.5,
        sources=sources, config=config, history=history,
        rewards=[sum(r['reward'] for r in run['rows']) for run in runs],
        checkpoint_sha256=hashlib.sha256((out / 'checkpoint.pt').read_bytes()).hexdigest(),
        elapsed_s=time.monotonic() - started, peak_cuda_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
        promoted=False)
    if profile is not None and (not execution):
        result['skill_rewards'] = [sum((r['skill_reward'] for r in run['rows'])) for run in runs]
    if execution:
        result.update(
            program_rewards=[sum(r['program_reward'] for r in run['rows']) for run in runs],
            calibration=calibration, retention_sources=retention_sources, ppo_rows=len(rows),
            retention_rows=0 if retention_features is None else retention_features.shape[1])
    result['source_code'] = source_code
    write_json(out / 'training.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'history', 'config', 'source_code')}), flush=True)


def main(profile=None, argv=None):
    """Parse the existing R1/E1/E2 CLI, then run the independent trainer."""
    train(build_parser(profile).parse_args(argv), profile)
