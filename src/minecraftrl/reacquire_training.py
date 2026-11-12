"""Train the reacquire residual: guided BC on teacher labels with a KL leash to the official policy (design §6).

Only ReacquireHead.adapter learns; the official STEVE-1 head and everything before it stay frozen. The loss is the
mean NLL of the labels under the guided distribution the game samples, (1 + s) * conditional - s * unconditional,
renormalized, plus kl_coef times the mean KL to the official guided distribution on unlabelled rows. An epoch whose
mean KL on the training rows exceeds kl_limit is rolled back and ends training. The holdout run is only measured.
"""
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .pipeline.stage_replay import STAGES
from .reacquire_head import STAGE_DIM, ReacquireHead, load_into, save_checkpoint, sha256_file

SOURCES = ('src/minecraftrl/reacquire_training.py', 'src/minecraftrl/reacquire_head.py')
EVAL_BATCH = 1024


def stage_onehots(stage, device, dim=STAGE_DIM):
    st = stage.long().to(device)
    onehot = torch.zeros(len(st), dim, device=device)
    valid = st >= 0
    onehot[valid.nonzero().squeeze(1), st[valid]] = 1.
    return onehot


def guided_logp(head, cond, uncond, stage, cond_scale, *, official=False):
    """{key: [N, n] log-probs} of the guided distribution for N rows; `official` skips the residual."""
    x = torch.stack([cond.float(), uncond.float()], 0)           # [2 (cond, uncond), N, width]
    if official: pd = head.base(x)
    else: pd = head(x, stage_onehot=stage_onehots(stage, x.device, head.stage_dim).unsqueeze(0).expand(2, -1, -1))
    out = {}
    for key, value in pd.items():
        value = value.float().reshape(2, x.shape[1], -1)          # drop the singleton action dimension
        out[key] = torch.log_softmax((1 + cond_scale) * value[0] - cond_scale * value[1], -1)
    return out


def bc_nll(logp, label_buttons, label_camera, camera_on, weight=None):
    """Mean label NLL (weighted mean with `weight`); the camera term counts only where the label moves the camera
    (official camera meta)."""
    rows = torch.arange(len(label_buttons), device=logp['buttons'].device)
    nll = -logp['buttons'][rows, label_buttons.long().to(rows.device)]
    camera = -logp['camera'][rows, label_camera.long().clamp(min=0).to(rows.device)]
    per_row = nll + camera * camera_on.to(rows.device).float()
    if weight is None: return per_row.mean()
    weight = weight.to(rows.device).float()
    return (per_row * weight).sum() / weight.sum().clamp(min=1e-12)


def group_weights(labelled, stage, block, rows):
    """Per-row BC weights over `rows` (multi-block design 2026-11-08 §7): each labelled row gets 1 / (labelled rows of
    its (block, stage) group among `rows`), scaled so the labelled rows of `rows` average 1. Other rows get 0. Without a
    block record the groups are the stages."""
    n = len(labelled)
    weight = torch.zeros(n)
    rows = torch.as_tensor(rows, dtype=torch.long).cpu()
    lab_rows = rows[labelled.cpu()[rows]]
    if not len(lab_rows): return weight
    st = stage.long().cpu()[lab_rows]
    bl = torch.zeros_like(st) if block is None else block.long().cpu()[lab_rows]
    key = bl * 64 + (st + 1)
    _, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)
    raw = 1. / counts[inverse].float()
    weight[lab_rows] = raw / raw.mean()
    return weight


def kl_to_official(logp_student, logp_official):
    """Per-row KL(student || official), summed over the action keys."""
    return sum((s.exp() * (s - logp_official[k])).sum(-1) for k, s in logp_student.items())


def over_leash(entry, kl_limit):
    """Roll back an epoch when an update broke the policy (KL not a number) or the unlabelled rows, which should keep the
    official behaviour, moved more than kl_limit. The labelled rows are meant to move and are not leashed."""
    if not math.isfinite(entry['kl']): return True
    free = entry['kl_unlabelled']
    return math.isfinite(free) and free > kl_limit


def official_action_head(root):
    """The official STEVE-1 action head (CPU) and the policy latent width."""
    import pickle
    import sys
    sys.path.insert(0, str(Path(root) / 'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.agent import CameraHierarchicalMapping
    from steve1.embed_conditioned_policy import MinecraftAgentPolicy
    with open(Path(root) / 'weights/steve1/2x.model', 'rb') as stream: parameters = pickle.load(stream)
    kwargs = parameters['model']['args']['net']['args']
    opts = dict(parameters['model']['args']['pi_head_opts']); opts['temperature'] = float(opts['temperature'])
    space = DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())
    policy = MinecraftAgentPolicy(policy_kwargs=kwargs, pi_head_kwargs=opts, action_space=space)
    policy.load_state_dict(torch.load(str(Path(root) / 'weights/steve1/steve1.weights'), map_location='cpu'), strict=True)
    return policy.pi_head.eval(), policy.net.output_latent_size()


def official_factory(root):
    base_sha = sha256_file(Path(root) / 'weights/steve1/steve1.weights')

    def factory(use_stage, width, stages=STAGES):
        base, base_width = official_action_head(root)
        if base_width != width: raise ValueError('dataset latent width %d differs from the policy (%d)' % (width, base_width))
        return ReacquireHead(base.requires_grad_(False), width, use_stage=use_stage, stages=stages), base_sha
    return factory


def train(dataset_path, *, use_stage, out_dir, lr=1e-4, batch=32, epochs=16, kl_coef=.01, kl_limit=3., clip=1., seed=17,
          head_factory=None, root=None, device=None, balance=False, patience=None):
    """`balance`: BC weighted by group_weights over the training rows (the holdout NLL uses its own rows' weights).
    `patience`: stop when the holdout NLL has not reached a new low for that many epochs and restore the best state
    (the untrained head counts as epoch 0)."""
    started = time.time(); out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()): raise FileExistsError('refusing to overwrite %s' % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    dataset_path = Path(dataset_path); data = torch.load(dataset_path)
    factory = head_factory or official_factory(root)
    # A dataset of another skill names its own stage list (jump course, design 2026-11-13); mining datasets keep the
    # default call, so earlier factories and checkpoints are unchanged.
    stages = tuple(data.get('stages') or STAGES)
    make = (lambda: factory(use_stage, int(data['width']))) if stages == tuple(STAGES) else (
        lambda: factory(use_stage, int(data['width']), stages))
    head, base_sha = make(); head = head.to(device)
    head.base.requires_grad_(False); head.adapter.requires_grad_(True)
    cond, uncond = data['cond'].to(device), data['uncond'].to(device)
    stage, labelled = data['stage'].to(device), data['labelled'].to(device)
    label_buttons, label_camera, camera_on = data['label_buttons'].to(device), data['label_camera'].to(device), data['camera_on'].to(device)
    holdout = data['holdout']
    held_ids = [int(h) for h in holdout] if isinstance(holdout, (list, tuple)) else [int(holdout)]
    held = torch.isin(data['run'], torch.tensor(held_ids, dtype=data['run'].dtype)).to(device)
    train_rows, hold_rows = (~held).nonzero().squeeze(1), held.nonzero().squeeze(1)
    if not len(train_rows): raise ValueError('no training rows: every row belongs to the holdout run')
    scale = float(data['cond_scale'])
    train_w = hold_w = None
    if balance:
        train_w = group_weights(data['labelled'], data['stage'], data.get('block'), train_rows.cpu()).to(device)
        hold_w = group_weights(data['labelled'], data['stage'], data.get('block'), hold_rows.cpu()).to(device)

    def measure(rows, weight=None):
        """(mean label NLL over labelled rows, mean KL to the official policy over all rows, the same KL over the
        unlabelled rows) without gradients. The KL limit applies to the unlabelled rows: labelled rows are meant to move."""
        nll_sum, nll_n, kl_sum, free_sum, free_n = 0., 0, 0., 0., 0
        with torch.no_grad():
            for b in range(0, len(rows), EVAL_BATCH):
                idx = rows[b:b + EVAL_BATCH]; lab = labelled[idx]
                logp = guided_logp(head, cond[idx], uncond[idx], stage[idx], scale)
                kl = kl_to_official(logp, guided_logp(head, cond[idx], uncond[idx], stage[idx], scale, official=True))
                kl_sum += kl.sum().item(); free_sum += kl[~lab].sum().item(); free_n += int((~lab).sum())
                if lab.any():
                    part = {k: v[lab] for k, v in logp.items()}
                    w = None if weight is None else weight[idx][lab]
                    mass = int(lab.sum()) if w is None else float(w.sum())
                    nll_sum += bc_nll(part, label_buttons[idx][lab], label_camera[idx][lab], camera_on[idx][lab],
                                      weight=w).item() * mass
                    nll_n += mass
        nan = float('nan')
        return (nll_sum / nll_n if nll_n else nan), (kl_sum / len(rows) if len(rows) else nan), (free_sum / free_n if free_n else nan)

    initial_nll, initial_hold_kl, _ = measure(hold_rows, hold_w); _, initial_kl, _ = measure(train_rows)
    if initial_kl != 0.: raise RuntimeError('the untrained head differs from the official policy (KL %r)' % initial_kl)
    optimizer = torch.optim.AdamW(head.adapter.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    log, rolled_back, updates_kept = [], [], 0
    best = dict(epoch=0, nll=initial_nll, state=copy.deepcopy(head.adapter.state_dict()), updates=0)
    stopped_early = False
    for epoch in range(1, epochs + 1):
        snapshot = copy.deepcopy(head.adapter.state_dict()); optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        order = train_rows[torch.randperm(len(train_rows), generator=generator).to(device)]
        losses, updates = [], 0
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]; lab = labelled[idx]
            logp = guided_logp(head, cond[idx], uncond[idx], stage[idx], scale)
            loss = torch.zeros((), device=device)
            if lab.any():
                loss = loss + bc_nll({k: v[lab] for k, v in logp.items()}, label_buttons[idx][lab], label_camera[idx][lab],
                                     camera_on[idx][lab], weight=None if train_w is None else train_w[idx][lab])
            if (~lab).any():
                with torch.no_grad(): official = guided_logp(head, cond[idx], uncond[idx], stage[idx], scale, official=True)
                loss = loss + kl_coef * kl_to_official({k: v[~lab] for k, v in logp.items()},
                                                       {k: v[~lab] for k, v in official.items()}).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.adapter.parameters(), clip); optimizer.step()
            losses.append(loss.item()); updates += 1
        train_nll, train_kl, train_free_kl = measure(train_rows, train_w); hold_nll, hold_kl, _ = measure(hold_rows, hold_w)
        entry = dict(epoch=epoch, train_loss=sum(losses) / max(len(losses), 1), train_nll=train_nll, kl=train_kl,
                     kl_unlabelled=train_free_kl, holdout_nll=hold_nll, holdout_kl=hold_kl, updates=updates)
        if over_leash(entry, kl_limit):
            head.adapter.load_state_dict(snapshot); optimizer.load_state_dict(optimizer_snapshot)
            rolled_back.append(epoch); log.append(dict(entry, rolled_back=True)); break
        updates_kept += updates; log.append(entry)
        if patience is not None:
            if math.isfinite(hold_nll) and hold_nll < best['nll']:
                best = dict(epoch=epoch, nll=hold_nll, state=copy.deepcopy(head.adapter.state_dict()), updates=updates_kept)
            elif epoch - best['epoch'] >= patience:
                stopped_early = True; break
    if patience is not None and best['epoch'] != len([e for e in log if not e.get('rolled_back')]):
        head.adapter.load_state_dict(best['state']); updates_kept = best['updates']
    final_nll, final_hold_kl, _ = measure(hold_rows, hold_w); _, final_kl, final_free_kl = measure(train_rows)
    candidate = updates_kept > 0 and math.isfinite(final_nll) and final_nll < initial_nll
    path = out_dir / ('checkpoint.pt' if candidate else 'checkpoint-rejected.pt')
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    hyper = dict(lr=lr, batch=batch, epochs=epochs, kl_coef=kl_coef, kl_limit=kl_limit, clip=clip, seed=seed,
                 balance=bool(balance), patience=patience)
    checkpoint_sha = save_checkpoint(head, path, base_weights_sha256=base_sha,
                                     metadata=dict(dataset=str(dataset_path), dataset_sha256=dataset_sha, hyper=hyper,
                                                   updates_kept=updates_kept, candidate=candidate))
    fresh, _ = make()
    fresh = load_into(fresh, path, base_weights_sha256=base_sha).to(device)
    check = train_rows[:256]
    with torch.no_grad():
        a = guided_logp(head, cond[check], uncond[check], stage[check], scale)
        b = guided_logp(fresh, cond[check], uncond[check], stage[check], scale)
    reload_diff = max(float((a[k] - b[k]).abs().max()) for k in a) if len(check) else 0.
    report = dict(dataset=str(dataset_path), dataset_sha256=dataset_sha, use_stage=use_stage, hyper=hyper,
                  rows=dict(train=len(train_rows), holdout=len(hold_rows), labelled_train=int(labelled[train_rows].sum()),
                            labelled_holdout=int(labelled[hold_rows].sum())),
                  holdout=[data['runs'][i] for i in held_ids] if isinstance(holdout, (list, tuple)) else data['runs'][held_ids[0]],
                  balance=bool(balance), patience=patience, best_epoch=best['epoch'] if patience is not None else None,
                  stopped_early=stopped_early, initial=dict(holdout_nll=initial_nll, holdout_kl=initial_hold_kl, kl=initial_kl),
                  epochs=log, rolled_back_epochs=rolled_back, updates_kept=updates_kept,
                  final=dict(holdout_nll=final_nll, holdout_kl=final_hold_kl, kl=final_kl, kl_unlabelled=final_free_kl),
                  kl_limit_rows='unlabelled training rows', candidate=candidate,
                  checkpoint=str(path), checkpoint_sha256=checkpoint_sha, base_weights_sha256=base_sha,
                  reload_max_abs_diff=reload_diff, seconds=time.time() - started, device=device,
                  torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if device == 'cuda' else None,
                  source={p: hashlib.sha256((Path(root) / p).read_bytes()).hexdigest() for p in SOURCES}
                  if root is not None else None,
                  trained='ReacquireHead.adapter only; official STEVE-1 frozen')
    if reload_diff != 0.: report['candidate'] = False; report['reload_failed'] = True
    (out_dir / 'train-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report
