"""Training examples for the reacquire residual from recorded probe runs (design 2026-11-07 §4, §6).

Each dispatched policy action of the reacquire step has a recorded latent (stage_live.LatentLog). Its label comes
from the privileged teacher on the state the policy saw when choosing it: the last telemetry row before that action.
Stages are recomputed with the same LiveStageFeed the game used; a recorded stage that differs refuses the run.
Only runs on the training world seeds are accepted.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .pipeline.monitor_replay import load_run
from .pipeline.stage_live import LiveStageFeed
from .pipeline.stage_replay import STAGES, _resolve
from .reacquire_teacher import MINING_FEATURES, aim_delta, in_view, mining_features, policy_label, target_of, teacher_env_action

TRAIN_SEEDS = (271828, 314159, 161803)
FORBIDDEN_SEEDS = (130363, 104729, 577215)   # development, analysed evaluation, final evaluation
SOURCES = ('src/minecraftrl/reacquire_data.py', 'src/minecraftrl/reacquire_teacher.py',
           'src/minecraftrl/pipeline/stage_live.py', 'src/minecraftrl/pipeline/stage_replay.py')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_latents(folder, stages=STAGES):
    """(manifest, arrays) of a complete latent record whose chunk hashes match. `stages` is the skill's stage list (the
    jump course has its own, design 2026-11-13)."""
    folder = Path(folder)
    if not (folder / 'manifest.json').exists(): raise ValueError('no latent record: %s' % folder)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'): raise ValueError('latent record was not closed: %s' % folder)
    if manifest.get('stages') != list(stages): raise ValueError('latent record uses another stage list')
    # A head with privileged features (jump direction input, design 2026-11-14) also logged them per decision.
    keys = ('step', 'cond', 'uncond', 'stage') + (('features',) if manifest.get('features') else ())
    parts = []
    for chunk in manifest['chunks']:
        path = folder / chunk['file']
        if sha256(path) != chunk['sha256']: raise ValueError('latent chunk hash differs: %s' % path)
        with np.load(path) as data: parts.append({k: data[k] for k in keys})
    arrays = {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros((0,)) for k in keys}
    if len(arrays['step']) != manifest['rows']: raise ValueError('latent record row count differs from its manifest')
    return manifest, arrays


def _checked_request(run_dir, root):
    request = json.loads((run_dir / 'request.json').read_text(encoding='utf-8'))
    seed = request.get('seed')
    if seed in FORBIDDEN_SEEDS or seed not in TRAIN_SEEDS: raise ValueError('seed %s is not a training seed' % seed)
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    if not summary.get('execution_valid'): raise ValueError('execution not valid: %s' % run_dir)
    fixture = _resolve(request['fixture'], root) if request.get('fixture') else None
    if fixture is None or sha256(fixture) != request.get('fixture_sha256'):
        raise ValueError('fixture missing or different from the record: %s' % run_dir)
    return request, json.loads(Path(fixture).read_text(encoding='utf-8'))


FEATURE_TOLERANCE = 1e-4


def run_examples(run_dir, mapper, transformer, *, root, teacher='centre', features=False):
    """`features`: also the privileged direction features of every decision (mining direction input design 2026-11-14),
    recomputed from telemetry; a latent record that logged them must agree."""
    run_dir = Path(run_dir)
    request, fixture = _checked_request(run_dir, root)
    manifest, lat = load_latents(run_dir / 'reacquire')
    if manifest.get('features') and list(manifest['features']) != list(MINING_FEATURES):
        raise ValueError('latent record uses another feature list: %s' % run_dir)
    run = load_run(run_dir)
    steps = [s for s in run.steps if s.skill == manifest['meta']['skill']]
    if len(steps) != 1: raise ValueError('expected one %s step, found %d' % (manifest['meta']['skill'], len(steps)))
    step = steps[0]; wanted = step.block or step.target
    feed = LiveStageFeed(fixture, wanted, step.target); started = False; decisions = []; feats = []
    for r in run.rows:
        if not started and r['t'] >= step.start_t: feed.start_step(step.start_t); started = True
        inside = r['step'] >= step.start_step and (step.end_step is None or r['step'] <= step.end_step)
        if started and inside and r.get('controller') == 'policy':
            view, s = feed.decision_view(), feed.last_sample
            # The label is made here, not after the loop: the sight teacher reads the fixture blocks of this moment
            # (the tracker removes broken blocks as the run goes on).
            geometry = feed.tracker.geometry if feed.tracker is not None else None
            action = None if view is None or s is None else teacher_env_action(s, view, wanted, aim=teacher, geometry=geometry)
            decisions.append((r['step'], view, s, action)); feats.append(mining_features(s, view, wanted))
        feed.observe(r)
    if [d[0] for d in decisions] != lat['step'].tolist():
        raise ValueError('latent record does not match the policy rows (%d latents, %d policy rows)'
                         % (len(lat['step']), len(decisions)))
    n = len(decisions)
    out = dict(cond=lat['cond'].astype(np.float16), uncond=lat['uncond'].astype(np.float16),
               stage=np.full(n, -1, np.int8), labelled=np.zeros(n, bool), label_buttons=np.full(n, -1, np.int64),
               label_camera=np.full(n, -1, np.int64), camera_on=np.zeros(n, bool), in_view=np.zeros(n, bool),
               label_pitch=np.zeros(n, np.float32), label_yaw=np.zeros(n, np.float32),
               step=lat['step'].astype(np.int32), cond_scale=float(manifest['cond_scale']))
    for i, (_, view, s, action) in enumerate(decisions):
        stage = -1 if view is None else STAGES.index(view.stage)
        if stage != int(lat['stage'][i]):
            raise ValueError('live stage differs from replay at step %d: recorded %d, recomputed %d'
                             % (decisions[i][0], int(lat['stage'][i]), stage))
        out['stage'][i] = stage
        if action is None: continue
        label = policy_label(action, mapper, transformer)
        dpitch, dyaw = aim_delta(view.eye, s.yaw, s.pitch, target_of(s, view, wanted))
        out['labelled'][i] = True; out['label_buttons'][i] = label['buttons']; out['label_camera'][i] = label['camera']
        out['camera_on'][i] = label['camera_on']; out['in_view'][i] = in_view(dpitch, dyaw)
        out['label_pitch'][i], out['label_yaw'][i] = label['decoded_camera']
    if 'features' in lat:
        recorded, replayed = np.asarray(lat['features'], np.float32), np.asarray(feats, np.float32).reshape(n, -1)
        if recorded.shape != replayed.shape or (n and float(np.abs(recorded - replayed).max()) > FEATURE_TOLERANCE):
            raise ValueError('live features differ from replay: %s' % run_dir)
    if features: out['features'] = np.asarray(feats, np.float32).reshape(n, len(MINING_FEATURES))
    out['manifest'] = manifest; out['seed'] = request['seed']; out['teacher'] = teacher; out['block'] = wanted
    return out


def build_dataset(run_dirs, out_path, *, holdout, mapper, transformer, root, teacher='centre', forbidden_blocks=(),
                  features=False):
    """Concatenate the runs' examples; `holdout` (a run folder name, or a list of them) is kept for validation only.
    A run whose step mines one of `forbidden_blocks` is refused (multi-block design 2026-11-08: iron stays held out)."""
    import torch
    run_dirs = [Path(d) for d in run_dirs]
    names = [d.parent.name + '/' + d.name for d in run_dirs]
    wanted = [holdout] if isinstance(holdout, str) else list(holdout)
    held = []
    for name in wanted:
        matches = [i for i, d in enumerate(run_dirs) if d.name == name]
        if len(matches) != 1: raise ValueError('holdout must name exactly one run folder: %s' % name)
        held.append(matches[0])
    if not held or len(set(held)) != len(held): raise ValueError('holdout runs must be distinct and not empty')
    parts = []
    for d in run_dirs:
        part = run_examples(d, mapper, transformer, root=root, teacher=teacher, features=features)
        if part['block'] in forbidden_blocks: raise ValueError('run %s mines the held-out block %s' % (d, part['block']))
        parts.append(part)
    blocks = sorted({p['block'] for p in parts})
    scales = {p['cond_scale'] for p in parts}
    if len(scales) != 1: raise ValueError('runs use different guidance scales: %s' % sorted(scales))
    keys = ('cond', 'uncond', 'stage', 'labelled', 'label_buttons', 'label_camera', 'camera_on', 'in_view',
            'label_pitch', 'label_yaw', 'step')
    data = {k: torch.from_numpy(np.concatenate([p[k] for p in parts])) for k in keys}
    data.update(run=torch.from_numpy(np.concatenate([np.full(len(p['step']), i, np.int64) for i, p in enumerate(parts)])),
                block=torch.from_numpy(np.concatenate([np.full(len(p['step']), blocks.index(p['block']), np.int64)
                                                       for p in parts])),
                runs=names, blocks=blocks, holdout=held[0] if isinstance(holdout, str) else held,
                cond_scale=scales.pop(), width=int(parts[0]['cond'].shape[1]), stages=list(STAGES), teacher=teacher)
    if features:
        data.update(features=torch.from_numpy(np.concatenate([p['features'] for p in parts])),
                    feature_names=list(MINING_FEATURES))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    labelled = data['labelled'].numpy(); stage = data['stage'].numpy(); inview = data['in_view'].numpy()
    summary = dict(dataset=str(out_path), dataset_sha256=sha256(out_path), teacher=teacher, holdout=holdout, runs=[],
                   rows=int(len(labelled)),
                   labelled=int(labelled.sum()),
                   labelled_by_stage={s: int((labelled & (stage == i)).sum()) for i, s in enumerate(STAGES)},
                   labelled_by_block={b: int(sum(p['labelled'].sum() for p in parts if p['block'] == b)) for b in blocks},
                   labelled_in_view=int((labelled & inview).sum()), labelled_out_of_view=int((labelled & ~inview).sum()),
                   privileged='teacher labels and stages use fixture positions and the server pose',
                   source={p: sha256(Path(root) / p) for p in SOURCES if (Path(root) / p).exists()})
    if features:
        summary['features'] = list(MINING_FEATURES)
        summary['privileged'] += '; direction features to the target block of the teacher from the same positions and pose'
    for name, d, p in zip(names, run_dirs, parts):
        summary['runs'].append(dict(run=name, block=p['block'], seed=p['seed'], rows=int(len(p['step'])), labelled=int(p['labelled'].sum()),
                                    chunks=[c['sha256'] for c in p['manifest']['chunks']], checkpoint=p['manifest']['meta'].get('checkpoint')))
    out_path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary
