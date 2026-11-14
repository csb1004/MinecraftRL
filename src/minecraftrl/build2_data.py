"""Training examples for the recovering building head (design docs/specs/2026-11-15-build-recover-design.md §4).

Same contract as build_data (every policy action has a recorded latent; its label is the teacher's on the last telemetry
row before it), but stages, features and labels always come from the recovering feed (build_recover.RecoverFeed):
- a run recorded with the round 1-2 feed is relabelled: its states are real, its recorded stages and features belong
  to the round 1 lists and are not compared;
- a run recorded with the recovering feed must match the replay, as in build_data.
The dataset also names, per official button combination, whether it presses `use` (`use_combos`), for the optional
use loss of reacquire_training.
"""
import json
from pathlib import Path

import numpy as np

from .build_data import FEATURE_TOLERANCE, GROUP
from .build_recover import BUILD2_FEATURES, BUILD2_STAGES, RecoverFeed
from .build_task import BUILD_STAGES
from .pipeline.monitor_replay import load_run
from .reacquire_data import _checked_request, load_latents, sha256
from .reacquire_teacher import policy_label

SOURCES = ('src/minecraftrl/build2_data.py', 'src/minecraftrl/build_recover.py', 'src/minecraftrl/build_task.py',
           'src/minecraftrl/place_task.py', 'src/minecraftrl/reacquire_teacher.py')


def use_combos(mapper):
    """[n combinations] bool: the official button combination presses `use`."""
    combos = mapper.BUTTONS_IDX_TO_COMBINATION
    return np.array([isinstance(combos[i], tuple) and 'use' in combos[i] for i in range(len(combos))], bool)


def build2_run_examples(run_dir, mapper, transformer, *, root):
    run_dir = Path(run_dir)
    request, fixture = _checked_request(run_dir, root)
    if not fixture.get('build'): raise ValueError('not a building run: %s' % run_dir)
    recorded = json.loads((run_dir / 'reacquire' / 'manifest.json').read_text(encoding='utf-8')).get('stages')
    if recorded == list(BUILD2_STAGES): own = True
    elif recorded == list(BUILD_STAGES): own = False
    else: raise ValueError('latent record uses another stage list: %s' % run_dir)
    manifest, lat = load_latents(run_dir / 'reacquire', tuple(recorded))
    if own and manifest.get('features') and list(manifest['features']) != list(BUILD2_FEATURES):
        raise ValueError('latent record uses another feature list: %s' % run_dir)
    run = load_run(run_dir)
    steps = [s for s in run.steps if s.skill == manifest['meta']['skill']]
    if len(steps) != 1: raise ValueError('expected one %s step, found %d' % (manifest['meta']['skill'], len(steps)))
    step = steps[0]; feed = RecoverFeed(fixture); decisions = []; feats = []
    for r in run.rows:
        inside = r['step'] >= step.start_step and (step.end_step is None or r['step'] <= step.end_step)
        if inside and r['t'] >= step.start_t and r.get('controller') == 'policy':
            view, seen = feed.decision_view(), feed.last_row
            action = None if view is None else feed.label(seen, view)
            decisions.append((r['step'], view, action)); feats.append(feed.features())
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
    for i, (_, view, action) in enumerate(decisions):
        stage = -1 if view is None else BUILD2_STAGES.index(view.stage)
        if own and stage != int(lat['stage'][i]):
            raise ValueError('live stage differs from replay at step %d: recorded %d, recomputed %d'
                             % (decisions[i][0], int(lat['stage'][i]), stage))
        out['stage'][i] = stage
        if action is None: continue
        label = policy_label(action, mapper, transformer)
        out['labelled'][i] = True; out['label_buttons'][i] = label['buttons']; out['label_camera'][i] = label['camera']
        out['camera_on'][i] = label['camera_on']; out['in_view'][i] = True
        out['label_pitch'][i], out['label_yaw'][i] = label['decoded_camera']
    replayed = np.asarray(feats, np.float32).reshape(n, len(BUILD2_FEATURES))
    if own and 'features' in lat:
        logged = np.asarray(lat['features'], np.float32)
        if logged.shape != replayed.shape or (n and float(np.abs(logged - replayed).max()) > FEATURE_TOLERANCE):
            raise ValueError('live features differ from replay: %s' % run_dir)
    out['features'] = replayed
    out.update(manifest=manifest, seed=request['seed'], task=fixture['build']['seed'], relabelled=not own,
               success=bool(json.loads((run_dir / 'summary.json').read_text(encoding='utf-8')).get('all_steps_completed')))
    return out


def build2_dataset(run_dirs, out_path, *, holdout, mapper, transformer, root):
    """Concatenate the runs' examples; `holdout` (run folder names) is kept for validation. One block group, so the
    balanced loss weighs the stages."""
    import torch
    run_dirs = [Path(d) for d in run_dirs]
    names = [d.parent.name + '/' + d.name for d in run_dirs]
    held = []
    for name in ([holdout] if isinstance(holdout, str) else list(holdout)):
        matches = [i for i, d in enumerate(run_dirs) if d.name == name]
        if len(matches) != 1: raise ValueError('holdout must name exactly one run folder: %s' % name)
        held.append(matches[0])
    if not held or len(set(held)) != len(held): raise ValueError('holdout runs must be distinct and not empty')
    parts = [build2_run_examples(d, mapper, transformer, root=root) for d in run_dirs]
    scales = {p['cond_scale'] for p in parts}
    if len(scales) != 1: raise ValueError('runs use different guidance scales: %s' % sorted(scales))
    keys = ('cond', 'uncond', 'stage', 'labelled', 'label_buttons', 'label_camera', 'camera_on', 'in_view',
            'label_pitch', 'label_yaw', 'step', 'features')
    data = {k: torch.from_numpy(np.concatenate([p[k] for p in parts])) for k in keys}
    data.update(run=torch.from_numpy(np.concatenate([np.full(len(p['step']), i, np.int64) for i, p in enumerate(parts)])),
                block=torch.from_numpy(np.concatenate([np.zeros(len(p['step']), np.int64) for p in parts])),
                runs=names, blocks=[GROUP], holdout=held[0] if isinstance(holdout, str) else held,
                cond_scale=scales.pop(), width=int(parts[0]['cond'].shape[1]), stages=list(BUILD2_STAGES),
                teacher='build_recover', feature_names=list(BUILD2_FEATURES),
                use_combos=torch.from_numpy(use_combos(mapper)))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    labelled = data['labelled'].numpy(); stage = data['stage'].numpy()
    summary = dict(dataset=str(out_path), dataset_sha256=sha256(out_path), teacher='build_recover', holdout=holdout, runs=[],
                   rows=int(len(labelled)), labelled=int(labelled.sum()), features=list(BUILD2_FEATURES),
                   labelled_by_stage={s: int((labelled & (stage == i)).sum()) for i, s in enumerate(BUILD2_STAGES)},
                   privileged=('teacher labels, stages and features use the blueprint, the client pose, the client crosshair and '
                               'the server placement record; success and wrong placements use the server placement record'),
                   source={p: sha256(Path(root) / p) for p in SOURCES if (Path(root) / p).exists()})
    for name, p in zip(names, parts):
        summary['runs'].append(dict(run=name, task=p['task'], seed=p['seed'], success=p['success'], rows=int(len(p['step'])),
                                    labelled=int(p['labelled'].sum()), relabelled=p['relabelled'],
                                    chunks=[c['sha256'] for c in p['manifest']['chunks']],
                                    checkpoint=p['manifest']['meta'].get('checkpoint')))
    out_path.with_suffix('.summary.json').write_bytes((json.dumps(summary, indent=2) + '\n').encode('utf-8'))
    return summary
