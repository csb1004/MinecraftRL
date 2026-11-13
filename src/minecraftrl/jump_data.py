"""Training examples for the jump course head from recorded runs (jump course design 2026-11-13 §4-§5).

Same contract as reacquire_data for mining: every dispatched policy action of the course step has a recorded latent,
and its label comes from the privileged course teacher on the state the policy saw when choosing it (the last telemetry
row before that action). Stages are recomputed with the JumpFeed the game used; a recorded stage that differs refuses
the run. Only runs on the training world seeds and on a course fixture are accepted.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .jump_course import JUMP_STAGES, JumpFeed
from .pipeline.monitor_replay import load_run
from .reacquire_data import _checked_request, load_latents, sha256
from .reacquire_teacher import policy_label

SOURCES = ('src/minecraftrl/jump_data.py', 'src/minecraftrl/jump_course.py', 'src/minecraftrl/reacquire_teacher.py')
GROUP = 'jump_course'


def jump_run_examples(run_dir, mapper, transformer, *, root):
    run_dir = Path(run_dir)
    request, fixture = _checked_request(run_dir, root)
    if not fixture.get('course'): raise ValueError('not a jump course run: %s' % run_dir)
    manifest, lat = load_latents(run_dir / 'reacquire', JUMP_STAGES)
    run = load_run(run_dir)
    steps = [s for s in run.steps if s.skill == manifest['meta']['skill']]
    if len(steps) != 1: raise ValueError('expected one %s step, found %d' % (manifest['meta']['skill'], len(steps)))
    step = steps[0]; feed = JumpFeed(fixture); decisions = []
    for r in run.rows:
        inside = r['step'] >= step.start_step and (step.end_step is None or r['step'] <= step.end_step)
        if inside and r['t'] >= step.start_t and r.get('controller') == 'policy':
            view, seen, source = feed.decision_view(), feed.last_row, feed.last_source
            action = None if view is None else feed.label(seen, view, source)
            decisions.append((r['step'], view, action))
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
        stage = -1 if view is None else JUMP_STAGES.index(view.stage)
        if stage != int(lat['stage'][i]):
            raise ValueError('live stage differs from replay at step %d: recorded %d, recomputed %d'
                             % (decisions[i][0], int(lat['stage'][i]), stage))
        out['stage'][i] = stage
        if action is None: continue
        label = policy_label(action, mapper, transformer)
        out['labelled'][i] = True; out['label_buttons'][i] = label['buttons']; out['label_camera'][i] = label['camera']
        out['camera_on'][i] = label['camera_on']; out['in_view'][i] = True
        out['label_pitch'][i], out['label_yaw'][i] = label['decoded_camera']
    out.update(manifest=manifest, seed=request['seed'], course=fixture['course']['seed'],
               success=bool(json.loads((run_dir / 'summary.json').read_text(encoding='utf-8')).get('all_steps_completed')))
    return out


def build_jump_dataset(run_dirs, out_path, *, holdout, mapper, transformer, root):
    """Concatenate the runs' examples; `holdout` (run folder names) is kept for validation. Every row is in one block
    group, so the balanced loss weighs the stages (reacquire_training.group_weights)."""
    import torch
    run_dirs = [Path(d) for d in run_dirs]
    names = [d.parent.name + '/' + d.name for d in run_dirs]
    held = []
    for name in ([holdout] if isinstance(holdout, str) else list(holdout)):
        matches = [i for i, d in enumerate(run_dirs) if d.name == name]
        if len(matches) != 1: raise ValueError('holdout must name exactly one run folder: %s' % name)
        held.append(matches[0])
    if not held or len(set(held)) != len(held): raise ValueError('holdout runs must be distinct and not empty')
    parts = [jump_run_examples(d, mapper, transformer, root=root) for d in run_dirs]
    scales = {p['cond_scale'] for p in parts}
    if len(scales) != 1: raise ValueError('runs use different guidance scales: %s' % sorted(scales))
    keys = ('cond', 'uncond', 'stage', 'labelled', 'label_buttons', 'label_camera', 'camera_on', 'in_view',
            'label_pitch', 'label_yaw', 'step')
    data = {k: torch.from_numpy(np.concatenate([p[k] for p in parts])) for k in keys}
    data.update(run=torch.from_numpy(np.concatenate([np.full(len(p['step']), i, np.int64) for i, p in enumerate(parts)])),
                block=torch.from_numpy(np.concatenate([np.zeros(len(p['step']), np.int64) for p in parts])),
                runs=names, blocks=[GROUP], holdout=held[0] if isinstance(holdout, str) else held,
                cond_scale=scales.pop(), width=int(parts[0]['cond'].shape[1]), stages=list(JUMP_STAGES), teacher='course')
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    labelled = data['labelled'].numpy(); stage = data['stage'].numpy()
    summary = dict(dataset=str(out_path), dataset_sha256=sha256(out_path), teacher='course', holdout=holdout, runs=[],
                   rows=int(len(labelled)), labelled=int(labelled.sum()),
                   labelled_by_stage={s: int((labelled & (stage == i)).sum()) for i, s in enumerate(JUMP_STAGES)},
                   privileged=('teacher labels and stages use the course and the client pose and ground flag '
                               '(rows without it: server pose plus a 4-tick speed prediction); success and falls use '
                               'the server pose'),
                   source={p: sha256(Path(root) / p) for p in SOURCES if (Path(root) / p).exists()})
    for name, p in zip(names, parts):
        summary['runs'].append(dict(run=name, course=p['course'], seed=p['seed'], success=p['success'], rows=int(len(p['step'])),
                                    labelled=int(p['labelled'].sum()), chunks=[c['sha256'] for c in p['manifest']['chunks']],
                                    checkpoint=p['manifest']['meta'].get('checkpoint')))
    out_path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary
