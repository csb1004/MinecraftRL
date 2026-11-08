"""Offline iron hold gate (multi-block design 2026-11-08 §8; no game, no optimizer step).

The multi-block student never trains on iron. Before any iron game, this scores how much it wants to hold still and
attack on iron states that were already recorded:
- iron-dataset rows (results/reacquire-20261107/data/r0-2-hold.pt: training-seed probes of the earlier experiment) in
  the aimed/mining stages whose hold teacher label is "attack, camera still", split by who played the run;
- final-eval play rows of the official arm (seed 577215, already spent) in the mining stage (no labels needed).
For each group: mean p_hold (attack pressed with the camera meta off) under the guided distribution the game samples,
for the official head and the student, and the 22-decision hold (geometric mean of p_hold to the 22nd power).
The multi-block dataset's holdout rows are scored the same way per block. Gate: the student's mean p_hold on the iron
mining rows of official runs is not below the official head's.
Run in the WSL steve1 env: offline_hold_gate.py --root R --checkpoint C --dataset D --out FILE
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HOLD_TICKS = 22


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True, help='the multi-block dataset the checkpoint was trained on')
    p.add_argument('--iron-dataset', type=Path, default=Path('results/reacquire-20261107/data/r0-2-hold.pt'))
    p.add_argument('--iron-play', default='results/reacquire-20261107/final/official/s577215-n*/reacquire/latents-*.npz')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'src')); sys.path.insert(0, str(root / 'external/steve1'))
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from minecraftrl.pipeline.stage_replay import STAGES
    from minecraftrl.reacquire_head import ReacquireHead, load_into, sha256_file
    from minecraftrl.reacquire_training import guided_logp, official_action_head

    torch.set_grad_enabled(False)
    base, width = official_action_head(root)
    student = ReacquireHead(base, width, use_stage=True)
    load_into(student, args.checkpoint, base_weights_sha256=sha256_file(root / 'weights/steve1/steve1.weights'))
    student.eval()
    mapping = CameraHierarchicalMapping(n_camera_bins=11)
    groups = list(mapping.BUTTONS_GROUPS.keys())
    combos = [mapping.BUTTONS_IDX_TO_COMBINATION[i] for i in range(len(mapping.BUTTONS_IDX_TO_COMBINATION))]
    hold = torch.tensor([c != 'inventory' and c[groups.index('attack')] == 'attack' and c[-1] == 'none' for c in combos])
    label = mapping.BUTTONS_COMBINATION_TO_IDX[tuple('attack' if g == 'attack' else 'none' for g in groups)]
    aimed, mining = STAGES.index('aimed'), STAGES.index('mining')

    def score(cond, uncond, stage, scale):
        out = {}
        for who, official in (('official', True), ('student', False)):
            values = []
            for b in range(0, len(cond), 1024):
                lp = guided_logp(student, cond[b:b + 1024], uncond[b:b + 1024], stage[b:b + 1024], scale, official=official)
                values.append(lp['buttons'].exp()[:, hold].sum(-1))
            v = torch.cat(values).numpy() if values else np.zeros(0)
            out[who] = dict(p_hold=float(v.mean()) if len(v) else None,
                            hold_22=float(np.exp(HOLD_TICKS * np.mean(np.log(np.clip(v, 1e-9, 1))))) if len(v) else None)
        return dict(rows=int(len(cond)), **out)

    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256_file(args.checkpoint),
                  scope='offline scores on recorded latents; iron rows are evaluation only (never trained on)', groups={})
    iron = torch.load(root / args.iron_dataset)
    scale = float(iron['cond_scale']); st = iron['stage'].long(); lab = iron['labelled'] & (iron['label_buttons'] == label)
    src = np.array([name.split('/')[0] for name in iron['runs']])
    for who in ('official', 'student'):
        runs = torch.tensor(src == 'official' if who == 'official' else src != 'official')[iron['run']]
        for stage_i in (aimed, mining):
            m = lab & runs & (st == stage_i)
            report['groups']['iron %s-run %s' % (who, STAGES[stage_i])] = score(iron['cond'][m], iron['uncond'][m], st[m], scale)
    cond, uncond, stage = [], [], []
    for f in sorted(glob.glob(str(root / args.iron_play))):
        with np.load(f) as z: cond.append(z['cond']); uncond.append(z['uncond']); stage.append(z['stage'])
    if cond:
        c, u, s = (torch.from_numpy(np.concatenate(x)) for x in (cond, uncond, stage)); s = s.long()
        m = s == mining
        report['groups']['iron final-play official mining'] = score(c[m], u[m], s[m], scale)
    data = torch.load(args.dataset)
    held = data['holdout'] if isinstance(data['holdout'], list) else [data['holdout']]
    in_hold = torch.isin(data['run'], torch.tensor(held, dtype=data['run'].dtype))
    dlab = data['labelled'] & (data['label_buttons'] == label); dst = data['stage'].long()
    for bi, block in enumerate(data.get('blocks', ['all'])):
        bm = (data['block'] == bi) if 'block' in data else torch.ones_like(in_hold)
        for stage_i in (aimed, mining):
            m = in_hold & bm & dlab & (dst == stage_i)
            report['groups']['holdout %s %s' % (block, STAGES[stage_i])] = score(
                data['cond'][m], data['uncond'][m], dst[m], float(data['cond_scale']))
    key = report['groups']['iron official-run mining']
    report['gate'] = dict(rule='student mean p_hold on iron official-run mining rows >= official head',
                          official=key['official']['p_hold'], student=key['student']['p_hold'],
                          passed=key['student']['p_hold'] is not None and key['student']['p_hold'] >= key['official']['p_hold'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    for name, g in report['groups'].items():
        o, s = g['official'], g['student']
        fmt = lambda x: '-' if x is None else '%.3f' % x
        print('%-40s n=%5d  official p_hold %s (22: %s)  student p_hold %s (22: %s)'
              % (name, g['rows'], fmt(o['p_hold']), fmt(o['hold_22']), fmt(s['p_hold']), fmt(s['hold_22'])))
    print('GATE', 'passed' if report['gate']['passed'] else 'failed', report['gate'])
    return 0 if report['gate']['passed'] else 3


if __name__ == '__main__':
    sys.exit(main())
