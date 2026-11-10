"""Can the skill stage be read from the policy latent alone? (press design 2026-11-10 §6; no game, no API)

The student's stage input is privileged: stage_replay computes it from fixture positions and the server pose. This
probe trains a small classifier from the recorded conditional policy latent (what STEVE-1 computed from the frames and
the prompt; 2048 values, recorded before the residual head) to that stage, and tests it on runs it never saw:
- `--test-prefix` runs of the same dataset (same blocks, a later round),
- `--iron` run folders (a block that is never trained on, other world seeds).
Two probes: a linear layer and one hidden layer of 256. Classes are weighted by inverse frequency. The probe is a
feasibility check only; nothing here is installed in the game.
Usage (WSL steve1 env): stage_probe.py --root R --dataset D --train-prefix official/ r1/ ... --test-prefix a-r4/
                        --iron RUN_GLOB... --out FILE
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HOLD = ('aimed', 'mining')


def scores(pred, truth, stages):
    out = dict(rows=int(len(truth)), accuracy=float((pred == truth).float().mean()) if len(truth) else None, stages={})
    recalls = []
    for i, name in enumerate(stages):
        support, called = int((truth == i).sum()), int((pred == i).sum())
        hit = int(((pred == i) & (truth == i)).sum())
        out['stages'][name] = dict(rows=support, recall=hit / support if support else None,
                                   precision=hit / called if called else None)
        if support: recalls.append(hit / support)
    out['balanced_accuracy'] = sum(recalls) / len(recalls) if recalls else None
    hold = torch.tensor([stages.index(s) for s in HOLD])
    t, p = torch.isin(truth, hold), torch.isin(pred, hold)
    out['hold_stage'] = dict(rows=int(t.sum()), recall=float((t & p).sum() / t.sum()) if int(t.sum()) else None,
                             precision=float((t & p).sum() / p.sum()) if int(p.sum()) else None)
    out['confusion'] = [[int(((truth == i) & (pred == j)).sum()) for j in range(len(stages))] for i in range(len(stages))]
    return out


def fit(model, x, y, weight, *, device, epochs, seed):
    torch.manual_seed(seed)
    model = model.to(device); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight.to(device))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for b in range(0, len(order), 512):
            idx = order[b:b + 512]
            loss = loss_fn(model(x[idx].to(device).float()), y[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def predict(model, x, device):
    with torch.no_grad():
        return torch.cat([model(x[b:b + 4096].to(device).float()).argmax(-1).cpu() for b in range(0, len(x), 4096)]) \
            if len(x) else torch.zeros(0, dtype=torch.long)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--train-prefix', nargs='+', required=True)
    p.add_argument('--test-prefix', nargs='+', required=True)
    p.add_argument('--iron', nargs='*', default=[])
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.load(args.dataset)
    stages = list(data['stages']); names = data['runs']

    def rows_of(prefixes):
        ids = torch.tensor([i for i, name in enumerate(names) if any(name.startswith(q) for q in prefixes)])
        return torch.isin(data['run'], ids) & (data['stage'] >= 0)

    train, test = rows_of(args.train_prefix), rows_of(args.test_prefix)
    if int((train & test).sum()): raise SystemExit('train and test prefixes overlap')
    mean = data['cond'][train].float().mean(0); std = data['cond'][train].float().std(0).clamp(min=1e-3)

    def norm(x): return ((x.float() - mean) / std).half()
    xtr, ytr = norm(data['cond'][train]), data['stage'][train].long()
    sets = {'test rounds (%s)' % ' '.join(args.test_prefix): (norm(data['cond'][test]), data['stage'][test].long())}
    cond, stage, iron_runs = [], [], 0
    for folder in sorted(d for pattern in args.iron for d in glob.glob(pattern)):
        folder = Path(folder)
        if not (folder / 'reacquire/manifest.json').exists(): continue
        manifest = json.loads((folder / 'reacquire/manifest.json').read_text(encoding='utf-8'))
        if manifest['stages'] != stages: raise SystemExit('other stage list in %s' % folder)
        for chunk in manifest['chunks']:
            with np.load(folder / 'reacquire' / chunk['file']) as z: cond.append(z['cond']); stage.append(z['stage'])
        iron_runs += 1
    if cond:
        c, s = torch.from_numpy(np.concatenate(cond)), torch.from_numpy(np.concatenate(stage)).long()
        sets['iron runs (%d folders)' % iron_runs] = (norm(c[s >= 0]), s[s >= 0])
    counts = torch.bincount(ytr, minlength=len(stages)).float()
    weight = torch.where(counts > 0, counts.sum() / (counts.clamp(min=1) * (counts > 0).sum()), torch.zeros_like(counts))
    width = xtr.shape[1]
    probes = dict(linear=torch.nn.Linear(width, len(stages)),
                  hidden256=torch.nn.Sequential(torch.nn.Linear(width, 256), torch.nn.ReLU(), torch.nn.Linear(256, len(stages))))
    report = dict(dataset=str(args.dataset), input='conditional policy latent (recorded before the residual head)',
                  target='privileged stage of stage_replay', stages=stages, train_prefix=args.train_prefix,
                  train_rows=int(len(ytr)), train_rows_per_stage={s: int(counts[i]) for i, s in enumerate(stages)},
                  epochs=args.epochs, seed=args.seed, probes={})
    for name, model in probes.items():
        model = fit(model, xtr, ytr, weight, device=device, epochs=args.epochs, seed=args.seed)
        result = {'train': scores(predict(model, xtr, device), ytr, stages)}
        for label, (x, y) in sets.items(): result[label] = scores(predict(model, x, device), y, stages)
        report['probes'][name] = result
        for label, r in result.items():
            fmt = lambda v: '  - ' if v is None else '%.2f' % v
            print('%-9s %-28s rows %6d acc %s balanced %s | hold stage recall %s precision %s' % (
                name, label, r['rows'], fmt(r['accuracy']), fmt(r['balanced_accuracy']), fmt(r['hold_stage']['recall']),
                fmt(r['hold_stage']['precision'])))
            print('          ' + '  '.join('%s %s/%s' % (s, fmt(v['recall']), fmt(v['precision'])) for s, v in r['stages'].items() if v['rows']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
