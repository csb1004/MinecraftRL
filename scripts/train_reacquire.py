"""Train the reacquire residual on a dataset from build_reacquire_dataset.py (GPU when available; no game or API).

Usage: train_reacquire.py --dataset results/.../data/r0.pt --use-stage on|off --out results/.../train/r1-on --root R
Writes <out>/checkpoint.pt (or checkpoint-rejected.pt when the holdout NLL did not fall) and <out>/train-report.json.
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--use-stage', choices=('on', 'off'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--kl-coef', type=float, default=.01)
    p.add_argument('--kl-limit', type=float, default=3.)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--balance', choices=('on', 'off'), default='off',
                   help='weight the BC loss over (block, stage) groups (multi-block design 2026-11-08 §7)')
    p.add_argument('--patience', type=int, default=None,
                   help='stop after this many epochs without a new holdout NLL low and keep the best epoch')
    p.add_argument('--feature-gain', type=float, default=1.,
                   help='multiplier of the feature input for a dataset with features (jump direction input design 2026-11-14)')
    p.add_argument('--use-coef', type=float, default=0.,
                   help='weight of the use loss; needs a dataset with use_combos (building design 2026-11-15)')
    args = p.parse_args(argv)
    root = args.root.resolve(); out = args.out.resolve()
    if not out.is_relative_to(root / 'results'): raise SystemExit('output must be under results/')
    from minecraftrl.reacquire_training import train
    report = train(args.dataset.resolve(), use_stage=args.use_stage == 'on', out_dir=out, lr=args.lr, batch=args.batch,
                   epochs=args.epochs, kl_coef=args.kl_coef, kl_limit=args.kl_limit, seed=args.seed, root=root,
                   balance=args.balance == 'on', patience=args.patience, feature_gain=args.feature_gain,
                   use_coef=args.use_coef)
    keep = ('candidate', 'checkpoint', 'rows', 'initial', 'final', 'rolled_back_epochs', 'updates_kept', 'seconds',
            'reload_max_abs_diff', 'torch_peak_allocated_mib', 'best_epoch', 'stopped_early')
    print(json.dumps({k: report[k] for k in keep}, indent=1))
    return 0 if report['candidate'] else 3


if __name__ == '__main__':
    sys.exit(main())
