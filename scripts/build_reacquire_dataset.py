"""Build the reacquire training set from recorded probe runs on the training seeds (no game, GPU or API).

Usage: build_reacquire_dataset.py --runs RUN_DIR... --holdout RUN_NAME... [--forbid-block B] --out results/.../data/r0.pt --root R
Writes the dataset (torch) and <out>.summary.json. Refuses runs on development or evaluation seeds.
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--holdout', required=True, nargs='+', help='folder name(s) of the run(s) kept for validation')
    p.add_argument('--forbid-block', action='append', default=[],
                   help='refuse runs that mine this block (multi-block design 2026-11-08: iron_ore stays held out)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--teacher', choices=('centre', 'hold', 'approach', 'press', 'sight'), default='centre',
                   help="aim label while the crosshair is on the target: 'centre' (design §4), 'hold' (keep still), "
                        "'approach' (hold, walk closer than 2 blocks first and walk onto the drop; design 2026-11-09) or "
                        "'press' (approach, with attack held on every mining-stage row; design 2026-11-10) or 'sight' "
                        "(press, with a near label that walks into sight of a hidden target; line-of-sight design 2026-11-10)")
    p.add_argument('--features', choices=('on', 'aim', 'off'), default='off',
                   help="add privileged direction features: 'on' to the target block centre (mining direction input "
                        "design 2026-11-14), 'aim' to the sight teacher's aim point (mining aim point design 2026-11-14)")
    args = p.parse_args(argv)
    root = args.root.resolve(); out = args.out.resolve()
    if not out.is_relative_to(root / 'results'): raise SystemExit('output must be under results/')
    if out.exists(): raise SystemExit('refusing to overwrite %s' % out)
    sys.path.insert(0, str(root / 'external/steve1'))
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.actions import ActionTransformer
    from minecraftrl.reacquire_data import build_dataset
    from minecraftrl.reacquire_teacher import AIM_FEATURES
    features = {'on': True, 'aim': AIM_FEATURES, 'off': False}[args.features]
    holdout = args.holdout[0] if len(args.holdout) == 1 else args.holdout
    summary = build_dataset(sorted(d.resolve() for d in args.runs), out, holdout=holdout,
                            mapper=CameraHierarchicalMapping(n_camera_bins=11),
                            transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS), root=root, teacher=args.teacher,
                            forbidden_blocks=tuple(args.forbid_block), features=features)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('runs', 'source')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
