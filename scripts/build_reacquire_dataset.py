"""Build the reacquire training set from recorded probe runs on the training seeds (no game, GPU or API).

Usage: build_reacquire_dataset.py --runs RUN_DIR... --holdout RUN_NAME --out results/.../data/r0.pt --root R
Writes the dataset (torch) and <out>.summary.json. Refuses runs on development or evaluation seeds.
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--holdout', required=True, help='folder name of the run kept for validation')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(argv)
    root = args.root.resolve(); out = args.out.resolve()
    if not out.is_relative_to(root / 'results'): raise SystemExit('output must be under results/')
    if out.exists(): raise SystemExit('refusing to overwrite %s' % out)
    sys.path.insert(0, str(root / 'external/steve1'))
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.actions import ActionTransformer
    from minecraftrl.reacquire_data import build_dataset
    summary = build_dataset(sorted(d.resolve() for d in args.runs), out, holdout=args.holdout,
                            mapper=CameraHierarchicalMapping(n_camera_bins=11),
                            transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS), root=root)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('runs', 'source')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
