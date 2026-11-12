"""Build the jump course dataset (design docs/specs/2026-11-13-jump-course-design.md §5) from recorded runs.
Run in the WSL steve1 env: build_jump_dataset.py --runs RUN... --holdout NAME... --out results/.../x.pt --root R"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--holdout', required=True, nargs='+', help='folder name(s) of the run(s) kept for validation')
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
    from minecraftrl.jump_data import build_jump_dataset
    holdout = args.holdout[0] if len(args.holdout) == 1 else args.holdout
    summary = build_jump_dataset(sorted(d.resolve() for d in args.runs), out, holdout=holdout,
                                 mapper=CameraHierarchicalMapping(n_camera_bins=11),
                                 transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS), root=root)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('runs', 'source')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
