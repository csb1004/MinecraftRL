"""Build the building dataset (design docs/specs/2026-11-14-build-design.md §4) from recorded runs.
Run in the WSL steve1 env: make_build_dataset.py --runs RUN... --holdout NAME... --out results/.../x.pt --root R
[--features on] adds the privileged features (design §3). --teacher recover (design 2026-11-15) relabels every run with the
recovering teacher, its stages and features (always on) and names the `use` combinations."""
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
    p.add_argument('--features', choices=('on', 'off'), default='off')
    p.add_argument('--teacher', choices=('round1', 'recover'), default='round1')
    args = p.parse_args(argv)
    root = args.root.resolve(); out = args.out.resolve()
    if not out.is_relative_to(root / 'results'): raise SystemExit('output must be under results/')
    if out.exists(): raise SystemExit('refusing to overwrite %s' % out)
    sys.path.insert(0, str(root / 'external/steve1'))
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.actions import ActionTransformer
    from minecraftrl.build_data import build_build_dataset
    holdout = args.holdout[0] if len(args.holdout) == 1 else args.holdout
    if args.teacher == 'recover':
        from minecraftrl.build2_data import build2_dataset
        summary = build2_dataset(sorted(d.resolve() for d in args.runs), out, holdout=holdout,
                                 mapper=CameraHierarchicalMapping(n_camera_bins=11),
                                 transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS), root=root)
        print(json.dumps({k: v for k, v in summary.items() if k not in ('runs', 'source')}, indent=1))
        return 0
    summary = build_build_dataset(sorted(d.resolve() for d in args.runs), out, holdout=holdout,
                                 mapper=CameraHierarchicalMapping(n_camera_bins=11),
                                 transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS), root=root,
                                 features=args.features == 'on')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('runs', 'source')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
