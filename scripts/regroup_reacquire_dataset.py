"""Layout-balanced copy of a reacquire dataset (mixed-layout design 2026-11-13 §4 amendment; no game, GPU or API).
The copy's balanced loss weighs (block, wall or new layout, stage) groups; everything else is the same dataset.
Run in the WSL steve1 env: regroup_reacquire_dataset.py IN.pt OUT.pt"""
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minecraftrl.layout_groups import regroup, run_layout  # noqa: E402


def main(argv=None):
    src, out = (Path(a) for a in (argv or sys.argv[1:]))
    if out.exists(): raise SystemExit('refusing to overwrite %s' % out)
    data = torch.load(src, weights_only=False)
    torch.save(regroup(data), out)
    layouts = {}
    for name in data['runs']: layouts[run_layout(name)] = layouts.get(run_layout(name), 0) + 1
    summary = dict(source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), dataset=str(out),
                   dataset_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), groups='block x (wall, new layout)',
                   runs_by_layout=layouts)
    out.with_suffix('.summary.json').write_bytes((json.dumps(summary, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
