"""Layout-balanced groups for a reacquire dataset (mixed-layout design 2026-11-13 §4 amendment).

The balanced BC loss (reacquire_training.group_weights) weighs every (block, stage) group equally. In the mixed-layout
rounds most near-stage rows still come from the old wall runs, so the new layouts barely move the student. `regroup`
splits each block into its wall rows and its new-layout rows, so that the loss weighs (block, wall or new, stage).
"""
import re

LAYOUT_OF = {271828: 'high', 314159: 'floor', 161803: 'pillar'}   # design §3, rounds sampled 9 and later
FIRST_LAYOUT_SAMPLING = 9


def run_layout(name):
    """'wall' or the training layout of a dataset run name ('<arm>/<block>-s<seed>-n<n>')."""
    m = re.search(r'-s(\d+)-n(\d+)$', name)
    if not m: raise ValueError('not a run name: %s' % name)
    seed, n = int(m.group(1)), int(m.group(2))
    return LAYOUT_OF[seed] if n >= FIRST_LAYOUT_SAMPLING else 'wall'


def regroup(data):
    """A copy of `data` whose `block` index is 2 * block + (run on a new layout) and whose `blocks` names both groups."""
    import torch
    new = torch.tensor([run_layout(name) != 'wall' for name in data['runs']], dtype=torch.long)
    out = dict(data)
    out['block'] = data['block'].long() * 2 + new[data['run'].long()]
    out['blocks'] = [name + suffix for name in data['blocks'] for suffix in ('', '@new')]
    out['groups'] = 'block x (wall, new layout)'
    return out
