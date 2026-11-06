"""Saved frames labelled with the server's crosshair block (task E 2026-11-07: probe of the frozen STEVE-1 CNN).

A frame `frames/NNNNNN.jpg` is joined with the telemetry row of the same step. Frames with a GUI open, an
unavailable server or no matching row are skipped. The label is privileged server state, used for offline analysis
only (never a policy input).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

CLASSES = ('air', 'barrier', 'grass_block', 'dirt', 'stone', 'log', 'iron_ore', 'crafting_table', 'other')


def block_class(block):
    if block.endswith('_log'): return 'log'
    return block if block in CLASSES else 'other'


def labelled_frames(run_dir):
    run = Path(run_dir)
    rows = {}
    with open(run / 'telemetry.jsonl', encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line); rows.setdefault(int(r['step']), r)
    out = []
    for path in sorted((run / 'frames').glob('*.jpg')):
        r = rows.get(int(path.stem))
        if r is None or r.get('screen', 0) != 0 or r.get('server_available') != 1: continue
        demo = r.get('demo') or {}
        if demo.get('demo_available') != 1: continue
        out.append(dict(path=path, step=int(path.stem), label=block_class(demo.get('demo_hit_block', 'air')),
                        run=run.name, controller=r.get('controller')))
    return out


def balanced_sample(items, per_class, seed):
    """At most `per_class` items of each label, drawn with a fixed seed, in label then original order."""
    groups = defaultdict(list)
    for i, item in enumerate(items): groups[item['label']].append((i, item))
    rng, out = random.Random(seed), []
    for label in sorted(groups):
        chosen = groups[label] if len(groups[label]) <= per_class else rng.sample(groups[label], per_class)
        out += [item for _, item in sorted(chosen, key=lambda x: x[0])]
    return out
