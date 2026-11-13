"""Rough offline simulation of the building teacher (design 2026-11-14 §3, not a game): one decision per tick, camera
turns as labelled, walking at 0.216 blocks a tick and stopped by placed blocks at the feet or head, a voxel crosshair
from the eye (4.5 reach), a placement when `use` lands a block in a free cell outside the player (4 ticks apart).
Usage: simulate_build_teacher.py [--seeds N] -> success count, median and worst decisions, wrong placements."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minecraftrl import build_task as ct  # noqa: E402
from minecraftrl.place_task import EYE, FACE_NORMALS  # noqa: E402

WALK, HALF, USE_DELAY = .216, .3, 4


def look(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    return (-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p))


def raycast(eye, d, solids, reach=4.5):
    cell = [math.floor(v) for v in eye]; step = [1 if v > 0 else -1 for v in d]
    t_max = [((c + (s > 0)) - e) / v if v else math.inf for c, e, v, s in zip(cell, eye, d, step)]
    t_delta = [abs(1 / v) if v else math.inf for v in d]
    face = None; t = 0.
    while t <= reach:
        if tuple(cell) in solids: return tuple(cell), face
        axis = min(range(3), key=t_max.__getitem__); t = t_max[axis]
        cell[axis] += step[axis]; t_max[axis] += t_delta[axis]
        face = {(0, 1): 4, (0, -1): 5, (1, 1): 0, (1, -1): 1, (2, 1): 2, (2, -1): 3}[(axis, step[axis])]
    return None, None


def body_cells(x, y, z):
    xs = range(math.floor(x - HALF), math.floor(x + HALF - 1e-9) + 1); zs = range(math.floor(z - HALF), math.floor(z + HALF - 1e-9) + 1)
    return {(cx, cy, cz) for cx in xs for cz in zs for cy in (math.floor(y), math.floor(y) + 1)}


def run(seed, start_yaw=0., limit=1200):
    task = ct.generate_build(seed); origin = (0, 64, 0)
    w = ct.world_build(task, origin, start_yaw); tr = ct.BuildTracker(w)
    solids = set(w['solids'])
    x, y, z, yaw, pitch = origin[0] + .5, 64., origin[2] + .5, start_yaw, 0.
    last_use = -USE_DELAY
    for k in range(limit):
        hit, face = raycast((x, y + EYE, z), look(yaw, pitch), solids)
        view = tr.update(x, y, z, yaw, pitch, hit, face)
        if view.stage == 'done': return dict(ok=True, steps=k, wrong=tr.wrong, shape=task['shape'])
        a = ct.teacher_action(x, y, z, yaw, pitch, view)
        if a['use'] and hit is not None and k - last_use >= USE_DELAY:
            last_use = k
            lands = tuple(h + n for h, n in zip(hit, FACE_NORMALS[face]))
            if lands not in solids and lands not in body_cells(x, y, z):
                solids.add(lands); tr.placed(lands)
        pitch = max(-90., min(90., pitch + float(a['camera'][0]))); yaw += float(a['camera'][1])
        v = WALK * (a['forward'] - a['back'])
        nx, nz = x - math.sin(math.radians(yaw)) * v, z + math.cos(math.radians(yaw)) * v
        if not (body_cells(nx, y, nz) & solids): x, z = nx, nz
    return dict(ok=False, steps=limit, wrong=tr.wrong, shape=task['shape'], filled=len(tr.filled_cells), cells=len(tr.cells))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=300)
    args = p.parse_args(argv)
    seeds = list(range(args.seeds)) + list(range(101, 113)) + list(range(201, 205)) + list(range(301, 309))
    results = [(s, yaw, run(s, yaw)) for s in seeds for yaw in (0., 90., 180., 270.)]
    ok = [r for _, _, r in results if r['ok']]
    steps = sorted(r['steps'] for r in ok)
    by_shape = {}
    for _, _, r in results:
        t, m = by_shape.get(r['shape'], (0, 0)); by_shape[r['shape']] = (t + 1, m + r['ok'])
    report = dict(success=len(ok), runs=len(results), wrong=sum(r['wrong'] for _, _, r in results), by_shape=by_shape,
                  median_steps=statistics.median(steps) if steps else None, worst_steps=steps[-1] if steps else None,
                  failures=[dict(seed=s, yaw=yaw, **r) for s, yaw, r in results if not r['ok']][:20])
    print(json.dumps(report, indent=1))
    return 0 if len(ok) == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
