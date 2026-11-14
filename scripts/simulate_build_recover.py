"""Rough offline simulation of the building teachers (not a game), design docs/specs/2026-11-15-build-recover-design.md.

Same model as simulate_build_teacher.py: one decision per tick, camera turns as labelled, walking at 0.216 blocks a
tick (forward or back along the yaw) and stopped by placed blocks at the feet or head, a voxel crosshair from the eye
(4.5 reach), a placement when `use` lands a block in a free cell outside the player (4 ticks apart).
version 1 is the round 1-2 teacher (build_task), version 2 the recovering teacher (build_recover), version 3 the
recovering teacher with careful place (round 4).
clutter: a placed block in front of the first cell before the start (the round 1-2 failure state).
stray: probability of an extra `use` press on a decision whose label has none (a student's stray press), seeded.
noise: standard deviation (degrees) added to each camera axis, and the same share of decisions (noise / 20) that do
nothing, a crude stand-in for a student that follows the teacher loosely.
Usage: simulate_build_recover.py [--seeds N] [--stray P ...] -> success and wrong placements per teacher and stray level.
"""
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minecraftrl import build_recover as br  # noqa: E402
from minecraftrl import build_task as ct  # noqa: E402
from minecraftrl.place_task import EYE, FACE_NORMALS, _at  # noqa: E402

WALK, HALF, USE_DELAY = .216, .3, 4


def look(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    return (-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p))


def body_cells(x, y, z):
    xs = range(math.floor(x - HALF), math.floor(x + HALF - 1e-9) + 1); zs = range(math.floor(z - HALF), math.floor(z + HALF - 1e-9) + 1)
    return {(cx, cy, cz) for cx in xs for cz in zs for cy in (math.floor(y), math.floor(y) + 1)}


def clutter_cell(task):
    """Task coordinates of a block right in front of the first cell (toward the start), when that is free."""
    dx, dy, dz = task['cells'][0]
    cell = (dx, 0, dz - 1)
    if list(cell) in task['cells'] or (dx, dz - 1) == (0, 0): return None
    return cell


def run(seed, start_yaw=0., limit=1200, version=2, clutter=False, stray=0., rng_seed=0, noise=0.):
    task = ct.generate_build(seed); origin = (0, 64, 0)
    w = br.world_build2(task, origin, start_yaw)
    tr = (br.CarefulTracker(w) if version == 3 else br.RecoverTracker(w) if version == 2
          else ct.BuildTracker(ct.world_build(task, origin, start_yaw)))
    teacher = br.teacher_action if version in (2, 3) else ct.teacher_action
    solids = set(w['solids'])
    rng = random.Random(rng_seed * 100003 + seed * 7 + int(start_yaw))
    if clutter:
        c = clutter_cell(task)
        if c is not None:
            cell = _at(origin, start_yaw, *c); solids.add(cell); tr.solids.add(cell)
    x, y, z, yaw, pitch = origin[0] + .5, 64., origin[2] + .5, start_yaw, 0.
    last_use = -USE_DELAY; wrong = 0
    for k in range(limit):
        hit, face = br.raycast((x, y + EYE, z), look(yaw, pitch), solids)
        view = tr.update(x, y, z, yaw, pitch, hit, face)
        if view.stage == 'done': return dict(ok=True, steps=k, wrong=wrong, shape=task['shape'])
        a = dict(teacher(x, y, z, yaw, pitch, view))
        if noise > 0:
            if rng.random() < noise / 20: a = dict(a, forward=0, back=0, use=0, camera=[0., 0.])
            a['camera'] = [float(a['camera'][0]) + rng.gauss(0, noise), float(a['camera'][1]) + rng.gauss(0, noise)]
        press = a['use'] or (stray > 0 and rng.random() < stray)
        if press and hit is not None and k - last_use >= USE_DELAY:
            last_use = k
            lands = tuple(h + n for h, n in zip(hit, FACE_NORMALS[face]))
            if lands not in solids and lands not in body_cells(x, y, z):
                solids.add(lands); tr.placed(lands); wrong += lands not in tr.cells
        pitch = max(-90., min(90., pitch + float(a['camera'][0]))); yaw += float(a['camera'][1])
        v = WALK * (a['forward'] - a['back'])
        nx, nz = x - math.sin(math.radians(yaw)) * v, z + math.cos(math.radians(yaw)) * v
        if not (body_cells(nx, y, nz) & solids): x, z = nx, nz
    return dict(ok=False, steps=limit, wrong=wrong, shape=task['shape'], filled=len(tr.filled_cells), cells=len(tr.cells))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=100)
    p.add_argument('--stray', type=float, nargs='+', default=[0., .01, .03])
    p.add_argument('--repeats', type=int, default=2, help='stray press draws per (seed, yaw)')
    p.add_argument('--noise', type=float, default=0.)
    args = p.parse_args(argv)
    seeds = list(range(args.seeds)) + list(range(101, 113)) + list(range(201, 205)) + list(range(301, 309))
    report = {}
    for clutter in (False, True):
        for stray in args.stray:
            for version in (1, 2):
                reps = 1 if stray == 0 and args.noise == 0 else args.repeats
                rs = [run(s, yaw, version=version, clutter=clutter, stray=stray, rng_seed=r, noise=args.noise)
                      for s in seeds for yaw in (0., 90., 180., 270.) for r in range(reps)]
                ok = [r for r in rs if r['ok']]
                report['noise=%g clutter=%s stray=%g v%d' % (args.noise, clutter, stray, version)] = dict(
                    success=len(ok), runs=len(rs), wrong=sum(r['wrong'] for r in rs),
                    median_steps=statistics.median(sorted(r['steps'] for r in ok)) if ok else None)
                print(json.dumps({k: report[k] for k in list(report)[-1:]}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
