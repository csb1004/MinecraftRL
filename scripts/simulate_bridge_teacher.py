"""Rough offline simulation of the bridging teacher (design 2026-11-14 §3, not a game): one decision per tick, camera
turns as labelled, sneaking backward at 0.065 blocks a tick with Minecraft's sneak edge back-off (0.05 steps), a voxel
crosshair from the sneaking eye (4.5 reach), a placement when `use` lands a block in a free cell (4 ticks apart).
Usage: simulate_bridge_teacher.py [--seeds N] -> success count, median and worst decisions, wrong placements."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minecraftrl import bridge_task as bt  # noqa: E402
from minecraftrl.place_task import FACE_NORMALS  # noqa: E402

SNEAK_SPEED, WALK_SPEED, HALF, USE_DELAY = .0655, .216, .3, 4


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


def supported(x, z, row, solids):
    """The player's box over the block row `row` touches a solid block."""
    for cx in range(math.floor(x - HALF), math.floor(x + HALF - 1e-9) + 1):
        for cz in range(math.floor(z - HALF), math.floor(z + HALF - 1e-9) + 1):
            if (cx, row, cz) in solids: return True
    return False


def back_off(x, z, dx, dz, row, solids):
    """Minecraft's sneak edge back-off: shrink each axis by 0.05 until the moved box still stands on a block."""
    def shrink(v): return 0. if abs(v) < .05 else v - .05 * math.copysign(1, v)
    while dx and not supported(x + dx, z, row, solids): dx = shrink(dx)
    while dz and not supported(x, z + dz, row, solids): dz = shrink(dz)
    while dx and dz and not supported(x + dx, z + dz, row, solids): dx, dz = shrink(dx), shrink(dz)
    return dx, dz


def run(seed, start_yaw=0., limit=1200):
    task = bt.generate_bridge(seed); origin = (0, 64, 0)
    w = bt.world_bridge(task, origin, start_yaw); tr = bt.BridgeTracker(w)
    fixture = bt.bridge_fixture(task)
    solids = {bt._at(origin, start_yaw, b['dx'], b['dy'], b['dz']) for b in fixture['blocks']}
    top = int(w['top']); row = top - 1
    for dx in range(-bt.SIDE, bt.SIDE + 1):
        for dz in range(-3, 12): solids.add(bt._at(origin, start_yaw, dx, -4, dz))
    x, y, z, yaw, pitch = origin[0] + .5, float(top), origin[2] + .5, start_yaw, 0.
    last_use = -USE_DELAY
    for k in range(limit):
        ground = y == top and supported(x, z, row, solids)
        hit, face = raycast((x, y + bt.SNEAK_EYE, z), look(yaw, pitch), solids)
        view = tr.update(x, y, z, yaw, pitch, ground, hit, face)
        if view.stage == 'goal': return dict(ok=True, steps=k, wrong=tr.wrong)
        if view.stage == 'fallen' or not ground: return dict(ok=False, steps=k, wrong=tr.wrong, why='fell', filled=tr.filled())
        a = bt.teacher_action(x, y, z, yaw, pitch, view)
        if a['use'] and hit is not None and k - last_use >= USE_DELAY:
            last_use = k
            lands = tuple(h + n for h, n in zip(hit, FACE_NORMALS[face]))
            feet = (math.floor(x - HALF), math.floor(x + HALF - 1e-9), math.floor(z - HALF), math.floor(z + HALF - 1e-9))
            inside = feet[0] <= lands[0] <= feet[1] and feet[2] <= lands[2] <= feet[3] and top <= lands[1] <= top + 1
            if lands not in solids and not inside:
                solids.add(lands); tr.placed(lands)
        pitch = max(-90., min(90., pitch + float(a['camera'][0]))); yaw += float(a['camera'][1])
        speed = SNEAK_SPEED if a['sneak'] else WALK_SPEED
        v = speed * (a['back'] - a['forward'])
        dx, dz = math.sin(math.radians(yaw)) * v, -math.cos(math.radians(yaw)) * v      # backward = minus the facing
        if a['sneak']: dx, dz = back_off(x, z, dx, dz, row, solids)
        x, z = x + dx, z + dz
        if not supported(x, z, row, solids): y -= 2.                                     # walked off: falls
    return dict(ok=False, steps=limit, wrong=tr.wrong, why='limit', filled=tr.filled())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=300)
    args = p.parse_args(argv)
    seeds = list(range(args.seeds)) + list(range(101, 113)) + list(range(201, 205)) + list(range(301, 309))
    results = [(s, yaw, run(s, yaw)) for s in seeds for yaw in (0., 90., 180., 270.)]
    ok = [r for _, _, r in results if r['ok']]
    steps = sorted(r['steps'] for r in ok)
    report = dict(success=len(ok), runs=len(results), wrong=sum(r['wrong'] for _, _, r in results),
                  median_steps=statistics.median(steps) if steps else None, worst_steps=steps[-1] if steps else None,
                  failures=[dict(seed=s, yaw=yaw, **r) for s, yaw, r in results if not r['ok']][:20])
    print(json.dumps(report, indent=1))
    return 0 if len(ok) == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
