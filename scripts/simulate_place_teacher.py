"""Rough offline simulation of the place teacher (design 2026-11-14 §2; not a game): straight walking, 10-degree camera steps, a voxel
raycast for the crosshair (4.5 reach), a placement when `use` lands a block in a cell. Reports success per seed."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minecraftrl import place_task as pt  # noqa: E402

WALK = .216


def look(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    return (-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p))


def raycast(eye, d, solids, reach=4.5):
    cell = [math.floor(v) for v in eye]; step = [1 if v > 0 else -1 for v in d]
    t_max = [((c + (s > 0)) - e) / v if v else math.inf for c, e, v, s in zip(cell, eye, d, step)]
    t_delta = [abs(1 / v) if v else math.inf for v in d]
    face = None; t = 0.
    while t <= reach:
        if tuple(cell) in solids:
            return tuple(cell), face
        axis = min(range(3), key=t_max.__getitem__); t = t_max[axis]
        cell[axis] += step[axis]; t_max[axis] += t_delta[axis]
        # entered through the face whose normal points back along the step
        face = {(0, 1): 4, (0, -1): 5, (1, 1): 0, (1, -1): 1, (2, 1): 2, (2, -1): 3}[(axis, step[axis])]
    return None, None


def run(seed, start_yaw=0.):
    task = pt.generate_target(seed)
    w = pt.world_task(task, (0, 64, 0), start_yaw)
    solids = set(w['solids']); tr = pt.PlaceTracker(w)
    x, y, z, yaw, pitch = .5, 64., .5, start_yaw, 0.
    for k in range(900):
        hit, face = raycast((x, y + pt.EYE, z), look(yaw, pitch), solids)
        view = tr.update(x, y, z, yaw, pitch, hit, face)
        if view.stage == 'done': return k, tr.wrong
        a = pt.teacher_action(x, y, z, yaw, pitch, view)
        if a['use'] and hit is not None:
            lands = tuple(h + n for h, n in zip(hit, pt.FACE_NORMALS[face]))
            feet = (math.floor(x), math.floor(y), math.floor(z))
            if lands != feet and lands != (feet[0], feet[1] + 1, feet[2]):
                tr.placed(lands); solids.add(lands)
        pitch = max(-90., min(90., pitch + float(a['camera'][0]))); yaw += float(a['camera'][1])
        dx, dz = -math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        v = WALK * (a['forward'] - a['back'])
        x, z = x + dx * v, z + dz * v
    return None, tr.wrong


seeds = list(range(300)) + list(range(101, 113)) + list(range(201, 205)) + list(range(301, 309))
ok = fail = wrong = 0; steps = []; bad = []
for s in seeds:
    for yaw in (0., 90., 180., 270.):
        k, w = run(s, yaw); wrong += w
        if k is None: fail += 1; bad.append((s, yaw))
        else: ok += 1; steps.append(k)
steps.sort()
print('success %d/%d, wrong placements %d, median steps %s, worst %s' % (ok, ok + fail, wrong, steps[len(steps) // 2], steps[-1]))
print('failures', bad[:20])
