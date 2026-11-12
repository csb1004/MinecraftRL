"""Rough Minecraft 1.16 movement simulation of the jump course teacher (design 2026-11-13 §4; analysis only, never a
game or policy result).
Per tick: input acceleration, move with platform collisions, then friction; jump vy 0.42, gravity (vy-0.08)*0.98,
sprint jump boost 0.2. Usage: python scripts/simulate_jump_teacher.py ROOT [first_course last_course]  (prints outcome counts)"""
import math
import sys
from collections import Counter

sys.path.insert(0, sys.argv[1] + '/src')
from minecraftrl import jump_course as jc  # noqa: E402

HALF, HEIGHT = .3, 1.8


def blocks_of(platforms):
    return [(p['X0'], p['X1'], p['Z0'], p['Z1'], p['top'] - 1., p['top']) for p in platforms]


def overlaps(x, y, z, b):
    return x + HALF > b[0] and x - HALF < b[1] and z + HALF > b[2] and z - HALF < b[3] and y + HEIGHT > b[4] and y < b[5]


def simulate(seed, ticks=1800, yaw0=0., jitter=0.):
    course = jc.generate_course(seed)
    plats = jc.world_platforms(course, (0, 64, 0), yaw0)
    blocks = blocks_of(plats); floor = 64 - 3.
    t = jc.JumpTracker(plats)
    x, y, z = .5, 64., .5; vx = vz = vy = 0.; ground = True; yaw, pitch = yaw0, 0.; sprinting = False
    for tick in range(ticks):
        view = t.update(x, y, z, yaw, ground)
        if view.stage in ('goal', 'fallen'): return view.stage, t.best, tick
        a = jc.teacher_action(x, y, z, yaw, pitch, view, plats) or {}
        cam = a.get('camera', (0., 0.)); pitch += cam[0]; yaw += cam[1]
        fwd = a.get('forward', 0); sprinting = bool(fwd and (a.get('sprint', 0) or sprinting))
        if not fwd: sprinting = False
        dx, dz = -math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        acc = (0.13 if sprinting else 0.1) if ground else (0.026 if sprinting else 0.02)
        if fwd: vx += acc * dx; vz += acc * dz
        if ground and a.get('jump'):
            vy = 0.42
            if sprinting: vx += 0.2 * dx; vz += 0.2 * dz
        # move: x, then z, then y, each blocked by platform blocks
        nx = x + vx
        if any(overlaps(nx, y, z, b) for b in blocks): vx = 0.; nx = x
        x = nx
        nz = z + vz
        if any(overlaps(x, y, nz, b) for b in blocks): vz = 0.; nz = z
        z = nz
        ny = y + vy
        def xz(b): return x + HALF > b[0] and x - HALF < b[1] and z + HALF > b[2] and z - HALF < b[3]
        if vy <= 0:
            tops = [b[5] for b in blocks if xz(b) and ny - 1e-9 <= b[5] <= y + 1e-9]
            if tops: ny = max(tops); vy = 0.; ground = True
            else: ground = False
        else:
            ground = False
            if any(overlaps(x, ny, z, b) for b in blocks): vy = 0.; ny = y
        if ny <= floor: ny = floor; vy = 0.; ground = True
        y = ny
        if not ground: vy = (vy - 0.08) * 0.98
        f = 0.546 if ground else 0.91
        vx *= f; vz *= f
    return 'timeout', t.best, ticks


def main():
    lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (101, 112)
    out = Counter(); fails = []
    for seed in range(lo, hi + 1):
        for yaw0 in (0., 90., 180., 270.):
            r = simulate(seed, yaw0=yaw0); out[r[0]] += 1
            if r[0] != 'goal': fails.append((seed, yaw0, r))
    print(dict(out))
    for f in fails[:12]:
        seed = f[0]; p = jc.generate_course(seed)['platforms']
        print(f, [(q['gap'], q['top'] - pp['top'], q['x0'] - pp['x0']) for pp, q in zip(p, p[1:])])


main()
