"""Jump course skill (design docs/specs/2026-11-13-jump-course-design.md): course generator, fixture, world geometry,
skill-internal stages and the privileged teacher.

A course is a start platform, three floating platforms and a gold goal platform over a lower fall floor. Coordinates
follow the fixture convention (relative to the start block: dx right, dy up, dz forward of the start facing). A
platform's `top` is the height of its upper face (its blocks sit at dy = top - 1).

The stage and the teacher read the course, the server position and the server ground flag. Those are privileged
observations: the stage input is disclosed as such, and the teacher only labels recorded states (or drives a
preparation check that is never counted as policy success).
"""
import math
import random
from dataclasses import dataclass

import numpy as np

from .pipeline.stage_replay import FACINGS, facing_index, fixture_origin
from .reacquire_teacher import BUTTONS, point_delta

SCHEMA = 'jump_course_v1'
JUMP_STAGES = ('turn', 'run', 'edge', 'air', 'fallen', 'goal')
LABELLED = ('turn', 'run', 'edge', 'air')
EYE = 1.62
HALF = .3              # half width of the player's hitbox
FRONT = 20.            # degrees of yaw: facing the next platform
EDGE = .35             # blocks from the centre to the platform edge toward the next platform: jump from here
FALL_MARGIN = 1.2      # feet this far below the lowest platform top = fallen
# Blocks past the next platform's near edge where the air label lets go of forward. Letting go also ends a sprint, so a
# sprint started for a long or rising jump does not carry into the next short jump. A rough movement simulation of the
# teacher (2026-11-13, analysis only) crossed 721/800 generated courses at 1.0, 762 at 0.5 and 788 at 0.
PAST_NEAR = 0.
MAX_TURN = 10.         # degrees per axis per decision
GROUND_EPS = .05
START = dict(x0=-1, x1=1, z0=-1, z1=1, top=0, gap=0, block='stone_bricks')
JUMPS = 4
LATERAL = 3            # platforms stay within dx -LATERAL..LATERAL
TOPS = (-1, 2)


def generate_course(seed, jumps=JUMPS):
    """A random course from `seed`: gaps 1-2, height changes -1/0/+1 (+1 only over a 1-block gap), sideways shifts of
    at most one block with some overlap, middle platforms 2 wide and 3 long, a 3x3 gold goal."""
    rng = random.Random(seed)
    platforms = [dict(START)]
    for i in range(jumps):
        prev = platforms[-1]; goal = i == jumps - 1
        width = 3 if goal else 2
        gap = rng.choice((1, 2))
        rises = (-1, 0, 0, 1) if gap == 1 else (-1, 0, 0)
        top = prev['top'] + rng.choice([r for r in rises if TOPS[0] <= prev['top'] + r <= TOPS[1]])
        centre = (prev['x0'] + prev['x1']) / 2.
        options = [x0 for x0 in range(-LATERAL, LATERAL - width + 2)
                   if abs(x0 + (width - 1) / 2. - centre) <= 1. and min(x0 + width - 1, prev['x1']) >= max(x0, prev['x0'])]
        x0 = rng.choice(options)
        z0 = prev['z1'] + gap + 1
        platforms.append(dict(x0=x0, x1=x0 + width - 1, z0=z0, z1=z0 + 2, top=top, gap=gap,
                              block='gold_block' if goal else 'stone_bricks'))
    return dict(schema=SCHEMA, seed=seed, platforms=platforms)


def course_fixture(course):
    """DemoFixture config: stone and a grass fall floor (top 3 below the start), cleared air, barrier walls, then every
    platform block. The course itself rides along under `course` (DemoFixture ignores it)."""
    end = max(p['z1'] for p in course['platforms']) + 3
    w = LATERAL + 2
    fill = [dict({'from': [-w, -8, -3], 'to': [w, -5, end]}, block='stone'),
            dict({'from': [-w, -4, -3], 'to': [w, -4, end]}, block='grass_block'),
            dict({'from': [-w, -3, -3], 'to': [w, 8, end]}, block='air'),
            dict({'from': [-w - 1, -8, -4], 'to': [-w - 1, 8, end + 1]}, block='barrier'),
            dict({'from': [w + 1, -8, -4], 'to': [w + 1, 8, end + 1]}, block='barrier'),
            dict({'from': [-w, -8, -4], 'to': [w, 8, -4]}, block='barrier'),
            dict({'from': [-w, -8, end + 1], 'to': [w, 8, end + 1]}, block='barrier'),
            dict({'from': [-w, -9, -3], 'to': [w, -9, end]}, block='barrier')]
    blocks = [dict(dx=x, dy=p['top'] - 1, dz=z, block=p['block']) for p in course['platforms']
              for x in range(p['x0'], p['x1'] + 1) for z in range(p['z0'], p['z1'] + 1)]
    return dict(schema='iron_demo_fixture_v1', rotate_with_facing=True, fill=fill, blocks=blocks, give=[], course=course,
                description='Jump course %s (design 2026-11-13): %d platforms over a fall floor 3 below the start; '
                            'the last (gold) platform is the goal.' % (course['seed'], len(course['platforms'])))


def world_platforms(course, origin, yaw):
    """Platforms in world coordinates for a fixture applied at `origin` (start block) with the start `yaw`: block
    bounds [X0, X1] x [Z0, Z1] of the upper face and its height."""
    i = facing_index(yaw)                  # course fixtures always rotate with the start facing
    (fx, fz), (rx, rz) = FACINGS[i], FACINGS[(i + 1) % 4]
    ox, oy, oz = origin
    out = []
    for p in course['platforms']:
        xs, zs = [], []
        for dx in (p['x0'], p['x1']):
            for dz in (p['z0'], p['z1']):
                xs.append(ox + fx * dz + rx * dx); zs.append(oz + fz * dz + rz * dx)
        out.append(dict(X0=float(min(xs)), X1=float(max(xs) + 1), Z0=float(min(zs)), Z1=float(max(zs) + 1),
                        top=float(oy + p['top']), gap=p['gap']))
    return out


def _centre(p): return (p['X0'] + p['X1']) / 2., (p['Z0'] + p['Z1']) / 2.


def _exit(px, pz, ux, uz, p):
    """Distance along (ux, uz) from (px, pz) to the far side of rectangle p in that direction (0 once past it). A centre
    that overhangs the near side (landed on the edge of a platform) is a whole platform away from jumping, not 0."""
    ts = []
    if ux: ts.append(((p['X1'] if ux > 0 else p['X0']) - px) / ux)
    if uz: ts.append(((p['Z1'] if uz > 0 else p['Z0']) - pz) / uz)
    return max(0., min(ts)) if ts else 0.


def _entry(px, pz, ux, uz, p):
    """Distance along (ux, uz) from (px, pz) to where the ray enters rectangle p (slab method), or None."""
    lo, hi = -math.inf, math.inf
    for o, u, a, b in ((px, ux, p['X0'], p['X1']), (pz, uz, p['Z0'], p['Z1'])):
        if abs(u) < 1e-12:
            if not a <= o <= b: return None
            continue
        t1, t2 = (a - o) / u, (b - o) / u
        lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
    return lo if lo <= hi else None


@dataclass(frozen=True)
class JumpView:
    stage: str
    k: int                 # the platform last stood on
    nxt: int               # the platform to reach next
    target: tuple          # centre of the next platform's upper face (x, y, z)
    edge: float | None     # blocks from the centre to the edge of platform k toward the target (on the ground)
    dyaw: float | None     # yaw from the view to the target


class JumpTracker:
    """Skill-internal stage machine of one course run (design §3); `update` once per valid sample in time order."""

    def __init__(self, platforms):
        self.platforms = platforms; self.k = 0; self.best = 0      # best: furthest platform stood on
        self.low = min(p['top'] for p in platforms) - FALL_MARGIN

    def standing_on(self, x, y, z, on_ground):
        if not on_ground: return None
        for i, p in enumerate(self.platforms):
            if (abs(y - p['top']) <= GROUND_EPS and x + HALF > p['X0'] and x - HALF < p['X1']
                    and z + HALF > p['Z0'] and z - HALF < p['Z1']): return i
        return None

    def update(self, x, y, z, yaw, on_ground):
        on = self.standing_on(x, y, z, on_ground)
        if on is not None: self.k = on; self.best = max(self.best, on)
        last = len(self.platforms) - 1
        nxt = min(self.k + 1, last)
        cx, cz = _centre(self.platforms[nxt]); target = (cx, self.platforms[nxt]['top'], cz)
        dyaw = point_delta((x, y + EYE, z), yaw, 0., target)[1]
        if y < self.low: return JumpView('fallen', self.k, nxt, target, None, dyaw)
        if on == last: return JumpView('goal', self.k, nxt, target, None, dyaw)
        if not on_ground: return JumpView('air', self.k, nxt, target, None, dyaw)
        ux, uz = cx - x, cz - z; norm = math.hypot(ux, uz) or 1.
        edge = _exit(x, z, ux / norm, uz / norm, self.platforms[self.k])
        if abs(dyaw) > FRONT: return JumpView('turn', self.k, nxt, target, edge, dyaw)
        return JumpView('edge' if edge <= EDGE else 'run', self.k, nxt, target, edge, dyaw)


def _turn(dpitch, dyaw):
    return np.array([max(-MAX_TURN, min(MAX_TURN, dpitch)), max(-MAX_TURN, min(MAX_TURN, dyaw))])


def past_near(x, z, view, platforms):
    """Blocks the player is past the near edge of the next platform, along the line between the two platform centres."""
    (ax, az), (bx, bz) = _centre(platforms[view.k]), _centre(platforms[view.nxt])
    ux, uz = bx - ax, bz - az; norm = math.hypot(ux, uz) or 1.; ux, uz = ux / norm, uz / norm
    entry = _entry(ax, az, ux, uz, platforms[view.nxt])
    if entry is None: return -math.inf
    return (x - ax) * ux + (z - az) * uz - entry


def teacher_action(x, y, z, yaw, pitch, view, platforms):
    """MineRL-style label for one state (design §4), or None on terminal stages."""
    if view.stage not in LABELLED: return None
    action = {k: 0 for k in BUTTONS}
    dpitch, dyaw = point_delta((x, y + EYE, z), yaw, pitch, view.target)
    action['camera'] = _turn(dpitch, dyaw)
    if view.stage == 'turn': return action
    cur, nxt = platforms[view.k], platforms[view.nxt]
    action['sprint'] = int(nxt['gap'] >= 2 or nxt['top'] > cur['top'])
    if view.stage == 'air':
        action['forward'] = int(past_near(x, z, view, platforms) < PAST_NEAR)
        return action
    action['forward'] = 1
    action['jump'] = int(view.stage == 'edge')
    return action


def _ground(row):
    v = row.get('server_on_ground')
    return None if v is None else bool(v)


class JumpFeed:
    """Live (and replayed) course stages from recorded telemetry rows, as LiveStageFeed does for mining."""

    def __init__(self, fixture):
        self.course = fixture['course']; self.tracker = None; self.platforms = None
        self.last_view = None; self.last_row = None

    def start_step(self, start_t): self.start_t = start_t

    def observe(self, row):
        self.last_row = row; self.last_view = None
        d = row.get('demo') or {}
        valid = row.get('server_available') == 1 and d.get('demo_available') == 1
        if self.tracker is None:
            start = fixture_origin([row])
            if start is None: return
            self.platforms = world_platforms(self.course, *start); self.tracker = JumpTracker(self.platforms)
        ground = _ground(row)
        if not valid or ground is None: return
        try: x, y, z, yaw = (float(row[k]) for k in ('server_x', 'server_y', 'server_z', 'yaw'))
        except (KeyError, TypeError, ValueError): return
        self.last_view = self.tracker.update(x, y, z, yaw, ground)

    def decision_view(self):
        return self.last_view

    def label(self, row, view):
        """Teacher label for a recorded row and its view."""
        return teacher_action(float(row['server_x']), float(row['server_y']), float(row['server_z']), float(row['yaw']),
                              float(row['pitch']), view, self.platforms)
