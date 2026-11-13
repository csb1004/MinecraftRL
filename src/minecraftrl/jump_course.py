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
# Game check 2026-11-13 (teacher check c101): the server position follows the keys by about 4 ticks (forward pressed at
# decision 22, the server moved at 26; the view turns at once). Jumping at 0.08 blocks before the server edge came too
# late: the client player had already walked off. Stages and the teacher therefore judge the edge and the air release at
# the position LAG_TICKS ahead at the recent server speed.
LAG_TICKS = 4
MAX_GAP_TICKS = 5      # a longer gap between samples leaves the speed unknown (0)
GROUND_EPS = .05
# 3 wide, 2.5 blocks of run-up in front of the player: with the server lag (LAG_TICKS) a shorter start left no time to jump.
START = dict(x0=-1, x1=1, z0=-1, z1=2, top=0, gap=0, block='stone_bricks')
# DemoFixture refuses offsets beyond 16 blocks (game check 2026-11-13: four jumps reached 25 and every launch failed), so
# a course has three jumps and ends right behind the goal: its barrier is at most 16 blocks ahead. Middle platforms stay 3
# long (2 long: the simulated teacher overshot and crossed only 415/800 courses).
JUMPS = 3
MID_LENGTH = 3
GOAL_LENGTH = 2
MAX_OFFSET = 16
LATERAL = 3            # platforms stay within dx -LATERAL..LATERAL
TOPS = (-1, 2)


def generate_course(seed, jumps=JUMPS):
    """A random course from `seed`: gaps 1-2, height changes -1/0/+1 (+1 only over a 1-block gap), sideways shifts of
    at most one block with some overlap, middle platforms 2 wide and 3 long, a 3x3 gold goal."""
    rng = random.Random(seed)
    while True:                                   # redraw a course that would reach past the fixture limit
        course = _draw(rng, jumps)
        if max(p['z1'] for p in course) + 1 <= MAX_OFFSET: return dict(schema=SCHEMA, seed=seed, platforms=course)


def _draw(rng, jumps):
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
        platforms.append(dict(x0=x0, x1=x0 + width - 1, z0=z0, z1=z0 + (GOAL_LENGTH if goal else MID_LENGTH) - 1, top=top,
                              gap=gap, block='gold_block' if goal else 'stone_bricks'))
    return platforms


def course_fixture(course):
    """DemoFixture config: stone and a grass fall floor (top 3 below the start), cleared air, barrier walls, then every
    platform block. The course itself rides along under `course` (DemoFixture ignores it)."""
    end = max(p['z1'] for p in course['platforms'])
    if end + 1 > MAX_OFFSET: raise ValueError('course %s reaches %d blocks ahead (limit %d)' % (course['seed'], end + 1, MAX_OFFSET))
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
    ahead: tuple | None = None   # (x, z) LAG_TICKS ahead at the recent server speed (where the client player is)


class JumpTracker:
    """Skill-internal stage machine of one course run (design §3); `update` once per valid sample in time order."""

    def __init__(self, platforms, lag=LAG_TICKS):
        """`lag`: ticks the samples trail the player (server pose: LAG_TICKS; client pose: 0)."""
        self.platforms = platforms; self.k = 0; self.best = 0      # best: furthest platform stood on
        self.lag = lag
        self.prev = None                                            # (x, z, tick) of the last sample
        self.low = min(p['top'] for p in platforms) - FALL_MARGIN

    def standing_on(self, x, y, z, on_ground):
        if not on_ground: return None
        for i, p in enumerate(self.platforms):
            if (abs(y - p['top']) <= GROUND_EPS and x + HALF > p['X0'] and x - HALF < p['X1']
                    and z + HALF > p['Z0'] and z - HALF < p['Z1']): return i
        return None

    def update(self, x, y, z, yaw, on_ground, tick=None):
        vx = vz = 0.
        if tick is not None and self.prev is not None and self.prev[2] is not None and 0 < tick - self.prev[2] <= MAX_GAP_TICKS:
            dt = tick - self.prev[2]; vx, vz = (x - self.prev[0]) / dt, (z - self.prev[1]) / dt
        self.prev = (x, z, tick); ahead = (x + vx * self.lag, z + vz * self.lag)
        on = self.standing_on(x, y, z, on_ground)
        if on is not None: self.k = on; self.best = max(self.best, on)
        last = len(self.platforms) - 1
        nxt = min(self.k + 1, last)
        cx, cz = _centre(self.platforms[nxt]); target = (cx, self.platforms[nxt]['top'], cz)
        dyaw = point_delta((x, y + EYE, z), yaw, 0., target)[1]
        if y < self.low: return JumpView('fallen', self.k, nxt, target, None, dyaw, ahead)
        if on == last: return JumpView('goal', self.k, nxt, target, None, dyaw, ahead)
        if not on_ground: return JumpView('air', self.k, nxt, target, None, dyaw, ahead)
        ux, uz = cx - x, cz - z; norm = math.hypot(ux, uz) or 1.
        edge = _exit(ahead[0], ahead[1], ux / norm, uz / norm, self.platforms[self.k])
        if abs(dyaw) > FRONT: return JumpView('turn', self.k, nxt, target, edge, dyaw, ahead)
        return JumpView('edge' if edge <= EDGE else 'run', self.k, nxt, target, edge, dyaw, ahead)


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
        ax, az = view.ahead if view.ahead is not None else (x, z)
        action['forward'] = int(past_near(ax, az, view, platforms) < PAST_NEAR)
        return action
    action['forward'] = 1
    action['jump'] = int(view.stage == 'edge')
    return action


POSES = {'server': ('server_x', 'server_y', 'server_z', 'server_on_ground', 'server_tick'),
         'client': ('client_x', 'client_y', 'client_z', 'client_on_ground', 'client_tick')}


def _pose(row, source):
    """(x, y, z, on_ground, tick) of one pose source in a recorded row, or None when it is missing."""
    keys = POSES[source]
    try:
        x, y, z = (float(row[k]) for k in keys[:3])
        ground = row[keys[3]]
    except (KeyError, TypeError, ValueError):
        return None
    if ground is None: return None
    tick = row.get(keys[4])
    return x, y, z, bool(ground), None if tick is None else float(tick)


class JumpFeed:
    """Live (and replayed) course stages from recorded telemetry rows, as LiveStageFeed does for mining.

    Two trackers (game check 2026-11-13: the server pose trails the keys by about LAG_TICKS ticks):
    - decision views (head stage input and teacher labels) follow the client pose, the player's own position at the
      moment of the action; rows without it fall back to the server pose with the LAG_TICKS prediction;
    - server views decide success and falls (the goal platform and the fall floor are judged on the server state only).
    """

    def __init__(self, fixture):
        self.course = fixture['course']; self.platforms = None; self.tracker = None; self.server = None
        self.last_view = None; self.last_server_view = None; self.last_row = None; self.last_source = None

    def start_step(self, start_t): self.start_t = start_t

    def observe(self, row):
        self.last_row = row; self.last_view = None; self.last_server_view = None; self.last_source = None
        d = row.get('demo') or {}
        if self.platforms is None:
            start = fixture_origin([row])
            if start is None: return
            self.platforms = world_platforms(self.course, *start)
            self.tracker = JumpTracker(self.platforms, lag=0); self.server = JumpTracker(self.platforms)
            self.fallback = JumpTracker(self.platforms)
        if not (row.get('server_available') == 1 and d.get('demo_available') == 1): return
        try: yaw = float(row['yaw'])
        except (KeyError, TypeError, ValueError): return
        server = _pose(row, 'server')
        if server is not None:
            x, y, z, ground, tick = server
            self.last_server_view = self.server.update(x, y, z, yaw, ground, tick)
        client = _pose(row, 'client')
        if client is not None:
            x, y, z, ground, tick = client
            self.last_view = self.tracker.update(x, y, z, yaw, ground, tick); self.last_source = 'client'
        elif self.last_server_view is not None:
            x, y, z, ground, tick = server
            self.last_view = self.fallback.update(x, y, z, yaw, ground, tick); self.last_source = 'server'

    def decision_view(self):
        """Stage of the state the next action will see (client pose), or None while unknown."""
        return self.last_view

    def server_view(self):
        """Stage on the server state of the last row: success ('goal') and falls ('fallen') are read here only."""
        return self.last_server_view

    def best(self):
        return self.server.best if self.server is not None else 0

    def label(self, row, view, source=None):
        """Teacher label for a recorded row and the decision view made from it (same pose source)."""
        x, y, z, _, _ = _pose(row, source or self.last_source or 'server')
        return teacher_action(x, y, z, float(row['yaw']), float(row['pitch']), view, self.platforms)
