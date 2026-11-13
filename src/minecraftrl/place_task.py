"""Block placing skill (design docs/specs/2026-11-14-place-block-design.md): one target cell in a small arena.

Coordinates of a task are relative to the start block (dx right, dy up, dz forward) and turn with the start facing like
DemoFixture. Stages and teacher labels use the client pose and the client crosshair (the player's own view); success and
wrong placements use the server placement record (`place` rows of the opt-in place monitor) only.
"""
import math
import random
from dataclasses import dataclass

import numpy as np

from .pipeline.stage_replay import FACINGS, facing_index, fixture_origin
from .reacquire_teacher import BUTTONS, point_delta

SCHEMA = 'place_task_v1'
PLACE_STAGES = ('reach', 'aim', 'place', 'done')
LABELLED = ('reach', 'aim', 'place')
EYE = 1.62
REACH = 4.0            # blocks from the eye to the target centre: closer than this the face is in reach
TOO_CLOSE = 1.3        # blocks, feet to the target centre, horizontal: step back (a block cannot go where the player is)
FRONT = 45.            # degrees of yaw: walk only toward a target this close to straight ahead
MAX_TURN = 10.         # degrees per axis per decision
GIVE_COUNT = 32
ARENA = dict(x=(-5, 5), z=(-2, 8), top=4)
# Direction ordinal of the crosshair face (workshop `hit_face`): down, up, north, south, west, east.
FACE_NORMALS = ((0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0))
PLACE_FEATURES = ('t_sin', 't_cos', 't_pitch', 't_dist', 'f_sin', 'f_cos', 'f_pitch')
FEATURE_PITCH = 60.
FEATURE_DIST = 6.


def generate_target(seed):
    """A target cell 2-5 ahead and -2..2 sideways, on the floor (dy 0) or one up (dy 1) on one stone-brick support below
    it or beside it (never above it, never behind the player's side)."""
    rng = random.Random(seed)
    dx, dz, dy = rng.randint(-2, 2), rng.randint(2, 5), rng.choice((0, 1))
    supports = []
    if dy == 1:
        supports = [list(rng.choice([(dx, 0, dz), (dx - 1, 1, dz), (dx + 1, 1, dz), (dx, 1, dz + 1)]))]
    # Lists, not tuples: the task rides along in the fixture JSON and must read back equal.
    return dict(schema=SCHEMA, seed=seed, target=[dx, dy, dz], supports=supports, block='cobblestone')


def place_fixture(task):
    """DemoFixture config: a grass floor under the arena, cleared air above it, barrier walls, the supports, and the
    blocks to place given to the player. The task rides along under `place` (DemoFixture ignores it)."""
    (x0, x1), (z0, z1), top = ARENA['x'], ARENA['z'], ARENA['top']
    fill = [dict({'from': [x0, -1, z0], 'to': [x1, -1, z1]}, block='grass_block'),
            dict({'from': [x0, 0, z0], 'to': [x1, top, z1]}, block='air'),
            dict({'from': [x0 - 1, -1, z0 - 1], 'to': [x0 - 1, top + 1, z1 + 1]}, block='barrier'),
            dict({'from': [x1 + 1, -1, z0 - 1], 'to': [x1 + 1, top + 1, z1 + 1]}, block='barrier'),
            dict({'from': [x0, -1, z0 - 1], 'to': [x1, top + 1, z0 - 1]}, block='barrier'),
            dict({'from': [x0, -1, z1 + 1], 'to': [x1, top + 1, z1 + 1]}, block='barrier')]
    blocks = [dict(dx=s[0], dy=s[1], dz=s[2], block='stone_bricks') for s in task['supports']]
    return dict(schema='iron_demo_fixture_v1', rotate_with_facing=True, fill=fill, blocks=blocks,
                give=[dict(item=task['block'], count=GIVE_COUNT)], place=task,
                description='Block placing task %s (design 2026-11-14): place one %s at %s relative to the start.'
                            % (task['seed'], task['block'], list(task['target'])))


def _at(origin, yaw, dx, dy, dz):
    i = facing_index(yaw)
    (fx, fz), (rx, rz) = FACINGS[i], FACINGS[(i + 1) % 4]
    ox, oy, oz = origin
    return ox + fx * dz + rx * dx, oy + dy, oz + fz * dz + rz * dx


def world_task(task, origin, yaw):
    """World cells of a task applied at `origin` (start block) facing `yaw`: the target and every solid cell next to the
    arena air (the supports and the floor)."""
    solids = {_at(origin, yaw, *s) for s in task['supports']}
    for dx in range(ARENA['x'][0], ARENA['x'][1] + 1):
        for dz in range(ARENA['z'][0], ARENA['z'][1] + 1): solids.add(_at(origin, yaw, dx, -1, dz))
    return dict(target=_at(origin, yaw, *task['target']), solids=solids)


def _centre(cell): return tuple(c + .5 for c in cell)


def choose_face(eye, target, solids):
    """The face to click for a block to land in `target`: a face of a solid neighbour that touches the target, seen
    from the eye (the eye is on its outer side), nearest to the eye. Returns dict(block, normal, centre) or None."""
    best = None
    for n in FACE_NORMALS:
        block = tuple(t + k for t, k in zip(target, n))
        if block not in solids: continue
        normal = tuple(-k for k in n)                               # the face of `block` that points into the target
        centre = tuple(c + .5 * k for c, k in zip(_centre(target), n))
        if sum((e - c) * k for e, c, k in zip(eye, centre, normal)) <= 0: continue
        d = math.dist(eye, centre)
        if best is None or d < best[0]: best = (d, dict(block=block, normal=normal, centre=centre))
    return None if best is None else best[1]


@dataclass(frozen=True)
class PlaceView:
    stage: str
    target: tuple          # target cell (world)
    face: dict | None      # choose_face for the current eye
    distance: float        # eye to target centre


class PlaceTracker:
    """Skill-internal stage machine of one placing run (design §3)."""

    def __init__(self, world):
        self.target = tuple(world['target']); self.solids = set(world['solids'])
        self.done = False; self.wrong = 0

    def placed(self, cell):
        """A server placement (from the place record)."""
        if tuple(cell) == self.target: self.done = True
        else: self.wrong += 1; self.solids.add(tuple(cell))

    def update(self, x, y, z, yaw, pitch, hit, face):
        """`hit`: the crosshair block (world cell) or None; `face`: its face ordinal (FACE_NORMALS)."""
        eye = (x, y + EYE, z)
        distance = math.dist(eye, _centre(self.target)); chosen = choose_face(eye, self.target, self.solids)
        if self.done: return PlaceView('done', self.target, chosen, distance)
        if distance > REACH: return PlaceView('reach', self.target, chosen, distance)
        if hit is not None and face is not None and 0 <= int(face) < 6:
            lands = tuple(h + k for h, k in zip(hit, FACE_NORMALS[int(face)]))
            if lands == self.target: return PlaceView('place', self.target, chosen, distance)
        return PlaceView('aim', self.target, chosen, distance)


def _turn(dpitch, dyaw):
    return np.array([max(-MAX_TURN, min(MAX_TURN, dpitch)), max(-MAX_TURN, min(MAX_TURN, dyaw))])


def teacher_action(x, y, z, yaw, pitch, view):
    """MineRL-style label for one state (design §4), or None when done."""
    if view is None or view.stage not in LABELLED: return None
    action = {k: 0 for k in BUTTONS}
    eye = (x, y + EYE, z)
    point = view.face['centre'] if view.stage != 'reach' and view.face is not None else _centre(view.target)
    dpitch, dyaw = point_delta(eye, yaw, pitch, point)
    action['camera'] = _turn(dpitch, dyaw)
    if view.stage == 'reach': action['forward'] = int(abs(dyaw) <= FRONT)
    elif view.stage == 'aim':
        cx, _, cz = _centre(view.target)
        action['back'] = int(math.hypot(cx - x, cz - z) < TOO_CLOSE)
    else: action['use'] = 1
    return action


def place_features(x, y, z, yaw, pitch, view):
    """Privileged head input next to the stage, in PLACE_FEATURES order: the direction to the target centre (sin, cos of
    the yaw difference, pitch difference / 60), the eye distance / 6, and the direction to the chosen face centre.
    Zeros when done or unknown; the face part is zero without a face."""
    if view is None or view.stage not in LABELLED: return [0.] * len(PLACE_FEATURES)
    eye = (x, y + EYE, z)
    tp, ty = point_delta(eye, yaw, pitch, _centre(view.target))
    out = [math.sin(math.radians(ty)), math.cos(math.radians(ty)), max(-1., min(1., tp / FEATURE_PITCH)),
           min(view.distance, FEATURE_DIST) / FEATURE_DIST]
    if view.face is None: return out + [0., 0., 0.]
    fp, fy = point_delta(eye, yaw, pitch, view.face['centre'])
    return out + [math.sin(math.radians(fy)), math.cos(math.radians(fy)), max(-1., min(1., fp / FEATURE_PITCH))]


def _client(row):
    try:
        x, y, z = (float(row[k]) for k in ('client_x', 'client_y', 'client_z'))
        return x, y, z, float(row['yaw']), float(row['pitch'])
    except (KeyError, TypeError, ValueError):
        return None


def _hit(row):
    d = row.get('demo') or {}
    if d.get('demo_hit_block', 'air') in ('air', None): return None, None
    try: return tuple(int(row['hit_' + a]) for a in 'xyz'), int(row['hit_face'])
    except (KeyError, TypeError, ValueError): return None, None


class PlaceFeed:
    """Live (and replayed) placing stages from recorded telemetry rows, as JumpFeed does for the jump course."""

    def __init__(self, fixture):
        self.task = fixture['place']; self.tracker = None; self.count = None
        self.last_view = None; self.last_row = None

    def start_step(self, start_t): self.start_t = start_t

    def observe(self, row):
        self.last_row = row; self.last_view = None
        if self.tracker is None:
            start = fixture_origin([row])
            if start is None: return
            self.tracker = PlaceTracker(world_task(self.task, *start))
        d = row.get('demo') or {}; p = row.get('place') or {}
        if not (row.get('server_available') == 1 and d.get('demo_available') == 1): return
        if p.get('place_available') == 1:
            count = int(p.get('place_count', 0))
            if self.count is None: self.count = count          # placements before the first row are not this task's
            elif count > self.count:
                self.count = count
                self.tracker.placed(tuple(int(p['place_last_' + a]) for a in 'xyz'))
        pose = _client(row)
        if pose is None: return
        hit, face = _hit(row)
        self.last_view = self.tracker.update(*pose, hit, face)

    def decision_view(self): return self.last_view

    def done(self): return self.tracker is not None and self.tracker.done

    def wrong(self): return 0 if self.tracker is None else self.tracker.wrong

    def label(self, row, view):
        pose = _client(row)
        return None if pose is None else teacher_action(*pose, view)

    def features(self):
        pose = None if self.last_row is None else _client(self.last_row)
        if pose is None or self.last_view is None: return [0.] * len(PLACE_FEATURES)
        return place_features(*pose, self.last_view)
