"""Building with recovery (design docs/specs/2026-11-15-build-recover-design.md): the building skill of
build_task.py with a teacher that can still act when placed blocks hide the current cell.

Round 1-2 of building (docs/build-results-2026-11-14.md) failed whenever the student placed a block in front of the next
cell: every face of the cell was then hidden, and the round 1 teacher kept aiming at a hidden face (choose_face ignores
occlusion), so its labels in those states taught nothing. Here a face counts only when the straight line from the eye
to its centre reaches it first (`visible_face`). When no face is seen within reach, the teacher walks to a standpoint:
the nearest floor cell, free at the feet and head and not an open blueprint cell, whose eye is within STAND_REACH of
the cell and sees one of its faces. Stages: reach (farther than REACH), move (in reach, nothing seen), aim, place, done.
Success and filled cells stay those of build_task (the server placement record covers the blueprint).
"""
import math
from dataclasses import dataclass

from .build_task import BUILD_STAGES, LEFT_SCALE, BuildFeed, build_fixture
from .place_task import (ARENA, EYE, FACE_NORMALS, FEATURE_DIST, FEATURE_PITCH, FRONT, PLACE_FEATURES, REACH, TOO_CLOSE,
                         _at, _centre, _client, _hit, _turn)
from .pipeline.stage_replay import fixture_origin
from .reacquire_teacher import BUTTONS, point_delta

TEACHER = 'recover'
BUILD2_STAGES = ('reach', 'move', 'aim', 'place', 'done')
LABELLED = ('reach', 'move', 'aim', 'place')
BUILD2_FEATURES = PLACE_FEATURES + ('left', 'm_sin', 'm_cos', 'm_dist')
STAND_REACH = 3.5      # blocks, standpoint eye to the cell centre (inside REACH, so arriving ends the move)
ARRIVE = .25           # blocks, feet to the standpoint: stop walking
CLEAR_STEP = .1
HALF = .3              # player half width
MARGIN = .3            # blocks around a standpoint centre that must also see the cell (more than ARRIVE)
MARGIN_EYES = [(a, b) for a in (0., -MARGIN, MARGIN) for b in (0., -MARGIN, MARGIN)]


def raycast(eye, direction, solids, reach=4.5):
    """(first solid cell, face ordinal entered) along `direction` from `eye` within `reach`, or (None, None)."""
    cell = [math.floor(v) for v in eye]; step = [1 if v > 0 else -1 for v in direction]
    t_max = [((c + (s > 0)) - e) / v if v else math.inf for c, e, v, s in zip(cell, eye, direction, step)]
    t_delta = [abs(1 / v) if v else math.inf for v in direction]
    face = None; t = 0.
    while t <= reach:
        if tuple(cell) in solids: return tuple(cell), face
        axis = min(range(3), key=t_max.__getitem__); t = t_max[axis]
        cell[axis] += step[axis]; t_max[axis] += t_delta[axis]
        face = {(0, 1): 4, (0, -1): 5, (1, 1): 0, (1, -1): 1, (2, 1): 2, (2, -1): 3}[(axis, step[axis])]
    return None, None


def _ordinal(normal): return FACE_NORMALS.index(tuple(normal))


def visible_face(eye, target, solids):
    """choose_face among the faces whose centre the eye's straight line reaches first (no block in between), nearest to
    the eye; None when every face of the cell is hidden or there is none."""
    best = None
    for n in FACE_NORMALS:
        block = tuple(t + k for t, k in zip(target, n))
        if block not in solids: continue
        normal = tuple(-k for k in n)
        centre = tuple(c + .5 * k for c, k in zip(_centre(target), n))
        if sum((e - c) * k for e, c, k in zip(eye, centre, normal)) <= 0: continue
        d = math.dist(eye, centre)
        if best is not None and d >= best[0]: continue
        direction = tuple((c - e) / d for c, e in zip(centre, eye))
        if raycast(eye, direction, solids, reach=d + .01) != (block, _ordinal(normal)): continue
        best = (d, dict(block=block, normal=normal, centre=centre))
    return None if best is None else best[1]


def _body(x, y, z):
    xs = range(math.floor(x - HALF), math.floor(x + HALF - 1e-9) + 1)
    zs = range(math.floor(z - HALF), math.floor(z + HALF - 1e-9) + 1)
    return {(cx, cy, cz) for cx in xs for cz in zs for cy in (math.floor(y), math.floor(y) + 1)}


def path_clear(start, end, solids):
    """No solid in the player's body along the straight horizontal walk from `start` to `end` (feet points)."""
    (x0, y0, z0), (x1, _, z1) = start, end
    n = max(1, int(math.hypot(x1 - x0, z1 - z0) / CLEAR_STEP))
    return all(not (_body(x0 + (x1 - x0) * i / n, y0, z0 + (z1 - z0) * i / n) & solids) for i in range(1, n + 1))


def world_build2(task, origin, yaw):
    """build_task.world_build plus the arena floor-level cells a player can stand in."""
    cells = [_at(origin, yaw, *c) for c in task['cells']]
    solids, stand = set(), []
    for dx in range(ARENA['x'][0], ARENA['x'][1] + 1):
        for dz in range(ARENA['z'][0], ARENA['z'][1] + 1):
            solids.add(_at(origin, yaw, dx, -1, dz)); stand.append(_at(origin, yaw, dx, 0, dz))
    return dict(cells=cells, solids=solids, stand=sorted(stand))


@dataclass(frozen=True)
class RecoverView:
    stage: str
    target: tuple | None   # the current cell (world), None when done
    face: dict | None      # the face seen from the eye (visible_face), or the crosshair's face in place
    distance: float        # eye to the current cell centre
    filled: int
    left: int
    stand: tuple | None    # feet point to end the walk at in reach and move, else None
    way: tuple | None      # feet point to walk toward now (on the grid path to `stand`), else None


class RecoverTracker:
    """Skill-internal stage machine of one building run with recovery (design §2)."""

    def __init__(self, world):
        self.cells = [tuple(c) for c in world['cells']]; self.solids = set(world['solids'])
        self.stand = [tuple(c) for c in world['stand']]
        self.filled_cells = set(); self.wrong = 0

    def placed(self, cell):
        """A server placement (from the place record)."""
        cell = tuple(int(c) for c in cell); self.solids.add(cell)
        if cell in self.cells: self.filled_cells.add(cell)
        else: self.wrong += 1

    def current(self):
        return next((c for c in self.cells if c not in self.filled_cells), None)

    @property
    def done(self): return self.current() is None

    def _usable(self, cell, target):
        feet = (cell[0] + .5, float(cell[1]), cell[2] + .5)
        if math.hypot(feet[0] - target[0] - .5, feet[2] - target[2] - .5) < TOO_CLOSE: return False
        eye = (feet[0], feet[1] + EYE, feet[2])
        if math.dist(eye, _centre(target)) > STAND_REACH: return False
        # Seen from anywhere the walk may stop (ARRIVE) around the cell centre, not only from the centre.
        return all(visible_face((eye[0] + a, eye[1], eye[2] + b), target, self.solids) is not None for a, b in MARGIN_EYES)

    def standpoint(self, x, y, z, target):
        """(standpoint, waypoint) feet points, or (None, None): the usable standpoint fewest grid steps away (4-connected
        floor cells free at the feet and head; open blueprint cells may be crossed but not stood in), and the farthest
        cell of that path a straight walk from the feet reaches without touching a block."""
        stand = set(self.stand); level = self.stand[0][1]
        open_cells = {c for c in self.cells if c not in self.filled_cells}

        def free(c): return c not in self.solids and (c[0], c[1] + 1, c[2]) not in self.solids
        start = (math.floor(x), level, math.floor(z))
        prev = {start: None}; order = [start]; i = 0
        while i < len(order):
            c = order[i]; i += 1
            if c not in open_cells and free(c) and self._usable(c, target):
                path = []
                while c is not None: path.append(c); c = prev[c]
                path.reverse()
                feet = [(p[0] + .5, float(p[1]), p[2] + .5) for p in path]
                way = next((f for f in reversed(feet[1:]) if path_clear((x, y, z), f, self.solids)),
                           feet[1] if len(feet) > 1 else feet[0])
                return feet[-1], way
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + dx, level, c[2] + dz)
                if n not in prev and n in stand and free(n): prev[n] = c; order.append(n)
        return None, None

    def update(self, x, y, z, yaw, pitch, hit, face):
        target = self.current(); filled = len(self.filled_cells); left = len(self.cells) - filled
        if target is None: return RecoverView('done', None, None, 0., filled, 0, None, None)
        eye = (x, y + EYE, z)
        distance = math.dist(eye, _centre(target))
        if distance <= REACH and hit is not None and face is not None and 0 <= int(face) < 6:
            n = FACE_NORMALS[int(face)]
            if tuple(h + k for h, k in zip(hit, n)) == target:
                seen = dict(block=tuple(hit), normal=tuple(n), centre=tuple(c - .5 * k for c, k in zip(_centre(target), n)))
                return RecoverView('place', target, seen, distance, filled, left, None, None)
        seen = visible_face(eye, target, self.solids)
        if distance <= REACH and seen is not None: return RecoverView('aim', target, seen, distance, filled, left, None, None)
        stand, way = self.standpoint(x, y, z, target)
        return RecoverView('reach' if distance > REACH else 'move', target, seen, distance, filled, left, stand, way)


def teacher_action(x, y, z, yaw, pitch, view):
    """MineRL-style label for one state, or None when done or unknown. reach and move walk toward the waypoint (yaw to
    it, pitch to the cell centre, forward when it is within FRONT degrees) until the feet are within ARRIVE of the
    standpoint; without a standpoint they walk to the cell like the round 1 reach. aim is the block placing teacher on
    the seen face; place clicks without turning."""
    if view is None or view.stage not in LABELLED: return None
    action = {k: 0 for k in BUTTONS}
    eye = (x, y + EYE, z)
    if view.stage in ('reach', 'move'):
        dpitch, dyaw = point_delta(eye, yaw, pitch, _centre(view.target))
        if view.stand is not None:
            if math.hypot(view.stand[0] - x, view.stand[2] - z) > ARRIVE:
                wx, _, wz = view.way
                dyaw = point_delta(eye, yaw, pitch, (wx, eye[1], wz))[1]
                action['forward'] = int(abs(dyaw) <= FRONT)
        else: action['forward'] = int(abs(dyaw) <= FRONT)
        action['camera'] = _turn(dpitch, dyaw)
        return action
    if view.stage == 'place':
        # The crosshair already names the cell: click without a turn in the same action (round 2 students turned while
        # clicking and put 9 of 23 place-stage blocks one cell over).
        action['use'] = 1; action['camera'] = _turn(0., 0.)
        return action
    point = view.face['centre'] if view.face is not None else _centre(view.target)
    action['camera'] = _turn(*point_delta(eye, yaw, pitch, point))
    cx, _, cz = _centre(view.target)
    action['back'] = int(math.hypot(cx - x, cz - z) < TOO_CLOSE)
    return action


def build2_features(x, y, z, yaw, pitch, view):
    """Privileged head input in BUILD2_FEATURES order: the direction to the cell centre (sin, cos of the yaw difference,
    pitch difference / 60), the eye distance / 6, the direction to the seen face centre (zeros when none is seen), the
    open cells / 4, and in reach and move the direction to the waypoint (sin, cos of the yaw difference) and the
    horizontal distance to the standpoint / 6 (zeros otherwise). Zeros when done or unknown."""
    if view is None or view.stage not in LABELLED: return [0.] * len(BUILD2_FEATURES)
    eye = (x, y + EYE, z)
    tp, ty = point_delta(eye, yaw, pitch, _centre(view.target))
    out = [math.sin(math.radians(ty)), math.cos(math.radians(ty)), max(-1., min(1., tp / FEATURE_PITCH)),
           min(view.distance, FEATURE_DIST) / FEATURE_DIST]
    if view.face is None: out += [0., 0., 0.]
    else:
        fp, fy = point_delta(eye, yaw, pitch, view.face['centre'])
        out += [math.sin(math.radians(fy)), math.cos(math.radians(fy)), max(-1., min(1., fp / FEATURE_PITCH))]
    out.append(view.left / LEFT_SCALE)
    if view.stand is None: return out + [0., 0., 0.]
    my = point_delta(eye, yaw, pitch, (view.way[0], eye[1], view.way[2]))[1]
    far = math.hypot(view.stand[0] - x, view.stand[2] - z)
    return out + [math.sin(math.radians(my)), math.cos(math.radians(my)), min(far, FEATURE_DIST) / FEATURE_DIST]


class RecoverFeed(BuildFeed):
    """BuildFeed with the recovering tracker, teacher and features (same server-record rules)."""

    def observe(self, row):
        self.last_row = row; self.last_view = None
        if self.tracker is None:
            start = fixture_origin([row])
            if start is None: return
            self.tracker = RecoverTracker(world_build2(self.task, *start))
        d = row.get('demo') or {}; p = row.get('place') or {}
        if not (row.get('server_available') == 1 and d.get('demo_available') == 1): return
        if p.get('place_available') == 1:
            count = int(p.get('place_count', 0))
            if self.count is None: self.count = count
            elif count > self.count:
                self.unknown += count - self.count - 1
                self.count = count
                self.tracker.placed(tuple(int(p['place_last_' + a]) for a in 'xyz'))
        pose = _client(row)
        if pose is None: return
        hit, face = _hit(row)
        self.last_view = self.tracker.update(*pose, hit, face)

    def label(self, row, view):
        pose = None if row is None else _client(row)
        return None if pose is None else teacher_action(*pose, view)

    def features(self):
        pose = None if self.last_row is None else _client(self.last_row)
        if pose is None or self.last_view is None: return [0.] * len(BUILD2_FEATURES)
        return build2_features(*pose, self.last_view)


def build2_fixture(task):
    """build_task.build_fixture with the recovering teacher named in the task."""
    fixture = build_fixture(task)
    fixture['build'] = dict(task, teacher=TEACHER)
    fixture['description'] += ' Recovering teacher (design 2026-11-15).'
    return fixture


def recovering(fixture): return (fixture.get('build') or {}).get('teacher') == TEACHER


def feed_for(fixture): return RecoverFeed(fixture) if recovering(fixture) else BuildFeed(fixture)


def stages_for(fixture): return BUILD2_STAGES if recovering(fixture) else BUILD_STAGES

