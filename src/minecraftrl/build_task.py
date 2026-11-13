"""Building skill (design docs/specs/2026-11-14-build-design.md): fill a small blueprint, one block placement per cell
in a fixed order (far cells and lower cells first), in the block placing arena.

Task coordinates are relative to the start block (dx right, dy up, dz forward) like DemoFixture. The current cell is the
first blueprint cell in order that the server placement record has not filled; for it the stages, the teacher and the
features are those of the block placing skill. Placed blocks (blueprint or not) are solid for the face choice.
Success is the server placement record covering every blueprint cell.
"""
import math
import random
from dataclasses import dataclass

from .pipeline.stage_replay import fixture_origin
from .place_task import (ARENA, EYE, FACE_NORMALS, GIVE_COUNT, PLACE_FEATURES, REACH, PlaceView, _at, _centre, _client,
                         _hit, choose_face, place_features)
from .place_task import teacher_action as place_teacher

SCHEMA = 'build_task_v1'
BUILD_STAGES = ('reach', 'aim', 'place', 'done')
# Blueprint cells relative to the anchor, in build order (design §1).
SHAPES = {
    'row3': [(-1, 0, 0), (0, 0, 0), (1, 0, 0)],
    'line3': [(0, 0, 2), (0, 0, 1), (0, 0, 0)],
    'pillar2': [(0, 0, 0), (0, 1, 0)],
    'corner3': [(0, 0, 1), (1, 0, 1), (0, 0, 0)],
    'stair3': [(0, 0, 1), (0, 1, 1), (0, 0, 0)],
    'square4': [(0, 0, 1), (1, 0, 1), (0, 0, 0), (1, 0, 0)],
}
LEFT_SCALE = 4.
BUILD_FEATURES = PLACE_FEATURES + ('left',)


def generate_build(seed):
    """A blueprint from `seed`: one of SHAPES at an anchor 3-4 ahead and -1..1 sideways."""
    rng = random.Random(seed)
    shape = rng.choice(sorted(SHAPES)); dx, dz = rng.randint(-1, 1), rng.randint(3, 4)
    return dict(schema=SCHEMA, seed=seed, shape=shape, cells=[[dx + c[0], c[1], dz + c[2]] for c in SHAPES[shape]],
                block='cobblestone')


def build_fixture(task):
    """DemoFixture config: the block placing arena (grass floor, cleared air, barrier walls) and the blocks to place. The
    task rides along under `build` (DemoFixture ignores it)."""
    (x0, x1), (z0, z1), top = ARENA['x'], ARENA['z'], ARENA['top']
    fill = [dict({'from': [x0, -1, z0], 'to': [x1, -1, z1]}, block='grass_block'),
            dict({'from': [x0, 0, z0], 'to': [x1, top, z1]}, block='air'),
            dict({'from': [x0 - 1, -1, z0 - 1], 'to': [x0 - 1, top + 1, z1 + 1]}, block='barrier'),
            dict({'from': [x1 + 1, -1, z0 - 1], 'to': [x1 + 1, top + 1, z1 + 1]}, block='barrier'),
            dict({'from': [x0, -1, z0 - 1], 'to': [x1, top + 1, z0 - 1]}, block='barrier'),
            dict({'from': [x0, -1, z1 + 1], 'to': [x1, top + 1, z1 + 1]}, block='barrier')]
    return dict(schema='iron_demo_fixture_v1', rotate_with_facing=True, fill=fill, blocks=[],
                give=[dict(item=task['block'], count=GIVE_COUNT)], build=task,
                description='Building task %s (design 2026-11-14): fill the %s blueprint (%d cells) with %s.'
                            % (task['seed'], task['shape'], len(task['cells']), task['block']))


def world_build(task, origin, yaw):
    """World cells of a blueprint applied at `origin` (start block) facing `yaw`, in build order, and the floor."""
    solids = set()
    for dx in range(ARENA['x'][0], ARENA['x'][1] + 1):
        for dz in range(ARENA['z'][0], ARENA['z'][1] + 1): solids.add(_at(origin, yaw, dx, -1, dz))
    return dict(cells=[_at(origin, yaw, *c) for c in task['cells']], solids=solids)


@dataclass(frozen=True)
class BuildView:
    stage: str
    target: tuple | None   # the current cell (world), None when done
    face: dict | None      # choose_face for the current cell and eye
    distance: float        # eye to the current cell centre
    filled: int
    left: int

    def place_view(self): return PlaceView(self.stage, self.target, self.face, self.distance)


class BuildTracker:
    """Skill-internal stage machine of one building run (design §2)."""

    def __init__(self, world):
        self.cells = [tuple(c) for c in world['cells']]; self.solids = set(world['solids'])
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

    def update(self, x, y, z, yaw, pitch, hit, face):
        target = self.current(); filled = len(self.filled_cells); left = len(self.cells) - filled
        if target is None: return BuildView('done', None, None, 0., filled, 0)
        eye = (x, y + EYE, z)
        distance = math.dist(eye, _centre(target)); chosen = choose_face(eye, target, self.solids)
        if distance > REACH: return BuildView('reach', target, chosen, distance, filled, left)
        if hit is not None and face is not None and 0 <= int(face) < 6:
            lands = tuple(h + k for h, k in zip(hit, FACE_NORMALS[int(face)]))
            if lands == target: return BuildView('place', target, chosen, distance, filled, left)
        return BuildView('aim', target, chosen, distance, filled, left)


def teacher_action(x, y, z, yaw, pitch, view):
    """The block placing teacher on the current cell, or None when done."""
    if view is None or view.target is None: return None
    return place_teacher(x, y, z, yaw, pitch, view.place_view())


def build_features(x, y, z, yaw, pitch, view):
    """Privileged head input in BUILD_FEATURES order: the block placing features of the current cell and the open cells
    / 4. Zeros when done or unknown."""
    if view is None or view.target is None: return [0.] * len(BUILD_FEATURES)
    return place_features(x, y, z, yaw, pitch, view.place_view()) + [view.left / LEFT_SCALE]


class BuildFeed:
    """Live (and replayed) building stages from recorded telemetry rows, as PlaceFeed does: decisions on the client pose
    and crosshair, filled cells and success on the server placement record (a count on the first row is a baseline)."""

    def __init__(self, fixture):
        self.task = fixture['build']; self.tracker = None; self.count = None
        self.last_view = None; self.last_row = None; self.unknown = 0

    def start_step(self, start_t): self.start_t = start_t

    def observe(self, row):
        self.last_row = row; self.last_view = None
        if self.tracker is None:
            start = fixture_origin([row])
            if start is None: return
            self.tracker = BuildTracker(world_build(self.task, *start))
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

    def decision_view(self): return self.last_view

    def done(self): return self.tracker is not None and self.tracker.done

    def filled(self): return 0 if self.tracker is None else len(self.tracker.filled_cells)

    def wrong(self): return 0 if self.tracker is None else self.tracker.wrong

    def label(self, row, view):
        pose = None if row is None else _client(row)
        return None if pose is None else teacher_action(*pose, view)

    def features(self):
        pose = None if self.last_row is None else _client(self.last_row)
        if pose is None or self.last_view is None: return [0.] * len(BUILD_FEATURES)
        return build_features(*pose, self.last_view)
