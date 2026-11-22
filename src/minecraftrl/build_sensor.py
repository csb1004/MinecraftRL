"""Sensor-tier building (sensor-tier design 2026-11-19 §4.0 category 4, plan 2026-11-22): the round 3 recovering teacher,
stages and features, computed from a blueprint instruction, the client pose, the client crosshair and the seen map.

- The blueprint is the only part of the fixture the decision path reads (`blueprint_of`): a block matrix
  `blocks[x][y][z]` relative to the start block (x right, y up, z forward), 'air' for cells that must stay empty, null
  for cells the blueprint does not care about, and the build order. The same format can hold knowledge
  (KB structure_template, knowledge-Q design §4.4) or a structure-block export.
- The start pose is the first sensor frame's feet block and yaw.
- Solids, floor, standpoints and filled cells come from the seen map; an unseen cell is neither floor nor free.
- Success, filled count and wrong placements for the report stay with the server placement record (judge).
"""
from .build_recover import (BUILD2_FEATURES, BUILD2_STAGES, SENSOR_TEACHER, RecoverFeed, RecoverTracker, build2_features,
                            teacher_action)
from .build_task import build_fixture
from .pipeline.seen_map import PASSABLE
from .place_task import _at

BLUEPRINT_SCHEMA = 'blueprint_v1'
SENSOR = SENSOR_TEACHER
SENSOR_BUILD_STAGES = BUILD2_STAGES
SENSOR_BUILD_FEATURES = tuple('s_' + name for name in BUILD2_FEATURES)


def blueprint_of(task):
    """The instruction part of a building task as a blueprint_v1 block matrix (nothing of the fixture's world)."""
    cells = [tuple(int(v) for v in c[:3]) for c in task['cells']]
    kinds = [c[3] if len(c) > 3 else task['block'] for c in task['cells']]
    low = [min(c[a] for c in cells) for a in range(3)]
    size = [max(c[a] for c in cells) - low[a] + 1 for a in range(3)]
    blocks = [[[None] * size[2] for _ in range(size[1])] for _ in range(size[0])]
    for (x, y, z), kind in zip(cells, kinds): blocks[x - low[0]][y - low[1]][z - low[2]] = kind
    return dict(schema=BLUEPRINT_SCHEMA, name=task.get('shape'), frame='start block; x right, y up, z forward',
                size=size, min=low, blocks=blocks, order=[list(c) for c in cells])


def sensor_fixture(task):
    """build_task.build_fixture with the sensor teacher named in the task (same arena and judge); a task's `clutter`
    blocks are set as in the occlusion check. The feed reads only the task's blueprint from it."""
    fixture = build_fixture(task)
    fixture['build'] = dict(task, teacher=SENSOR)
    fixture['description'] += ' Sensor tier (plan 2026-11-22): blueprint instruction, client pose, crosshair, seen map.'
    if task.get('clutter'):
        fixture['blocks'] = [dict(dx=c[0], dy=c[1], dz=c[2], block=task['block']) for c in task['clutter']]
    return fixture


def blueprint_cells(blueprint):
    """[((dx, dy, dz), block)] to place, in build order: the blueprint's `order`, else bottom up and far first."""
    low = blueprint['min']; blocks = blueprint['blocks']

    def at(c): return blocks[c[0] - low[0]][c[1] - low[1]][c[2] - low[2]]
    if blueprint.get('order'):
        return [(tuple(c), at(c)) for c in blueprint['order']]
    sx, sy, sz = blueprint['size']
    cells = [(x + low[0], y + low[1], z + low[2]) for x in range(sx) for y in range(sy) for z in range(sz)]
    cells = [c for c in cells if at(c) not in (None, 'air')]
    return [(c, at(c)) for c in sorted(cells, key=lambda c: (c[1], -c[2], c[0]))]


def blueprint_world(blueprint, origin, yaw):
    return [_at(origin, yaw, *c) for c, _ in blueprint_cells(blueprint)]


class SensorBuildTracker(RecoverTracker):
    """RecoverTracker on the seen map: no fixture floor, stand list or server placements."""

    def __init__(self, blueprint, origin, yaw):
        self.cells = blueprint_world(blueprint, origin, yaw); self.level = int(origin[1])
        self.solids = set(); self.stand = []; self.filled_cells = set(); self.wrong = 0

    def sync(self, seen):
        """Refresh from the seen map: measured solid cells, filled blueprint cells, and the floor-level cells a player
        can stand in (measured floor below, measured air at the feet and the head)."""
        self.solids = {c for c, (kind, _, _) in seen.cells.items() if kind not in PASSABLE}
        self.filled_cells = {c for c in self.cells if c in self.solids}
        level = self.level
        self.stand = sorted((x, level, z) for (x, y, z) in self.solids
                            if y == level - 1 and seen.kind((x, level, z)) == 'air' and seen.kind((x, level + 1, z)) == 'air')

    def standpoint(self, x, y, z, target, usable=None):
        if not self.stand: return None, None
        return super().standpoint(x, y, z, target, usable)


class SensorBuildFeed:
    """Live and replayed sensor building feed. Decisions (stage, features, teacher label) come from sensor frames and the
    seen map (`observe_frame`); `observe(row)` only feeds the judge (server placement record) for success and the report."""

    def __init__(self, fixture):
        self.blueprint = blueprint_of(fixture['build'])
        self.judge = RecoverFeed(fixture)
        self.tracker = None; self.last_view = None; self.last_frame = None; self.last_row = None

    def start_step(self, start_t):
        self.start_t = start_t; self.judge.start_step(start_t)

    def observe(self, row):
        self.last_row = row; self.judge.observe(row)

    def observe_frame(self, frame, seen):
        if frame is None: return
        if self.tracker is None: self.tracker = SensorBuildTracker(self.blueprint, frame.foot, frame.yaw)
        self.tracker.sync(seen); self.last_frame = frame
        kind, cell = frame.crosshair
        hit = cell if kind != 'air' else None
        face = frame.hit_face if hit is not None else None
        x, y, z = frame.position
        self.last_view = self.tracker.update(x, y, z, frame.yaw, frame.pitch, hit, face)

    def decision_view(self): return self.last_view

    def done(self): return self.judge.done()

    def filled(self): return self.judge.filled()

    def wrong(self): return self.judge.wrong()

    @property
    def unknown(self): return self.judge.unknown

    def label(self, row, view):
        f = self.last_frame
        return None if f is None or view is None else teacher_action(*f.position, f.yaw, f.pitch, view)

    def features(self):
        f = self.last_frame
        if f is None or self.last_view is None: return [0.] * len(SENSOR_BUILD_FEATURES)
        return build2_features(*f.position, f.yaw, f.pitch, self.last_view)
