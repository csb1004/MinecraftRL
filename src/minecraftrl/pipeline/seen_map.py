"""Seen map: what the body sensors measured, moved into world cells by the client pose (sensor-tier design §2.4).

Rules (design §2.4): every cell keeps its evidence (tick, source); a new measurement overwrites the old one; cells a
ray passed through are air; a cell never measured is 'unknown', neither air nor solid; one map per episode.

Limit: an outline ray can pass through a cell holding a small-shape block (torch, flower) without hitting it, so such
a cell may be marked air. The voxels around the feet are applied after the rays and win within a frame.
"""
import math

from .sensor import CROSSHAIR, VOXEL_OFFSETS

UNKNOWN = 'unknown'
FLUID_KINDS = ('water', 'lava')
PASSABLE = ('air', 'fire', 'soul_fire') + FLUID_KINDS     # what neither stops a pick nor holds the feet up (fluids aside)
EPS = 1e-6


def traverse(origin, direction, max_dist):
    """Yield (cell, entry distance) of every cell a ray from origin crosses up to max_dist, in order (Amanatides-Woo)."""
    cell = [math.floor(c) for c in origin]; step = [0, 0, 0]; t_max = [math.inf] * 3; t_delta = [math.inf] * 3
    for a in range(3):
        d = direction[a]
        if d > 0: step[a] = 1; t_max[a] = (cell[a] + 1 - origin[a]) / d; t_delta[a] = 1 / d
        elif d < 0: step[a] = -1; t_max[a] = (origin[a] - cell[a]) / -d; t_delta[a] = -1 / d
    t = 0.
    while t <= max_dist:
        yield tuple(cell), t
        a = min(range(3), key=lambda i: t_max[i])
        if t_max[a] == math.inf: return
        t = t_max[a]; cell[a] += step[a]; t_max[a] += t_delta[a]


class SeenMap:
    def __init__(self):
        self.cells = {}

    def mark(self, cell, kind, tick, source):
        self.cells[tuple(cell)] = (kind, int(tick), source)

    def kind(self, cell):
        found = self.cells.get(tuple(cell))
        return UNKNOWN if found is None else found[0]

    def evidence(self, cell):
        found = self.cells.get(tuple(cell))
        return None if found is None else dict(kind=found[0], tick=found[1], source=found[2])

    def update(self, frame):
        rng = frame.layout['range']
        for i, (kind, dist, cell, direction) in enumerate(zip(frame.ray_kinds, frame.ray_dist, frame.ray_cells, frame.ray_dirs)):
            if i == CROSSHAIR:
                # The client pick is reach-limited and an entity can take it: only its hit cell is evidence.
                if cell is not None: self.mark(cell, kind, frame.tick, 'ray')
                continue
            solid = dist if dist >= 0 and cell is not None else rng
            fluid_d = frame.ray_fluid_dist[i]
            wet = frame.ray_fluids[i] if frame.ray_fluids[i] in FLUID_KINDS and 0 <= fluid_d < solid else None
            fluid_cell = None
            if wet: fluid_cell = tuple(math.floor(frame.eye[a] + direction[a] * (fluid_d + 1e-4)) for a in range(3))
            for c, entry in traverse(frame.eye, direction, solid):
                if entry >= solid - EPS or c == cell: break
                if c == fluid_cell: self.mark(c, wet, frame.tick, 'ray'); break
                self.mark(c, 'air', frame.tick, 'ray')
            if cell is not None and dist >= 0: self.mark(cell, kind, frame.tick, 'ray')
        fx, fy, fz = frame.foot
        for (dx, dy, dz), kind, fluid in zip(VOXEL_OFFSETS, frame.voxel_kinds, frame.voxel_fluids):
            self.mark((fx + dx, fy + dy, fz + dz), fluid if fluid in FLUID_KINDS else kind, frame.tick, 'voxel')

    def cast(self, origin, direction, reach):
        """What a pick along direction would meet on this map: (kind, cell) of the first non-passable cell,
        ('air', None) when only passable cells lie within reach, or ('unknown', cell) at the first unmeasured cell."""
        for c, _ in traverse(origin, direction, reach):
            k = self.kind(c)
            if k == UNKNOWN: return UNKNOWN, c
            if k not in PASSABLE: return k, c
        return 'air', None

    def depth_below(self, cell, limit=8):
        """(blocks the feet would fall from `cell`, landing kind), or None when the floor is unmeasured or deeper than
        limit. Landing in a fluid counts as landing (depth up to the fluid)."""
        x, y, z = cell
        for depth in range(limit + 1):
            k = self.kind((x, y - 1 - depth, z))
            if k == UNKNOWN: return None
            if k not in ('air', 'fire', 'soul_fire'): return depth, k
        return None
