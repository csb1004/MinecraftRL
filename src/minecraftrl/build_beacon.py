"""Beacon pyramid demo (user request 2026-11-23): a multi-material blueprint that fits the building head's two layers.

A 3x3 base of mineral blocks (gold corners, iron edges, a diamond centre) and a beacon on the centre, 10 cells. The
second layer is placed on the top face of the centre, which the eye sees from the floor. Cells carry their block as a
4th value; the fixture gives each block (sensor_fixture), the session's slot-selection rule holds the current cell's
block, and the judge counts a block of another kind in a blueprint cell as wrong. Cells are in build order: far row
first, left to right, then the beacon.
"""
import random

from .build_task import SCHEMA

BEACON_KINDS = ('gold_block', 'iron_block', 'diamond_block', 'beacon')


def _base_kind(x, z):
    """Block of base cell (x, z), x and z in -1..1 around the centre."""
    if (x, z) == (0, 0): return 'diamond_block'
    return 'gold_block' if x and z else 'iron_block'


def beacon_build(seed):
    """The beacon pyramid for task id `seed`, its centre 4-5 ahead and -1..1 sideways (anchor drawn from the id like
    build_task.generate_build)."""
    rng = random.Random(seed); dx, dz = rng.randint(-1, 1), rng.randint(3, 4)
    cells = [[dx + x, 0, dz + 1 + z, _base_kind(x, z)] for z in (1, 0, -1) for x in (-1, 0, 1)]
    cells.append([dx, 1, dz + 1, 'beacon'])
    return dict(schema=SCHEMA, seed=seed, shape='beacon', cells=cells, block=cells[0][3])
