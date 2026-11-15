"""Larger blueprints for the building size check (design docs/specs/2026-11-15-build-large-check-design.md).

The building tasks so far (build_task.SHAPES) have 2-4 cells. These shapes have 6-16 cells, at most two high (the
second layer is placed on the top face of the first, which the eye sees from the floor). Cells are in build order: far
rows and lower layers first. Placement, success and the teachers are those of build_recover; only the blueprint differs.
"""
import random

from .build_task import SCHEMA

_RING = [(-1, 0, 2), (0, 0, 2), (1, 0, 2), (-1, 0, 1), (1, 0, 1), (-1, 0, 0), (0, 0, 0), (1, 0, 0)]
LARGE_SHAPES = {
    'box8': [(0, 0, 1), (1, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 0), (1, 1, 0)],
    'hut16': _RING + [(x, 1, z) for x, _, z in _RING],       # 3x3 walls two high around one empty cell
    'ring8': _RING,
    'wall6': [(-1, 0, 0), (0, 0, 0), (1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0)],
}


def large_build(seed):
    """A larger blueprint for task id `seed`: the shape cycles with the id (sorted names, so 8 consecutive ids cover each
    shape twice), at an anchor 3-4 ahead and -1..1 sideways drawn from the id like build_task.generate_build."""
    shape = sorted(LARGE_SHAPES)[seed % len(LARGE_SHAPES)]
    rng = random.Random(seed); dx, dz = rng.randint(-1, 1), rng.randint(3, 4)
    return dict(schema=SCHEMA, seed=seed, shape=shape, cells=[[dx + c[0], c[1], dz + c[2]] for c in LARGE_SHAPES[shape]],
                block='cobblestone')
