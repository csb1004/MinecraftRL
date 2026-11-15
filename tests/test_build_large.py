"""Larger blueprints for the building size check (design docs/specs/2026-11-15-build-large-check-design.md)."""
import collections
import importlib.util
import unittest
from pathlib import Path

from minecraftrl import build_large as bl
from minecraftrl import build_recover as br
from minecraftrl.place_task import ARENA


class LargeBuildTest(unittest.TestCase):
    def test_the_shape_cycles_with_the_task_id_so_eight_ids_cover_each_shape_twice(self):
        shapes = [bl.large_build(k)['shape'] for k in range(1001, 1009)]
        self.assertEqual(collections.Counter(shapes), {s: 2 for s in bl.LARGE_SHAPES})
        self.assertEqual(set(bl.LARGE_SHAPES), {'box8', 'hut16', 'ring8', 'wall6'})
        self.assertEqual({s: len(c) for s, c in bl.LARGE_SHAPES.items()}, dict(box8=8, hut16=16, ring8=8, wall6=6))

    def test_cells_are_distinct_in_the_arena_off_the_start_and_at_most_two_high(self):
        for k in range(1001, 1101):
            with self.subTest(task=k):
                task = bl.large_build(k); cells = [tuple(c) for c in task['cells']]
                self.assertEqual(len(cells), len(set(cells)))
                self.assertTrue(all(ARENA['x'][0] <= x <= ARENA['x'][1] and ARENA['z'][0] <= z <= ARENA['z'][1]
                                    and 0 <= y <= 1 for x, y, z in cells))
                self.assertFalse({(x, z) for x, _, z in cells} & {(0, 0)})
                self.assertEqual((task['schema'], task['block'], task['seed']), ('build_task_v1', 'cobblestone', k))

    def test_every_upper_cell_comes_after_the_cell_below_it(self):
        for shape, cells in bl.LARGE_SHAPES.items():
            for i, (x, y, z) in enumerate(cells):
                if y: self.assertLess(cells.index((x, y - 1, z)), i, (shape, x, y, z))

    def test_the_task_makes_a_careful_fixture_like_the_small_blueprints(self):
        fixture = br.build2_fixture(bl.large_build(1005), teacher='careful')      # 1005 % 4 = 1: hut16
        self.assertEqual(fixture['build']['teacher'], 'careful'); self.assertEqual(fixture['blocks'], [])
        self.assertIn('hut16 blueprint (16 cells)', fixture['description'])

    def test_the_careful_teacher_fills_every_large_blueprint_in_the_simulator(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/simulate_build_recover.py'
        spec = importlib.util.spec_from_file_location('simulate_build_recover', path)
        sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
        for k in range(1001, 1009):
            for yaw in (0., 90., 180., 270.):
                with self.subTest(task=k, yaw=yaw):
                    r = sim.run(k, yaw, version=3, make=bl.large_build, limit=3000)
                    self.assertTrue(r['ok'], r); self.assertEqual(r['wrong'], 0)


if __name__ == '__main__':
    unittest.main()
