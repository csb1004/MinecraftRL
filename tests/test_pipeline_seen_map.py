"""Seen map: sensor memory in world cells (sensor-tier design 2026-11-19 §2.4; plan 2026-11-22 task 3)."""
import unittest

from minecraftrl.pipeline.seen_map import SeenMap, traverse
from minecraftrl.pipeline.sensor import sensor_frame
from test_pipeline_sensor import RAYS, info


def frame(tick=500, **sensor):
    return sensor_frame(info(sensor_client_tick=tick, **sensor), 0.)


MISS = dict(sensor_ray_kind=['air'] * RAYS, sensor_ray_dist=[-1.] * RAYS, sensor_ray_x=[-999] * RAYS,
            sensor_ray_y=[-999] * RAYS, sensor_ray_z=[-999] * RAYS)


class TraverseTests(unittest.TestCase):
    def test_cells_in_order_along_an_axis(self):
        self.assertEqual([c for c, _ in traverse((0.5, 65.5, 0.5), (0., 0., 1.), 2.4)], [(0, 65, 0), (0, 65, 1), (0, 65, 2)])

    def test_negative_and_diagonal_directions(self):
        self.assertEqual([c for c, _ in traverse((0.5, 65.5, 0.5), (0., -1., 0.), 1.2)], [(0, 65, 0), (0, 64, 0)])
        cells = [c for c, _ in traverse((0.5, 65.5, 0.5), (0.7071, 0., 0.7071), 1.5)]
        self.assertEqual(cells[0], (0, 65, 0)); self.assertIn((1, 65, 1), cells)


class SeenMapTests(unittest.TestCase):
    def test_unseen_cells_are_unknown(self):
        self.assertEqual(SeenMap().kind((10, 10, 10)), 'unknown')

    def test_ray_marks_passed_cells_air_and_the_hit_cell(self):
        seen = SeenMap(); seen.update(frame())       # every ray: from eye (0.5, 65.62, 0.5) north-south +z, stone at 2.5
        self.assertEqual([seen.kind((0, 65, z)) for z in range(4)], ['air', 'air', 'air', 'stone'])
        self.assertEqual(seen.kind((0, 65, 4)), 'unknown')                         # behind the hit: not seen
        self.assertEqual(seen.evidence((0, 65, 3)), dict(kind='stone', tick=500, source='ray'))

    def test_missed_rays_mark_air_to_the_range(self):
        seen = SeenMap(); seen.update(frame(**MISS))
        self.assertEqual(seen.kind((0, 65, 8)), 'air')
        self.assertEqual(seen.kind((0, 65, 9)), 'unknown')

    def test_fluid_nearer_than_the_solid_hit(self):
        seen = SeenMap(); seen.update(frame(sensor_ray_fluid=['lava'] * RAYS, sensor_ray_fluid_dist=[1.6] * RAYS))
        self.assertEqual([seen.kind((0, 65, z)) for z in range(4)], ['air', 'air', 'lava', 'stone'])

    def test_voxels_mark_the_cells_around_the_feet(self):
        kinds = ['stone'] * 9 + ['air'] * 18; fluids = ['none'] * 27; kinds[13] = 'other'; fluids[13] = 'water'
        seen = SeenMap(); seen.update(frame(sensor_voxel_kind=kinds, sensor_voxel_fluid=fluids))
        self.assertEqual(seen.kind((0, 63, 0)), 'stone')
        self.assertEqual(seen.kind((1, 64, 1)), 'air')
        self.assertEqual(seen.kind((0, 64, 0)), 'water')
        self.assertEqual(seen.evidence((0, 63, 0))['source'], 'voxel')

    def test_new_measurements_overwrite_old_ones(self):
        seen = SeenMap(); seen.update(frame())
        seen.update(frame(tick=501, sensor_ray_kind=['air'] * RAYS, sensor_ray_dist=[3.5] * RAYS, sensor_ray_z=[4] * RAYS))
        self.assertEqual(seen.kind((0, 65, 3)), 'air')                              # the stone was mined
        self.assertEqual(seen.evidence((0, 65, 3))['tick'], 501)

    def test_cast_stops_at_solid_air_or_unknown(self):
        seen = SeenMap(); seen.update(frame())
        self.assertEqual(seen.cast((0.5, 65.62, 0.5), (0., 0., 1.), 4.5), ('stone', (0, 65, 3)))
        self.assertEqual(seen.cast((0.5, 65.62, 0.5), (0., 0., 1.), 2.), ('air', None))
        self.assertEqual(seen.cast((0.5, 65.62, 0.5), (1., 0., 0.), 4.5)[0], 'unknown')

    def test_depth_below(self):
        seen = SeenMap()
        for y in range(60, 65): seen.mark((3, y, 0), 'air', 1, 'ray')
        seen.mark((3, 59, 0), 'stone', 1, 'ray'); seen.mark((4, 63, 0), 'water', 1, 'ray')
        for y in range(55, 64): seen.mark((5, y, 0), 'air', 1, 'ray')
        self.assertEqual(seen.depth_below((3, 64, 0)), (4, 'stone'))    # feet go from y 64 to y 60, on the stone at 59
        self.assertEqual(seen.depth_below((4, 64, 0)), (0, 'water'))
        self.assertIsNone(seen.depth_below((6, 64, 0)))                 # floor never seen
        self.assertIsNone(seen.depth_below((5, 64, 0), limit=8))        # deeper than the sensor range: unknown


if __name__ == '__main__':
    unittest.main()
