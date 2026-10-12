import tempfile
import unittest
from pathlib import Path

from minecraftrl.spatial_memory import SpatialMemory
from minecraftrl.f3_coordinates import parse_f3_xyz


class SpatialTests(unittest.TestCase):
    def memory(self):
        return SpatialMemory('world-a','episode-a',source='privileged_server',dimension='overworld')

    def test_entry_counts_and_directed_edges_not_dwell_or_guessed_reverse(self):
        m=self.memory()
        for tick,x in enumerate((.5,.6,2.5,2.6),1):m.observe([x,65.,.5],tick)
        self.assertEqual(m.summary()['unique_cells'],2)
        self.assertEqual(m.summary()['cell_entries'],2)
        self.assertEqual(len(m.edges),1)
        self.assertIsNone(m.route([0,16,0],start=[1,16,0]))

    def test_gap_or_teleport_does_not_invent_an_edge(self):
        m=self.memory();m.observe([.5,65,.5],1);m.observe([2.5,65,.5],8)
        m.observe([30.5,65,.5],9)
        self.assertEqual(len(m.edges),0)
        self.assertEqual(m.summary()['discontinuities'],2)
        with self.assertRaises(ValueError):m.observe([31.,65,.5],9)

    def test_negative_positions_height_and_jump(self):
        m=self.memory();m.observe([-.1,65,.5],1);m.observe([-.1,66.25,.5],2)
        self.assertEqual(m.summary()['unique_cells'],1)
        self.assertEqual(m.current,(-1,16,0))
        m.observe([-.1,70,.5],3)
        self.assertNotEqual(m.current,(-1,16,0))

    def test_landmark_persistence_and_wrong_world_rejection(self):
        m=self.memory();m.observe([.5,65,.5],1);m.set_landmark('home')
        m.observe([2.5,65,.5],2);m.observe([.5,65,.5],3)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'map.json';m.save(path)
            restored=SpatialMemory.load(path,world_id='world-a',source='privileged_server',dimension='overworld')
            self.assertEqual(restored.landmarks,m.landmarks)
            self.assertEqual(restored.summary()['unique_cells'],2)
            restored.observe([2.5,65,.5],1)
            self.assertEqual(restored.route_to_landmark('home')['cells'],[[1,16,0],[0,16,0]])
            with self.assertRaises(ValueError):SpatialMemory.load(path,world_id='different',source='privileged_server',dimension='overworld')
            with self.assertRaises(ValueError):SpatialMemory.load(path,world_id='world-a',source='f3_ocr',dimension='overworld')
            restored.remove_landmark('home');self.assertEqual(restored.landmarks,{})

    def test_return_uses_familiar_short_route_explore_prefers_less_used_route(self):
        m=self.memory()
        # Two actually traversed routes: A-B-C and A-D-E-F-C, no reverse assumptions.
        path=[(.5,.5),(2.5,.5),(4.5,.5),(2.5,.5),(.5,.5)]*6
        path +=[(.5,2.5),(2.5,2.5),(4.5,2.5),(4.5,.5),(2.5,.5),(.5,.5)]
        for tick,(x,z) in enumerate(path,1):m.observe([x,65,z],tick)
        self.assertEqual(len(m.route([2,16,0],mode='return')['cells']),3)
        self.assertEqual(len(m.route([2,16,0],mode='explore')['cells']),5)
        self.assertFalse(m.summary()['waypoint_execution_supported'])

    def test_f3_strict_xyz_not_block_or_ambiguous_or_nan(self):
        self.assertEqual(parse_f3_xyz('XYZ: -12.500 / 65.000 / 2.250\nBlock: -13 65 2'),[-12.5,65.,2.25])
        for text in ('Block: 12 65 2','XYZ: NaN / 65 / 2','XYZ: 1 / 2 / 3\nXYZ: 4 / 5 / 6','XYZ: 999999999 / 2 / 3'):
            with self.subTest(text=text),self.assertRaises(ValueError):parse_f3_xyz(text)


if __name__=='__main__':unittest.main()
