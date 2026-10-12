import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('nbtlib'),'isolated fixture-tools environment required')
class FixtureTests(unittest.TestCase):
    def test_explicit_start_yaw_faces_boundary_and_mirrors(self):
        f=self.module()
        for mirror,expected in ((False,180.),(True,0.)):
            _,yaw=f.start_pose(f.arena('recovery',wall_end=8,mirror_z=mirror),
                               start_xz=(6.3,1.3),mirror_z=mirror,start_yaw=180.)
            self.assertEqual(yaw,expected)
        for value in (float('nan'),float('inf'),True,181):
            with self.assertRaises(ValueError):f.start_pose(f.arena('recovery'),start_yaw=value)

    def test_gap_start_pose_is_clear_and_mirrors_continuous_position(self):
        f=self.module()
        normal=f.start_pose(f.arena('recovery',wall_end=3),start_xz=(5.5,5.5))
        mirror=f.start_pose(f.arena('recovery',wall_end=3,mirror_z=True),start_xz=(5.5,5.5),mirror_z=True)
        self.assertEqual(normal[0],[5.5,201.,5.5])
        self.assertEqual(mirror[0],[5.5,201.,10.5])
        self.assertAlmostEqual(normal[1]+mirror[1],-180.)
        for pos in ((7.5,3.5),(.5,5.5),(float('nan'),5.),(True,4.)):
            with self.assertRaises(ValueError):f.start_pose(f.arena('recovery',wall_end=3),start_xz=pos)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('minecraftrl.rl_fixture'))
        from minecraftrl import rl_fixture
        return rl_fixture

    def test_section_palette_crosses_bit_width_without_corruption(self):
        f=self.module()
        from minecraftrl.saved_block import section_block
        blocks=['minecraft:air']*4096
        for i in range(25):blocks[i]='minecraft:block_'+str(i)
        section=f.section(12,blocks)
        for i in range(25):self.assertEqual(section_block(section,i%16,192,i//16),blocks[i])
        self.assertEqual(section_block(section,15,207,15),'minecraft:air')

    def test_region_replacement_preserves_other_chunk(self):
        f=self.module()
        region=bytes(8192)
        first=f.replace_chunk(region,0,b'first')
        second=f.replace_chunk(first,1,b'second')
        changed=f.replace_chunk(second,0,b'changed'*4000)
        self.assertEqual(f.chunk_bytes(changed,0),b'changed'*4000)
        self.assertEqual(f.chunk_bytes(changed,1),b'second')

    def test_valid_partial_last_sector_can_be_copied(self):
        f=self.module();region=f.replace_chunk(bytes(8192),0,b'original')
        length=int.from_bytes(region[8192:8196],'big')
        partial=region[:8192+4+length]
        result=f.replace_chunk(partial,1,b'new')
        self.assertEqual(f.chunk_bytes(result,0),b'original')
        self.assertEqual(f.chunk_bytes(result,1),b'new')
        with self.assertRaises(ValueError):f.replace_chunk(partial[:-1],1,b'new')

    def test_arena_has_open_detour_and_different_barrier_materials(self):
        f=self.module()
        a=f.arena('break');b=f.arena('detour')
        self.assertEqual(a[(7,202,3)],'minecraft:dirt')
        self.assertEqual(b[(7,202,3)],'minecraft:stone')
        self.assertEqual(a.get((7,202,14),'minecraft:air'),'minecraft:air')
        self.assertEqual(b.get((7,202,2),'minecraft:air'),'minecraft:air')
        self.assertEqual(a[(11,201,3)],'minecraft:oak_log')
        recovery=f.arena('recovery')
        self.assertEqual(recovery[(7,202,3)],'minecraft:bedrock')
        self.assertNotIn((7,202,14),recovery)
        with self.assertRaises(ValueError):f.arena('unknown')

    def test_curriculum_changes_gap_and_mirrors_entire_geometry(self):
        f=self.module();normal=f.arena('recovery',wall_end=3)
        mirrored=f.arena('recovery',wall_end=3,mirror_z=True)
        self.assertEqual(mirrored,{(x,y,15-z):v for (x,y,z),v in normal.items()})
        self.assertEqual(normal[(7,202,3)],'minecraft:bedrock')
        self.assertNotIn((7,202,4),normal)
        self.assertEqual(mirrored[(11,201,12)],'minecraft:oak_log')
        self.assertNotIn((7,202,11),mirrored)

    def test_curriculum_rejects_invalid_or_non_recovery_options(self):
        f=self.module()
        for value in (2,14,3.5,True):
            with self.assertRaises(ValueError):f.arena('recovery',wall_end=value)
        with self.assertRaises(ValueError):f.arena('break',wall_end=3)
        with self.assertRaises(ValueError):f.arena('death',mirror_z=True)

    def test_curriculum_all_gaps_have_a_two_block_high_walking_path(self):
        f=self.module()
        for end in (3,5,7):
            for mirror in (False,True):
                blocks=f.arena('recovery',wall_end=end,mirror_z=mirror)
                start=(3,12 if mirror else 3);goal=(10,start[1]);seen={start};queue=[start]
                for x,z in queue:
                    for p in ((x-1,z),(x+1,z),(x,z-1),(x,z+1)):
                        if p in seen or not all(0<=v<16 for v in p):continue
                        if any((p[0],y,p[1]) in blocks for y in (201,202)):continue
                        seen.add(p);queue.append(p)
                self.assertIn(goal,seen)
