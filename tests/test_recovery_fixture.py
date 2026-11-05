import unittest
from minecraftrl.recovery_fixture import geometry


class RecoveryFixtureTests(unittest.TestCase):
    def layout(self,**changes):
        return dict(item='dirt',distance=3.,bearing=0.,view_offset=0.,pitch=25.,obstacle='none',**changes)

    def test_start_separated_and_no_harvestable_target_in_arena(self):
        blocks,player,item,yaw=geometry(self.layout())
        self.assertEqual(set(blocks.values()),{'minecraft:bedrock'})
        self.assertGreater(sum((a-b)**2 for a,b in zip(player,item)),2.4**2)
        self.assertEqual(yaw,0)
        self.assertNotIn((7,201,7),blocks)

    def test_view_offset_changes_view_not_world_or_item(self):
        a=self.layout();b=dict(a,view_offset=150.)
        ga,gb=geometry(a),geometry(b)
        self.assertEqual(ga[:3],gb[:3]);self.assertNotEqual(ga[3],gb[3])

    def test_step_and_occluder_do_not_occupy_player(self):
        for obstacle in ('step','occluder'):
            layout=dict(self.layout(),obstacle=obstacle)
            blocks,player,item,_=geometry(layout)
            self.assertNotIn((7,201,7),blocks)
            if obstacle=='step':self.assertGreater(item[1],202.)
            else:self.assertIn((7,202,9),blocks)

    def test_reject_invalid_geometry(self):
        for key,value in (('distance',0),('distance',float('nan')),('pitch',True),('item','diamond'),('obstacle','script')):
            with self.assertRaises(ValueError):geometry(dict(self.layout(),**{key:value}))


if __name__=='__main__':unittest.main()
