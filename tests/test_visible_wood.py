import math
import unittest
from minecraftrl.visible_wood import validate_acquisition


class VisibleWoodTests(unittest.TestCase):
    def observation(self):
        e=dict(source='privileged_camera_surface_rays_v1',tick=10,yaw_offset=30,pitch_offset=0,
            yaw=0,pitch=0,range=16,eye_x=.5,eye_y=1.62,eye_z=.5,
            hit_x=-1,hit_y=1.62,hit_z=.5+3*math.sqrt(3)/2,block_x=-2,block_y=1,block_z=3)
        return dict(server_tick=10,wood_camera_rays_enabled=1,wood_v3_kind=1,
            wood_v3_target_x=-2,wood_v3_target_y=1,wood_v3_target_z=3,wood_camera_acquisition=e)

    def test_peripheral_log_receipt_is_inside_fov_and_binds_target(self):
        obs=self.observation();self.assertEqual(validate_acquisition(obs),obs['wood_camera_acquisition'])

    def test_forged_stale_outside_or_direction_mismatched_receipts_rejected(self):
        for key,value in [('tick',9),('yaw_offset',45),('pitch_offset',30),('range',32),
                          ('hit_x',1),('hit_y',float('nan')),('yaw',180),('block_x',3)]:
            obs=self.observation();obs['wood_camera_acquisition'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_acquisition(obs)
        for key,value in [('wood_camera_rays_enabled',0),('wood_v3_kind',0)]:
            obs=self.observation();obs[key]=value
            with self.assertRaises(ValueError):validate_acquisition(obs)

    def test_absence_is_not_discovery_evidence(self):
        self.assertIsNone(validate_acquisition({}))


if __name__=='__main__':unittest.main()
