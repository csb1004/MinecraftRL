import importlib.util
import unittest
import math

class ProbeTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('minecraftrl.fixture_probe'))
        from minecraftrl import fixture_probe
        return fixture_probe
    def test_navigation_turns_before_forward_and_wraps_yaw(self):
        f=self.module();g=dict(server_x=0,server_z=0,yaw=90,pitch=0)
        action,arrived=f.navigate(g,2,0)
        self.assertFalse(arrived);self.assertFalse(action.get('forward',0))
        g['yaw']=-90;action,arrived=f.navigate(g,2,0)
        self.assertEqual(action['forward'],1)
        g['server_x']=1.9;self.assertTrue(f.navigate(g,2,0)[1])
    def test_normal_world_cannot_silently_use_scripted_probe(self):
        f=self.module()
        with self.assertRaises(ValueError):f.ProbePolicy(None,{},'break')
        with self.assertRaises(ValueError):f.ProbePolicy(None,dict(kind='death',chunk=[0,0]),'break')

    def test_calibration_moves_to_pickup_after_confirmed_target_disappears(self):
        from types import SimpleNamespace
        f=self.module();raw=SimpleNamespace(action_space=SimpleNamespace(noop=lambda:{}))
        p=f.ProbePolicy(raw,dict(kind='break',chunk=[0,0]),'break');p.reset()
        p.phase='harvest';p.waypoints=[];p.harvest_target_seen=True
        p.environment=SimpleNamespace(gui=dict(server_x=10.7,server_z=3.5,yaw=-90.,pitch=8.5,
            eye_x=10.7,eye_y=202.62,eye_z=3.5,hit_x=15,hit_y=201,hit_z=3))
        action=p.act(None)
        self.assertEqual(p.phase,'clear_stump');self.assertFalse(action.get('jump'))
        g=p.environment.gui
        g.update(pitch=math.degrees(math.atan2(1.12,.8)),hit_x=11,hit_y=201,hit_z=3)
        action=p.act(None);self.assertEqual(action.get('attack'),1)
        g['hit_x']=15
        action=p.act(None)
        self.assertEqual(p.phase,'collect');self.assertEqual(action.get('forward'),1)
        self.assertFalse(action.get('jump'))
