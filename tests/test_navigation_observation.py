import copy
import unittest
from minecraftrl.navigation_observation import position,client_tick_delta
from minecraftrl.navigation_refinement import motion_goal,training_pose_source
from minecraftrl.navigation_arrival import ArrivalWindow
from minecraftrl.fixture_probe import navigate


def pose(server_z=0.,client_z=0.,tick=1):
    return dict(server_x=0.,server_y=201.,server_z=server_z,server_available=1,
        client_x=0.,client_y=201.,client_z=client_z,client_available=1,
        server_tick=tick,client_tick=tick+50,server_on_ground=1,client_on_ground=1,
        yaw=0.,pitch=0.,error=0,screen=0)


class NavigationObservationTests(unittest.TestCase):
    def test_current_view_pose_drives_goal_without_mutating_server_evidence(self):
        g=pose(server_z=0.,client_z=1.);original=copy.deepcopy(g);target=[0.,201.,2.]
        self.assertEqual(motion_goal(g,target,pose_source='client')[2],1/16)
        self.assertEqual(motion_goal(g,target)[2],2/16)
        self.assertEqual(g,original)

    def test_motion_uses_current_client_displacement_and_both_tick_counters(self):
        before=pose(client_z=.8);after=pose(client_z=1.,tick=2)
        goal=motion_goal(after,[0.,201.,2.],before,pose_source='client')
        self.assertAlmostEqual(goal[7],.2)
        self.assertEqual(motion_goal(after,[0.,201.,2.],before)[7],0.)
        after['client_tick']+=1
        with self.assertRaises(ValueError):motion_goal(after,[0.,201.,2.],before,pose_source='client')

    def test_teacher_releases_at_current_position_before_lagging_server_arrives(self):
        g=pose(server_z=0.,client_z=.95)
        self.assertEqual(navigate(g,0.,1.,pose_source='client'),({},True))
        self.assertEqual(navigate(g,0.,1.)[0]['forward'],1)

    def test_client_unavailable_cannot_silently_fall_back_to_server(self):
        for field,value in (('client_available',0),('client_tick',-1),('client_tick',float('nan')),
                            ('client_x',float('inf'))):
            g=pose();g[field]=value
            with self.assertRaises(ValueError):position(g,'client')
        g=pose();del g['client_x']
        with self.assertRaises(KeyError):position(g,'client')

    def test_settled_client_alone_is_not_authoritative_success(self):
        window=ArrivalWindow(3,pose_source='client')
        for tick in range(1,8):
            self.assertFalse(window.advance(pose(server_z=1.,tick=tick),pose(server_z=1.,tick=tick+1),[0.,201.,0.]))
        self.assertEqual(window.count,0)

    def test_client_departure_cannot_hide_behind_stale_stationary_server(self):
        window=ArrivalWindow(3,pose_source='client')
        window.advance(pose(),pose(tick=2),[0.,201.,0.])
        self.assertFalse(window.advance(pose(tick=2),pose(client_z=.2,tick=3),[0.,201.,0.]))
        self.assertEqual(window.count,0)
        for tick in (3,4):self.assertFalse(window.advance(pose(tick=tick),pose(tick=tick+1),[0.,201.,0.]))
        self.assertTrue(window.advance(pose(tick=5),pose(tick=6),[0.,201.,0.]))

    def test_training_cannot_mix_delayed_and_current_control_inputs(self):
        self.assertEqual(training_pose_source([dict(config={})]),'server')
        self.assertEqual(training_pose_source([dict(config={'pose_source':'client'})]),'client')
        for data in ([],[dict(config={}),dict(config={'pose_source':'client'})],
                     [dict(config={'pose_source':'unknown'})]):
            with self.assertRaises(ValueError):training_pose_source(data)

