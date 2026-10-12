import math
import unittest
from minecraftrl.waypoints import goal_features,arrived,label_waypoints,RouteProgress


def pose(x=0,z=0,yaw=0):
    return dict(server_x=x,server_y=201.,server_z=z,yaw=yaw,pitch=0.,server_available=1,error=0,screen=0)


class WaypointTests(unittest.TestCase):
    def test_route_arrival_stop_and_height_boundaries(self):
        route=RouteProgress([[0,201,0],[0,201,2]])
        self.assertEqual(route.observe(pose(),sequence=0),[0,201,2])
        self.assertEqual(route.receipts[0]['actions'],0)
        self.assertEqual(route.observe(pose(z=2),sequence=5,stopped=True),None)
        self.assertFalse(route.completed)
        self.assertEqual(route.index,1)
        self.assertEqual(route.observe(pose(z=2),sequence=5),None)
        self.assertTrue(route.completed)
        self.assertEqual(route.receipts[1]['actions'],5)

    def test_relative_bearing_wrap_and_distance(self):
        self.assertEqual(goal_features(pose(),[0,201,8]),[0.,1.,.5,0.,0.,1.])
        self.assertAlmostEqual(goal_features(pose(),[8,201,0])[0],-1.)
        a=goal_features(pose(yaw=179),[0,201,-8]);b=goal_features(pose(yaw=-181),[0,201,-8])
        for x,y in zip(a,b):self.assertAlmostEqual(x,y)
        self.assertAlmostEqual(goal_features(pose(x=-3,z=-9),[-3,201,-1])[2],.5)

    def test_arrival_height_and_invalid_observation(self):
        self.assertTrue(arrived(pose(),[.2,201,0]))
        self.assertFalse(arrived(pose(),[.2,204,0]))
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):goal_features(pose(x=value),[1,201,0])

    def test_teacher_alignment_skips_prefix_and_arrival_without_skipping_turn(self):
        session=dict(handoffs=[dict(step=1,waypoints=[[.5,0],[.5,2]])])
        world=dict(chunk=[0,0],curriculum=dict(mirror_z=False))
        rows=[dict(owner='student',loss_eligible=False,server_before=pose()),
              dict(owner='teacher',loss_eligible=True,phase='move',server_before=pose()),
              dict(owner='teacher',loss_eligible=True,phase='move',server_before=pose(x=.49)),
              dict(owner='teacher',loss_eligible=True,phase='move',server_before=pose(x=.49,z=.1)),
              dict(owner='teacher',loss_eligible=True,phase='move',server_before=pose(x=.5,z=2)),
              dict(owner='teacher',loss_eligible=True,phase='harvest',server_before=pose(x=.5,z=2))]
        labels=label_waypoints(session,world,rows)
        self.assertEqual(set(labels),{1,3})
        self.assertEqual(labels[1]['target'],[.5,201.,0.])
        self.assertEqual(labels[3]['target'],[.5,201.,2.])
        self.assertEqual(labels[3]['waypoint_index'],1)

    def test_mirror_world_target_and_incomplete_route(self):
        session=dict(handoffs=[dict(step=0,waypoints=[[5.5,3.5]])])
        world=dict(chunk=[-1,-2],curriculum=dict(mirror_z=True))
        rows=[dict(owner='teacher',loss_eligible=True,phase='move',server_before=pose(x=-10.5,z=-19.6))]
        self.assertEqual(label_waypoints(session,world,rows),{})
        rows[0]['server_before']=pose(x=-12,z=-20)
        with self.assertRaises(ValueError):label_waypoints(session,world,rows)


if __name__=='__main__':unittest.main()
