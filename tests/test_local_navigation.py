import copy
import unittest

from minecraftrl.local_navigation import LocalNavigation,checked_grid,clear_segment,validate_candidate,SOURCE


def observation(x=0.5,z=0.5,tick=1,target=None):
    # Two rooms divided by x=2, with an opening only at z=-3.
    cx,cz=int(x//1),int(z//1)
    grid=dict(x=cx,y=10,z=cz,radius=6,tick=tick,source=SOURCE,walkable=[],wood_access=[])
    for dz in range(-6,7):
        for dx in range(-6,7):
            a,b=cx+dx,cz+dz
            free=-3<=a<=5 and -4<=b<=4 and not (a==2 and b>=-2)
            grid['walkable'].append(int(free))
            grid['wood_access'].append(int(free and target is not None and a==4 and b==0))
    return dict(server_x=x,server_y=10.,server_z=z,server_tick=tick,server_available=1,error=0,screen=0,
        wood_v3_kind=int(target is not None),wood_v3_target_x=5,wood_v3_target_y=11,wood_v3_target_z=0,
        local_navigation=grid)


class LocalNavigationTests(unittest.TestCase):
    def test_wall_requires_detour_before_approaching_known_tree(self):
        obs=observation(target=True);planner=LocalNavigation();choice=planner.waypoint(obs)
        self.assertEqual(choice['purpose'],'approach')
        self.assertLess(choice['target'][2],0)
        self.assertIn([2,-3],choice['path'])
        self.assertTrue(validate_candidate(obs,choice['target']))
        self.assertFalse(validate_candidate(obs,[3.5,10.,.5]))

    def test_unknown_and_outer_wall_are_never_candidate_footprints(self):
        obs=observation();planner=LocalNavigation()
        choices=planner.choices(obs)
        self.assertTrue(choices)
        for choice in choices:self.assertTrue(validate_candidate(obs,choice['target']))
        self.assertFalse(validate_candidate(obs,[-3.9,10.,.5]))

    def test_swept_body_cannot_cut_corner(self):
        known={(0,0):True,(1,0):True,(1,1):True}
        self.assertFalse(clear_segment(known,[.5,10.,.5],[1.5,10.,1.5]))
        self.assertTrue(clear_segment(known,[.5,10.,.5],[1.5,10.,.5]))

    def test_missing_stale_and_pose_mismatched_sensor_rejected(self):
        obs=observation()
        for mutation in ('missing','stale','pose','access'):
            bad=copy.deepcopy(obs)
            if mutation=='missing':del bad['local_navigation']
            elif mutation=='stale':bad['local_navigation']['tick']=0
            elif mutation=='pose':bad['local_navigation']['x']=1
            else:bad['local_navigation']['wood_access'][0]=1
            with self.assertRaises(ValueError):checked_grid(bad)

    def test_observed_wall_change_invalidates_cached_route(self):
        planner=LocalNavigation();obs=observation(target=True)
        self.assertEqual(planner.waypoint(obs)['purpose'],'approach')
        new=observation(tick=2,target=True)
        for z in range(-6,7):
            if z<=-3:new['local_navigation']['walkable'][(z+6)*13+2+6]=0
        choices=planner.choices(new)
        self.assertTrue(all(c['purpose']=='explore' for c in choices))

    def test_jump_near_next_integer_height_does_not_erase_ground_map(self):
        planner=LocalNavigation();obs=observation(target=True);planner.observe(obs)
        before=dict(planner.known)
        jump=observation(tick=2,target=True);jump['server_y']=11.02
        jump['local_navigation'].update(y=11,walkable=[0]*169,wood_access=[0]*169)
        self.assertFalse(planner.observe(jump));self.assertEqual(planner.floor,10)
        self.assertEqual(planner.known,before)

    def test_fresh_support_sensor_plans_while_airborne(self):
        obs=observation(target=True);obs['server_y']=11.25
        planner=LocalNavigation();choice=planner.waypoint(obs)
        self.assertIsNotNone(choice);self.assertEqual(choice['target'][1],10)
        self.assertTrue(validate_candidate(obs,choice['target']))
        obs['server_y']=13.
        with self.assertRaises(ValueError):checked_grid(obs)

    def test_route_walks_around_wall_without_cutting_it(self):
        planner=LocalNavigation();obs=observation(target=True);trace=[]
        for tick in range(1,20):
            choice=planner.waypoint(obs)
            self.assertIsNotNone(choice)
            self.assertTrue(validate_candidate(obs,choice['target']))
            x,_,z=choice['target'];trace.append((x,z))
            if (x,z)==(4.5,.5):break
            obs=observation(x,z,tick+1,target=True)
        self.assertEqual(trace[-1],(4.5,.5))
        self.assertTrue(any(z<=-2.5 for x,z in trace))


if __name__=='__main__':unittest.main()
