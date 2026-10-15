import unittest
from unittest.mock import patch
import test_rm_navigation
import test_planning_recovery
from minecraftrl.motion_blockage import MotionBlockage
from test_exploration_route import pose
from minecraftrl.adaptive_route import AdaptiveRoute


class MotionBlockageTests(unittest.TestCase):
    def test_repeated_bumps_survive_short_turns_and_jumps(self):
        detector=MotionBlockage();found=None
        for i in range(24):
            a=pose();b=pose();a['server_y']=201+(i%4)*.3;b['server_y']=201+((i+1)%4)*.3
            result=detector.observe(a,b,{'forward':[1]} if i%2==0 else {},sequence=i+1)
            if result:found=result
        self.assertIsNotNone(found);self.assertEqual(found['support_actions'],12)
        self.assertLessEqual(found['window_actions'],24)
        self.assertFalse(found['terrain_proven'])

    def test_real_motion_and_alternating_directions_do_not_prove_blockage(self):
        detector=MotionBlockage()
        for i in range(40):
            self.assertIsNone(detector.observe(pose(i*.1,0),pose((i+1)*.1,0),{'forward':[1]},sequence=i+1))
        detector=MotionBlockage()
        for i in range(24):
            action={'forward':[1]} if i%3==0 else {'back':[1]} if i%3==1 else {}
            self.assertIsNone(detector.observe(pose(),pose(),action,sequence=i+1))

    def test_old_support_expires_and_changed_floor_cannot_share_evidence(self):
        detector=MotionBlockage()
        for i in range(11):self.assertIsNone(detector.observe(pose(),pose(),{'forward':[1]},sequence=i+1))
        for i in range(24):self.assertIsNone(detector.observe(pose(),pose(),{},sequence=i+12))
        self.assertIsNone(detector.observe(pose(),pose(),{'forward':[1]},sequence=36))
        raised=pose();raised['server_y']=204
        self.assertIsNone(detector.observe(raised,raised,{'forward':[1]},sequence=37))

    def test_route_remembers_windowed_blockage_and_crossing_can_refute_it(self):
        route=AdaptiveRoute(pose(),[],windowed_motion=True)
        for i in range(24):route.observe_action(pose(),pose(),{'forward':[1]} if i%2==0 else {},sequence=i+1)
        self.assertTrue(route.blocked(pose(),[3.,201.,0.]))
        route.observe_action(pose(),pose(.8,0),{'forward':[1]},sequence=25)
        self.assertFalse(route.blocked(pose(.8,0),[3.,201.,0.]))

    def test_early_failure_requires_current_target_aligned_evidence(self):
        from minecraftrl.waypoint_skill import capability,assess_navigation
        cap=capability(0,'fixture','a'*64,'b'*64,[3.,201.,0.]);rows=[]
        for i in range(24):
            a=pose();b=pose();a.update(server_tick=i,wood_v3_reachable=0,server_inventory={})
            b.update(server_tick=i+1,wood_v3_reachable=0,server_inventory={})
            rows.append(dict(server_before=a,server_after=b,action={'forward':[1]} if i%2 else {}))
        result=assess_navigation(cap,rows,reason='observed_blockage')
        self.assertEqual(result['outcome'],'failed');self.assertEqual(result['diagnosis'],'blocked_direction_suspected')
        bad=capability(0,'fixture','a'*64,'b'*64,[-3.,201.,0.])
        self.assertEqual(assess_navigation(bad,rows,reason='observed_blockage')['outcome'],'unverified')

    def test_crossing_on_another_floor_cannot_refute_local_blockage(self):
        route=AdaptiveRoute(pose(),[],windowed_motion=True)
        for i in range(12):route.observe_action(pose(),pose(),{'forward':[1]},sequence=i+1)
        a=pose();b=pose(.8,0);a['server_y']=b['server_y']=205
        route.observe_action(a,b,{'forward':[1]},sequence=13)
        self.assertEqual(route.blocked_directions[0]['status'],'suspected')
        self.assertTrue(route.blocked(pose(),[3.,201.,0.]))
        self.assertFalse(route.blocked(a,[3.,205.,0.]))

    def test_recovered_movement_cannot_create_fresh_blockage_evidence(self):
        detector=MotionBlockage()
        for i in range(12):detector.observe(pose(),pose(),{'forward':[1]},sequence=i+1)
        self.assertIsNone(detector.observe(pose(),pose(.08,0),{'forward':[1]},sequence=13))


class MotionBound(test_rm_navigation.Bound):
    def __init__(self,*args,**kwargs):super().__init__(*args,navigation_strategy='coverage_motion',**kwargs)
    def start_node(self,node,reason):
        super().start_node(node,reason)
        self.student.motion_goal=(self.navigation_cap['arguments']['target'],self.environment.gui) if self.phase=='waypoint' else None


class MotionIntegrationTests(unittest.TestCase):
    run_case=test_planning_recovery.PlanningRecoveryTests.run_case
    def case(self,**kwargs):
        original=test_planning_recovery.Actor.act
        def sample(actor,screen):
            action=original(actor,screen)
            if getattr(actor,'motion_goal',None):
                from minecraftrl.waypoints import goal_features
                target,g=actor.motion_goal;f=goal_features(g,target)
                key=('right' if f[0]>0 else 'left') if abs(f[0])>abs(f[1]) else ('forward' if f[1]>0 else 'back')
                action[key]=1
            return action
        with patch('minecraftrl.planning_recovery.PlanningRecoveryController',MotionBound),patch.object(test_planning_recovery.Actor,'act',sample):
            return self.run_case(provider_factory=lambda *a:test_rm_navigation.FixedNavigationProvider(.001,max_calls=2),
                continue_after_return=True,**kwargs)

    def test_verified_blockage_ends_option_before_budget_and_enters_next_plan(self):
        p,a,g,r,events,provider=self.case()
        self.assertNotIn('error',r);self.assertEqual(provider.calls,2)
        receipts=p.navigation_receipts();self.assertTrue(receipts)
        self.assertEqual(receipts[0]['reason'],'observed_blockage')
        self.assertLess(receipts[0]['committed_steps'],100)
        self.assertTrue(provider.packets[1]['waypoint_context']['motion_evidence'])
        self.assertFalse(p.scheduler.state['root']['completed'])

    def test_stop_before_first_dispatch_cannot_create_a_blockage_failure(self):
        p,a,g,r,events,provider=self.case(stop_recovery_inference=True)
        self.assertNotIn('error',r);self.assertEqual(r['status'],'cancelled')
        self.assertTrue(all(x['outcome']!='failed' for x in p.navigation_receipts()))
        self.assertEqual(g.state['held_key_count'],0)


if __name__=='__main__':unittest.main()
