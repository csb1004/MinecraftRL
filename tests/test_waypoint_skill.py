import unittest
from copy import deepcopy
from test_planning_recovery import Game
from minecraftrl.waypoint_skill import capability,validate_capability,assess_navigation,same_failed_route


def cap(index=0,target=None):
    return capability(index,'policy','a'*64,'b'*64,target or [6.7,201.,6.5])


def trace(count=12,*,yaw=0.,dx=0.,dz=0.):
    before=Game().state;before['yaw']=yaw;rows=[]
    for i in range(count):
        after=deepcopy(before);after['server_tick']+=1;after['server_x']+=dx;after['server_z']+=dz
        rows.append(dict(server_before=before,server_after=after,action={'forward':[1],'camera':[0.,0.]}))
        before=after
    return rows


class WaypointSkillTests(unittest.TestCase):
    def test_contract_binds_target_weights_and_fixed_completion(self):
        value=cap();self.assertEqual(validate_capability(value),value)
        for change in ({'max_budget_ticks':101},{'completion':'goal_inventory'}):
            with self.assertRaises(ValueError):validate_capability(dict(value,**change))
        with self.assertRaises(ValueError):cap(target=[float('nan'),201.,1.])

    def test_motion_failure_classes_do_not_invent_terrain(self):
        aligned=assess_navigation(cap(),trace(),reason='budget_exhausted')
        self.assertEqual(aligned['diagnosis'],'blocked_direction_suspected')
        self.assertFalse(aligned['terrain_proven']);self.assertEqual(aligned['evidence']['aligned_low_motion_streak'],12)
        misaligned=assess_navigation(cap(),trace(yaw=90.),reason='budget_exhausted')
        self.assertEqual(misaligned['diagnosis'],'heading_misaligned')
        self.assertEqual(misaligned['evidence']['aligned_low_motion_streak'],0)

    def test_goal_arrival_partial_progress_stop_and_invalid_observation(self):
        reached=assess_navigation(cap(),trace(10,dz=.3),reason='predicate_satisfied')
        self.assertEqual(reached['outcome'],'completed');self.assertFalse(reached['root_completed'])
        near=assess_navigation(cap(),trace(8,dz=.3),reason='budget_exhausted')
        self.assertEqual(near['outcome'],'failed');self.assertEqual(near['diagnosis'],'arrival_precision')
        self.assertTrue(near['partial_progress'])
        for reason in ('stop','control_budget','environment_done'):
            result=assess_navigation(cap(),trace(),reason=reason)
            self.assertEqual(result['outcome'],'interrupted');self.assertFalse(result['retry_blocking'])
        self.assertEqual(assess_navigation(cap(),[],reason='budget_exhausted')['outcome'],'unrun')
        bad=trace();bad[-1]['server_after']['error']=1
        self.assertEqual(assess_navigation(cap(),bad,reason='budget_exhausted')['outcome'],'unverified')

    def test_equivalence_ignores_alias_and_tiny_target_shift_but_allows_new_approach(self):
        rows=trace();report=assess_navigation(cap(),rows,reason='budget_exhausted')
        receipt=dict(method=cap(),assessment=report,before=dict(world_id='w',episode_id='e'),after={})
        pose=rows[-1]['server_after']
        self.assertTrue(same_failed_route(cap(3,[6.8,201.,6.6]),pose,receipt,world_id='w',episode_id='e'))
        self.assertFalse(same_failed_route(cap(),pose,receipt,world_id='other',episode_id='e'))
        self.assertFalse(same_failed_route(cap(1,[6.7,201.,.5]),pose,receipt,world_id='w',episode_id='e'))
        changed=dict(pose,wood_v3_reachable=1)
        self.assertFalse(same_failed_route(cap(),changed,receipt,world_id='w',episode_id='e'))
        receipt['assessment']=assess_navigation(cap(),rows,reason='stop')
        self.assertFalse(same_failed_route(cap(),pose,receipt,world_id='w',episode_id='e'))


if __name__=='__main__':unittest.main()
