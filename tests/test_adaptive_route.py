import unittest
from minecraftrl.adaptive_route import AdaptiveRoute
from test_exploration_route import pose


class AdaptiveRouteTests(unittest.TestCase):
    def blocked(self):
        r=AdaptiveRoute(pose(),[])
        for i in range(12):r.observe_action(pose(),pose(),{'forward':[1],'camera':[[0,0]]},sequence=i+1)
        return r

    def test_blocked_direction_generalizes_to_nearby_origin(self):
        r=self.blocked()
        self.assertTrue(r.blocked(pose(.6,.2),[3.6,201,.2]))
        self.assertFalse(r.blocked(pose(.6,.2),[-2.4,201,.2]))
        self.assertFalse(r.blocked(pose(10,10),[13,201,10]))

    def test_turning_and_actual_movement_do_not_claim_blockage(self):
        r=AdaptiveRoute(pose(),[])
        for i in range(15):r.observe_action(pose(i*.2,0),pose((i+1)*.2,0),{'forward':[1]},sequence=i+1)
        self.assertEqual(r.blocked_directions,[])
        for i in range(15):r.observe_action(pose(yaw=i*30),pose(yaw=i*30),{'forward':[1]},sequence=i+20)
        self.assertEqual(r.blocked_directions,[])

    def test_actual_crossing_refutes_prior_local_failure(self):
        r=self.blocked();r.observe_action(pose(),pose(.8,0),{'forward':[1]},sequence=13)
        self.assertFalse(r.blocked(pose(.5,0),[3.5,201,0]))
        self.assertEqual(r.blocked_directions[0]['status'],'refuted_by_crossing')

    def test_failed_direction_is_not_admissible_in_next_candidates(self):
        r=self.blocked();r.replan(pose(.5,0),sequence=12,reason='test_failure')
        for c in r.decisions[-1]['candidates']:
            if c['target'][0]>2:self.assertFalse(c['admissible'])

    def test_external_choice_cannot_select_blocked_or_change_started_goal(self):
        r=self.blocked();r.replan(pose(),sequence=12,reason='test_failure')
        blocked=next(i for i,c in enumerate(r.decisions[-1]['candidates']) if not c['admissible'])
        with self.assertRaises(ValueError):r.choose(blocked,sequence=12,source='test')
        legal=next(i for i,c in enumerate(r.decisions[-1]['candidates']) if c['admissible'])
        r.choose(legal,sequence=12,source='test')
        with self.assertRaises(ValueError):r.choose(legal,sequence=13,source='stale')


if __name__=='__main__':unittest.main()
