import copy
import unittest

from test_navigation_goal_queries import fixture
from minecraftrl.navigation_goal_queries import goal_variation_requests,flat_body_clearance


class ApproachQueryTests(unittest.TestCase):
    def test_adds_one_intermediate_goal_to_every_original_dense_state(self):
        data,observed=fixture();data['world']['sha256']='b'*64
        data['rows']=[copy.deepcopy(data['rows'][0]) for _ in range(60)]
        observed=[copy.deepcopy(observed[0]) for _ in range(60)]
        before=copy.deepcopy((data,observed))
        dense,_=goal_variation_requests(data,observed,'dense')
        boundary,receipt=goal_variation_requests(data,observed,'boundary')
        self.assertEqual(boundary[:len(dense)],dense)
        extra=boundary[len(dense):];self.assertEqual(len(extra),60)
        self.assertEqual({r['row'] for r in extra},set(range(60)))
        self.assertEqual({r['distance'] for r in extra},{.26,.34,.46})
        for r in extra:
            self.assertTrue(flat_body_clearance(data['world'],r['target']))
            self.assertEqual(r['changes']['forward'],1)
            self.assertEqual(r['motion_goal'][6:9],data['rows'][r['row']]['motion_goal'][6:9])
        self.assertEqual(receipt['coverage'],'all_eligible_logged_states_boundary_v1')
        self.assertEqual(goal_variation_requests(data,observed,'boundary'),(boundary,receipt))
        self.assertEqual((data,observed),before)

    def test_uses_admissible_ray_at_a_corner_and_rejects_development(self):
        from minecraftrl.waypoints import goal_features
        data,observed=fixture();data['world']['sha256']='c'*64
        g=observed[0]['server_before'];g.update(server_x=6.69,server_z=14.69,yaw=-45.)
        goal=goal_features(g,observed[0]['target'])+[0.,0.,0.,1.]
        data['rows'][0]['motion_goal']=goal;observed[0]['goal']=list(goal)
        dense,_=goal_variation_requests(data,observed,'dense')
        boundary,_=goal_variation_requests(data,observed,'boundary')
        extra=boundary[len(dense):];self.assertEqual(len(extra),1)
        self.assertTrue(flat_body_clearance(data['world'],extra[0]['target']))
        self.assertEqual(extra[0]['bearing'],next(r for r in dense if r['distance']<.22)['bearing'])
        data['split']='development'
        with self.assertRaises(ValueError):goal_variation_requests(data,observed,'boundary')


if __name__=='__main__':unittest.main()
