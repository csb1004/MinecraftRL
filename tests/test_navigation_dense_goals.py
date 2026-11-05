import copy,math,unittest
from test_navigation_goal_queries import fixture
from minecraftrl import navigation_goal_queries as module
from minecraftrl.waypoints import goal_features


class DenseCoverageTests(unittest.TestCase):
    def data(self):
        data,observed=fixture();data['world']['sha256']='a'*64
        return data,observed

    def test_every_state_gets_near_far_pair_without_mutating_original(self):
        data,observed=self.data();data['rows']=[copy.deepcopy(data['rows'][0]) for _ in range(40)]
        observed=[copy.deepcopy(observed[0]) for _ in range(40)];before=copy.deepcopy((data,observed))
        requests,receipt=module.goal_variation_requests(data,observed,'dense')
        self.assertEqual(receipt['selected_rows'],list(range(40)));self.assertEqual(len(requests),120)
        for index in range(40):
            labels=[r for r in requests if r['row']==index]
            self.assertEqual(len([r for r in labels if r['distance']<.22]),1)
            near=next(r for r in labels if r['distance']<.22)
            far=next(r for r in labels if r['distance']>=.6 and r['bearing']==near['bearing'])
            self.assertEqual(near['changes'],{});self.assertEqual(far['changes']['forward'],1)
            self.assertEqual(near['bearing'],0.)
            self.assertTrue(all(r['motion_goal'][6:9]==data['rows'][index]['motion_goal'][6:9] for r in labels))
        self.assertEqual((data,observed),before)
        self.assertEqual(module.goal_variation_requests(data,observed,'dense'),(requests,receipt))

    def test_corner_fallback_and_body_clearance(self):
        for x,z,yaw in ((6.69,14.69,-45.),(1.31,1.31,135.),(4.,14.69,0.)):
            data,observed=self.data();g=observed[0]['server_before'];g.update(server_x=x,server_z=z,yaw=yaw)
            goal=goal_features(g,observed[0]['target'])+[0.,0.,0.,1.]
            data['rows'][0]['motion_goal']=goal;observed[0]['goal']=list(goal)
            requests,receipt=module.goal_variation_requests(data,observed,'dense');self.assertTrue(requests)
            for r in requests:self.assertTrue(module.flat_body_clearance(data['world'],r['target']))
            near=next(r for r in requests if r['distance']<.22)
            self.assertTrue(any(r['bearing']==near['bearing'] and r['distance']>=.6 for r in requests))

    def test_legacy_output_unchanged_and_invalid_data_rejected(self):
        data,observed=self.data();self.assertEqual(module.goal_variation_requests(data,observed),module.goal_variation_requests(data,observed,'sparse'))
        with self.assertRaises(ValueError):module.goal_variation_requests(data,observed,'unknown')
        data['split']='development'
        with self.assertRaises(ValueError):module.goal_variation_requests(data,observed,'dense')
        data['split']='train';del observed[0]['server_before']['server_on_ground']
        self.assertEqual(module.goal_variation_requests(data,observed,'dense')[0],[])


if __name__=='__main__':unittest.main()
