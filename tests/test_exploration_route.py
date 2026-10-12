import unittest
from minecraftrl.exploration_route import ExplorationRoute, pose_only


def pose(x=0.,z=0.,yaw=-90.):
    return dict(server_x=x,server_y=201.,server_z=z,yaw=yaw,pitch=0.,screen=0,error=0,server_available=1)


class ExplorationTests(unittest.TestCase):
    def test_pose_boundary_does_not_read_map_or_target(self):
        class Restricted(dict):
            def __getitem__(self,key):
                if key not in pose():raise AssertionError('privileged map read')
                return super().__getitem__(key)
        g=Restricted(pose());g['wood_target_x']=999;g['wall_end']=999
        self.assertEqual(pose_only(g),pose())
        a=ExplorationRoute(pose_only(g),[])
        b=ExplorationRoute(pose(),[])
        self.assertEqual(a.targets,b.targets)
        self.assertEqual(a.decisions[0]['traversability'],'unknown')

    def test_failed_candidate_is_not_immediately_repeated(self):
        r=ExplorationRoute(pose(),[]);first=r.targets[0]
        for i in range(41):r.observe(pose(),sequence=i)
        self.assertEqual(len(r.failures),1)
        self.assertNotEqual(r.targets[r.index],first)
        self.assertEqual(r.receipts,[])

    def test_real_arrivals_end_after_three_and_stop_does_not_advance(self):
        r=ExplorationRoute(pose(),[])
        for i in range(1,4):
            t=r.targets[r.index];g=pose(t[0],t[2])
            self.assertIsNone(r.observe(g,sequence=i,stopped=True))
            self.assertEqual(len(r.receipts),i-1)
            r.observe(g,sequence=i)
        self.assertTrue(r.completed)
        self.assertEqual([x['actions'] for x in r.receipts],[1,1,1])

    def test_visited_direction_cost_changes_candidate_without_map(self):
        a=ExplorationRoute(pose(),[])
        target=a.targets[0]
        b=ExplorationRoute(pose(),[target]*8)
        self.assertNotEqual(a.targets[0],b.targets[0])

    def test_sequence_regression_and_invalid_pose_rejected(self):
        r=ExplorationRoute(pose(),[]);r.observe(pose(),sequence=2)
        with self.assertRaises(ValueError):r.observe(pose(),sequence=1)
        bad=pose();bad['server_x']=float('nan')
        with self.assertRaises(ValueError):ExplorationRoute(bad,[])


if __name__=='__main__':unittest.main()
