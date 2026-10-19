import math
import unittest
from copy import deepcopy

from minecraftrl.local_execution import configure_stall,mining_temperature
from minecraftrl.stall_detection import StallDetector


class LocalExecutionTests(unittest.TestCase):
    def mining(self):
        obs=dict(server_available=1,server_alive=1,screen=0,error=0,wood_v3_kind=1,
            wood_server_mining=1,wood_v3_reachable=1)
        for prefix in ('wood_v3_target_','wood_mining_','hit_'):
            obs.update({prefix+a:v for a,v in zip('xyz',(4,12,-3))})
        return obs

    def test_cold_sampling_requires_current_matching_log_interaction(self):
        obs=self.mining()
        self.assertEqual(mining_temperature(obs,'resumed'),.35)
        self.assertEqual(mining_temperature(obs,'collect'),.35)
        for phase in ('subgoal','waypoint','release','refresh','paused'):
            self.assertEqual(mining_temperature(obs,phase),1.)
        for key,value in [('wood_server_mining',0),('hit_x',5),('wood_v3_kind',2),
                          ('screen',1),('server_alive',0),('wood_v3_reachable',0)]:
            bad=deepcopy(obs);bad[key]=value
            self.assertEqual(mining_temperature(bad,'resumed'),1.)

    def test_local_windows_do_not_change_legacy_detectors(self):
        local=StallDetector();configure_stall(local,'search')
        self.assertEqual(local.window,80);self.assertEqual(local.goal_budget_ticks,160)
        self.assertEqual(StallDetector().goal_budget_ticks,800)
        self.assertEqual(StallDetector().windows['search'],400)

    def test_temperature_changes_normalized_distribution_not_actions(self):
        import torch
        from minecraftrl.rl_policy import guided
        def head(_):return {'buttons':torch.tensor([[[0.,1.]],[[0.,1.]]])}
        warm=guided(head,None);cold=guided(head,None,.35)
        self.assertGreater(cold['buttons'].exp()[0,0,1],warm['buttons'].exp()[0,0,1])
        self.assertAlmostEqual(cold['buttons'].exp().sum().item(),1.,places=6)
        for value in (0,-1,2,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):guided(head,None,value)


if __name__=='__main__':unittest.main()
