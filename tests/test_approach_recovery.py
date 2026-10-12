import unittest
from minecraftrl import rl_core


def obs(tick=0,x=0.,distance=22.,kind=1,reachable=0,target=10,logs=0):
    return dict(server_tick=tick,server_inventory={'oak_log':logs},server_x=x,server_y=64.,server_z=0.,
                wood_v3_kind=kind,wood_v3_distance=distance,wood_v3_reachable=reachable,
                wood_v3_target_x=target,wood_v3_target_y=64.,wood_v3_target_z=0.)


class ApproachRecoveryTests(unittest.TestCase):
    def machine(self):return rl_core.SHAPED_REWARDS['wood-potential-v4'](obs())

    def test_stationary_approach_enters_search_after_full_window(self):
        r=self.machine()
        for i in range(1,200):self.assertEqual(r.advance(obs(i))['stage'],'approach')
        result=r.advance(obs(200))
        self.assertEqual(result['stage'],'search')
        self.assertEqual(result['recovery']['event'],'entered')
        self.assertEqual(r.context(),[1.,0.,0.,0.,0.,0.])
        self.assertLess(result['shaping'],0.)

    def test_continuing_approach_and_detour_do_not_trigger(self):
        for mode in ('closer','detour'):
            r=self.machine()
            for i in range(1,401):
                result=r.advance(obs(i,x=i*.02,distance=22-i*.01 if mode=='closer' else 22+i*.01))
                self.assertFalse(result['recovery']['active'])

    def test_readiness_and_collection_do_not_trigger(self):
        for kind,reachable in ((1,1),(2,0)):
            r=self.machine()
            for i in range(1,251):
                self.assertFalse(r.advance(obs(i,kind=kind,reachable=reachable))['recovery']['active'])

    def test_recovery_requires_real_progress_or_new_target(self):
        r=self.machine()
        for i in range(1,201):r.advance(obs(i))
        self.assertTrue(r.advance(obs(201,x=5.,distance=23.))['recovery']['active'])
        end=r.advance(obs(202,x=5.,distance=20.9))
        self.assertEqual(end['recovery']['event'],'approach_progress')
        self.assertEqual(end['stage'],'approach')
        for i in range(203,404):r.advance(obs(i,x=5.,distance=20.9))
        end=r.advance(obs(404,target=11))
        self.assertEqual(end['recovery']['event'],'target_changed')

    def test_terminal_success_overrides_recovery_without_extra_reward(self):
        r=self.machine()
        for i in range(1,201):r.advance(obs(i))
        end=r.advance(obs(201,logs=1))
        self.assertEqual(end['task'],1.)
        self.assertEqual(end['stage'],'complete')
        self.assertEqual(end['potential_after'],0.)
        with self.assertRaises(RuntimeError):r.advance(obs(202,logs=1))

    def test_tick_gap_or_bad_position_does_not_silently_advance_detector(self):
        r=self.machine()
        with self.assertRaises(ValueError):r.advance(obs(3))
        bad=obs(1);bad['server_x']=float('nan')
        with self.assertRaises(ValueError):r.advance(bad)
        self.assertEqual(r.advance(obs(1))['recovery']['window_ticks'],1)

    def test_recovery_cycle_shaping_telescopes(self):
        r=self.machine();initial=r.potential;total=0.
        for i in range(1,202):
            part=r.advance(obs(i,distance=20.9 if i==201 else 22.),terminal=i==201)
            total+=r.gamma**(i-1)*part['shaping']
        self.assertAlmostEqual(total,-r.alpha*initial)
        self.assertEqual(part['failure'],-1.)
