import unittest
from minecraftrl import rl_core


def observation(tick=0, distance=8., kind=1, reachable=0, logs=0):
    return dict(server_tick=tick, server_inventory={'oak_log': logs},
                wood_kind=kind, wood_distance=distance, wood_reachable=reachable)


class WoodShapingTests(unittest.TestCase):
    def test_collector_records_components_and_updates_policy_context(self):
        from types import SimpleNamespace
        from minecraftrl.rl_collect import RLEnvironment
        class Env:
            def __init__(self):self.calls=0;self.raw=SimpleNamespace(action_space=SimpleNamespace(noop=lambda:{}))
            def reset(self):pass
            def step(self,action):
                self.calls+=1
                gui=observation(self.calls,8-self.calls,logs=int(self.calls==3))
                gui.update(server_available=1,error=0)
                return {'pov':None},0,False,{'workshop':gui}
        class Policy:
            def __init__(self):self.rows=[];self.contexts=[]
            def set_reward_context(self,context):self.contexts.append(context)
            def commit(self,*args):self.rows.append({})
        import inspect
        self.assertIn('reward_mode',inspect.signature(RLEnvironment).parameters)
        p=Policy();wrapper=RLEnvironment(Env(),p,reward_mode='wood-potential-v1')
        wrapper.reset();wrapper.step({});wrapper.step({})
        self.assertEqual(p.rows[-1]['reward_components']['task'],1)
        self.assertEqual(len(p.contexts),3)
        with self.assertRaises(RuntimeError):wrapper.step({})

    def machine(self, obs=None):
        self.assertTrue(hasattr(rl_core, 'ShapedWoodReward'), 'stage reward not implemented')
        return rl_core.ShapedWoodReward(observation() if obs is None else obs)

    def test_approach_beats_retreat_and_no_motion_costs_time(self):
        rewards=[]
        for distance in (7.,8.,9.):
            r=self.machine(); rewards.append(r.advance(observation(1,distance))['total'])
        self.assertGreater(rewards[0],rewards[1]);self.assertGreater(rewards[1],rewards[2])
        self.assertLess(rewards[1],0)

    def test_discounted_round_trip_has_no_positive_shaping(self):
        r=self.machine();a=r.advance(observation(1,7.));b=r.advance(observation(2,8.))
        self.assertLessEqual(a['shaping']+r.gamma*b['shaping'],0)

    def test_target_loss_and_reacquisition_cannot_mint_shaping(self):
        r=self.machine();a=r.advance(observation(1,0,kind=0));b=r.advance(observation(2,8.))
        self.assertLessEqual(a['shaping']+r.gamma*b['shaping'],0)

    def test_inventory_accepts_immediately_even_without_target(self):
        r=self.machine();result=r.advance(observation(1,0,kind=0,logs=1))
        self.assertTrue(r.completed);self.assertEqual(result['stage'],'complete')
        self.assertEqual(result['task'],1);self.assertEqual(result['potential_after'],0)
        with self.assertRaises(RuntimeError):r.advance(observation(2,logs=1))

    def test_stages_and_terminal_failure(self):
        r=self.machine(observation(kind=0));self.assertEqual(r.stage,'search')
        self.assertEqual(r.advance(observation(1))['stage'],'approach')
        self.assertEqual(r.advance(observation(2,2,reachable=1))['stage'],'mine')
        self.assertEqual(r.advance(observation(3,2,kind=2))['stage'],'collect')
        result=r.advance(observation(4,2,kind=2),terminal=True)
        self.assertEqual(result['potential_after'],0);self.assertEqual(result['task'],0)

    def test_missing_nonfinite_and_tick_gaps_fail_closed(self):
        for obs in ({},observation(distance=float('nan')),observation(kind=3)):
            with self.assertRaises(ValueError):self.machine(obs)
        r=self.machine()
        with self.assertRaises(ValueError):r.advance(observation(2))

    def test_terminal_shaping_telescopes_across_stage_changes(self):
        r=self.machine();initial=r.potential;total=0
        seq=[observation(1,5),observation(2,2,reachable=1),observation(3,2,kind=2),observation(4,logs=1)]
        for i,obs in enumerate(seq):total+=r.gamma**i*r.advance(obs)['shaping']
        self.assertAlmostEqual(total,-r.alpha*initial)

    def test_training_rejects_wrong_discount_contract(self):
        data=dict(eligible=True,split='train',scope='head_only_frozen_features',
                  rows=[dict(old_logp=0.,value=0.,reward=0.)],
                  reward_config=dict(name='wood-potential-v1',gamma=.999))
        with self.assertRaises(ValueError):rl_core.validate_training_data(data)

    def test_terminal_failure_cost_prevents_early_failure_advantage(self):
        self.assertTrue(hasattr(rl_core,'ShapedWoodRewardV2'))
        r=rl_core.ShapedWoodRewardV2(observation(kind=0))
        failure=r.advance(observation(1,kind=0),terminal=True)
        self.assertEqual(failure['failure'],-1.)
        self.assertEqual(failure['task'],0.)
        # Independent full discounted objective, same initial potential=0.
        gamma=.99;cost=.0001
        success=gamma**999-cost*(1-gamma**1000)/(1-gamma)
        self.assertGreater(success,failure['total'])
        for length in (1,100,1000,3600):
            failed=-gamma**(length-1)-cost*(1-gamma**length)/(1-gamma)
            self.assertGreater(success,failed)
        r=rl_core.ShapedWoodRewardV2(observation(kind=0))
        self.assertEqual(r.advance(observation(1,logs=1),terminal=True)['failure'],0.)
        r=rl_core.ShapedWoodRewardV2(observation(kind=0))
        self.assertEqual(r.advance(observation(1))['failure'],0.)
