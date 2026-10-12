import unittest
from minecraftrl.rl_core import WoodReward,gae,validate_training_data

class RLCoreTests(unittest.TestCase):
    def test_reward_terminal_latch_and_unavailable_server(self):
        r=WoodReward({'oak_log':0})
        self.assertEqual(r.observe({'oak_log':0}),0)
        self.assertEqual(r.observe({'oak_log':1}),1)
        self.assertTrue(r.completed)
        self.assertEqual(r.observe({'oak_log':0}),0)
        self.assertEqual(r.observe({'oak_log':1}),0)
        with self.assertRaises(ValueError):WoodReward({'oak_log':1})
        with self.assertRaises(ValueError):WoodReward({}).observe(None)
    def test_terminal_no_bootstrap_and_truncation_bootstrap(self):
        a,ret=gae([0,1],[0,0],0,terminal=True,gamma=.9,lam=1)
        self.assertEqual(a,[.9,1]);self.assertEqual(ret,a)
        a,ret=gae([0],[2],3,terminal=False,gamma=.9,lam=1)
        self.assertAlmostEqual(ret[0],2.7)
        self.assertEqual(gae([1],[2],999,terminal=True)[1],[1])
    def test_eval_and_cancelled_data_cannot_train(self):
        good=dict(eligible=True,split='train',scope='head_only_frozen_features',rows=[dict(old_logp=-1.,value=0.,reward=1.)])
        validate_training_data(good)
        for field,value in (('split','eval'),('eligible',False),('rows',[])):
            with self.assertRaises(ValueError):validate_training_data(dict(good,**{field:value}))
        with self.assertRaises(ValueError):validate_training_data(dict(good,rows=[dict(old_logp=float('nan'),value=0.,reward=1.)]))

    def test_server_reward_ignores_client_prediction_and_blocks_next_action(self):
        from types import SimpleNamespace
        from minecraftrl.rl_collect import RLEnvironment
        class Env:
            def __init__(self):self.calls=0;self.raw=SimpleNamespace(action_space=SimpleNamespace(noop=lambda:{}))
            def reset(self):pass
            def step(self,action):
                self.calls+=1
                return {'inventory':{'oak_log':99}},0,False,{'workshop':{'error':0,'server_available':1,'server_inventory':{'oak_log':int(self.calls>=3)}}}
        commits=[];env=Env();wrapper=RLEnvironment(env,SimpleNamespace(commit=lambda *a:commits.append(a)))
        wrapper.reset();self.assertEqual(wrapper.step({})[1],0)
        self.assertEqual(wrapper.step({})[1],1)
        with self.assertRaises(RuntimeError):wrapper.step({})
        self.assertEqual(env.calls,3);self.assertEqual(len(commits),2)
