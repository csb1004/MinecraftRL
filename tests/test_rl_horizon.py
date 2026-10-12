import unittest
from minecraftrl.rl_horizon import censored_targets,checked_training_bundle
from minecraftrl.rl_core import gae

class HorizonTests(unittest.TestCase):
    def test_completed_return_and_censored_interval(self):
        self.assertEqual(censored_targets([0,1],True,.9,.5),([.9,1.],[.9,1.],[.9,1.]))
        target,low,high=censored_targets([0,0],False,.9,.5)
        self.assertAlmostEqual(target[0],.405)
        self.assertEqual(low,[0.,0.]);self.assertAlmostEqual(high[0],.81)
        with self.assertRaises(ValueError):censored_targets([0],False,.99,2)

    def test_mc_reaches_first_step_without_value_leakage(self):
        rewards=[0.]*300+[1.];values=[.37]*301
        a,r=gae(rewards,values,7,terminal=True,gamma=.999,lam=1)
        self.assertAlmostEqual(r[0],.999**300,places=12)
        self.assertAlmostEqual(a[0],.999**300-.37,places=12)

    def test_bundle_rejects_wrong_order_nan_and_mismatched_discount(self):
        b=dict(schema=1,gamma=.999,lam=1.,trajectory_sha256=['abc'],advantages=[.2],returns=[.7],value={'weight':0,'bias':0})
        checked_training_bundle(b,['abc'],1)
        for patch in ({'trajectory_sha256':['def']},{'advantages':[float('nan')]},{'gamma':1.1},{'returns':[]}):
            with self.assertRaises(ValueError):checked_training_bundle(dict(b,**patch),['abc'],1)
