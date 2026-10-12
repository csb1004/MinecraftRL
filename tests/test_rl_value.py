import unittest
from minecraftrl.rl_value import discounted_returns,seed_folds,fit_residual_value,calibration_passes


class ValueTargetsTests(unittest.TestCase):
    def test_gate_scales_world_count_and_rejects_early_regression(self):
        scores=dict(inherited=dict(mse=.13,early_mse=.014),constant=dict(mse=.08),calibrated=dict(mse=.05,early_mse=.01))
        self.assertTrue(calibration_passes(scores,6,8))
        self.assertFalse(calibration_passes(scores,11,16))
        self.assertTrue(calibration_passes(scores,12,16))
        scores['calibrated']['early_mse']=.02
        self.assertFalse(calibration_passes(scores,16,16))

    def test_terminal_and_truncated_returns(self):
        self.assertEqual(discounted_returns([0,1],999,True,gamma=.5),[.5,1.])
        self.assertEqual(discounted_returns([0,0],4,False,gamma=.5),[1.,2.])
        with self.assertRaises(ValueError):discounted_returns([0],float('nan'),False)

    def test_seed_folds_exclude_all_repeats_and_eval(self):
        data=[dict(seed=s,split='train') for s in (1,1,2,2,3,3)]
        for seed,train,test in seed_folds(data):
            self.assertTrue(all(data[i]['seed']!=seed for i in train))
            self.assertEqual([data[i]['seed'] for i in test],[seed,seed])
        with self.assertRaises(ValueError):list(seed_folds(data+[dict(seed=4,split='eval')]))

    def test_fit_recovers_linear_relation_and_does_not_mutate_inputs(self):
        try:import torch
        except ImportError:self.skipTest('tensor runtime required')
        x=torch.tensor([[-2.,4.],[-1.,1.],[0.,0.],[1.,1.],[2.,4.]],dtype=torch.float64)
        y=3*x[:,0]-2*x[:,1]+5;w=torch.zeros(1,2,dtype=torch.float64);b=torch.zeros(1,dtype=torch.float64)
        fit=fit_residual_value(x,y,w,b,ridge=1e-9)
        pred=torch.nn.functional.linear(x,fit['weight'],fit['bias']).flatten()
        self.assertLess((pred-y).abs().max().item(),1e-7)
        self.assertEqual(w.abs().sum().item(),0);self.assertEqual(b.item(),0)
