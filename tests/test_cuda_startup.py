import unittest

from minecraftrl.cuda_startup import prime_convolution


class ConvolutionPreparationTests(unittest.TestCase):
    def test_primes_without_changing_weights_rng_or_training_mode(self):
        try:
            import torch
        except ImportError:
            self.skipTest('tensor runtime required')
        layer=torch.nn.Conv2d(3,4,3,padding=1).eval()
        weights={k:v.clone() for k,v in layer.state_dict().items()}
        rng=torch.get_rng_state().clone();seen=[]
        hook=layer.register_forward_pre_hook(lambda m,args:seen.append((tuple(args[0].shape),tuple(args[0].stride()),torch.is_grad_enabled())))
        result=prime_convolution(layer)
        hook.remove()
        self.assertEqual(len(seen),1)
        self.assertEqual(seen[0][0],(2,3,128,128))
        self.assertEqual(seen[0][1][1],1)
        self.assertFalse(seen[0][2])
        self.assertFalse(layer.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,layer.state_dict()[k]) for k,v in weights.items()))
        self.assertTrue(result['completed']);self.assertGreaterEqual(result['seconds'],0)

    def test_failure_is_not_reported_as_success(self):
        try:
            import torch
        except ImportError:
            self.skipTest('tensor runtime required')
        layer=torch.nn.Conv2d(3,4,3,padding=1)
        with torch.no_grad():layer.weight.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            prime_convolution(layer)
