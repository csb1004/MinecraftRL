import unittest
try:import torch
except ImportError:torch=None

@unittest.skipIf(torch is None,'training tensor runtime required')
class OptimizationTests(unittest.TestCase):
    def test_context_extension_preserves_old_logits_and_learns_context(self):
        from minecraftrl import rl_policy
        self.assertTrue(hasattr(rl_policy,'extend_context_head'))
        head=torch.nn.Sequential(torch.nn.Linear(3,2))
        x=torch.tensor([[1.,2.,3.]])
        before=head(x).detach().clone()
        rl_policy.extend_context_head(head,3,6)
        full=torch.cat([x,torch.tensor([[1.,0.,0.,0.,0.,.5]])],-1)
        torch.testing.assert_close(before,head(full),rtol=1e-6,atol=1e-6)
        head(full).sum().backward()
        self.assertGreater(head[0].weight.grad[:,3:].abs().sum().item(),0)

    def test_minibatch_preserves_rollout_forward_shape_and_gradients(self):
        from minecraftrl.rl_policy import guided,guided_minibatch
        class Head(torch.nn.Module):
            def __init__(self):super().__init__();self.linear=torch.nn.Linear(2,3)
            def forward(self,x):
                if x.shape[1]!=1 or not x.is_contiguous():raise AssertionError('changed collection GEMM shape/strides')
                return {'buttons':torch.log_softmax(self.linear(x),dim=-1)}
        head=Head();features=torch.randn(2,3,2)
        actual=guided_minibatch(head,features)['buttons']
        expected=torch.cat([guided(head,features[:,i:i+1].contiguous())['buttons'] for i in range(3)],1)
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(tuple(actual.shape),(1,3,3))
        (-actual[:,:,0].mean()).backward()
        self.assertTrue(torch.isfinite(head.linear.weight.grad).all());self.assertGreater(head.linear.weight.grad.abs().sum().item(),0)

    def test_guidance_is_normalized_and_ppo_clips_positive_advantage(self):
        from minecraftrl.rl_opt import ppo_loss
        from minecraftrl.rl_policy import guided
        class Head(torch.nn.Module):
            def forward(self,x):return {'buttons':torch.log_softmax(x,dim=-1)}
        pd=guided(Head(),torch.tensor([[[2.,1.]],[[0.,3.]]]))['buttons']
        self.assertTrue(torch.allclose(pd.exp().sum(-1),torch.ones(1,1)))
        x=torch.tensor([2.],requires_grad=True)
        loss=ppo_loss(x.log(),torch.zeros(1),torch.ones(1),.2)
        self.assertAlmostEqual(loss.item(),-1.2,places=5)
        loss.backward();self.assertEqual(x.grad.item(),0)
