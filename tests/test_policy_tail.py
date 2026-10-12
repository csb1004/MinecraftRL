import unittest
from types import SimpleNamespace
import torch
from minecraftrl.policy_tail import load_tail,tail_state


class TailTests(unittest.TestCase):
    def net(self):return SimpleNamespace(lastlayer=torch.nn.Linear(3,3),final_ln=torch.nn.LayerNorm(3))

    def test_tail_checkpoint_restores_both_modules(self):
        a,b=self.net(),self.net();saved=tail_state(a)
        load_tail(b,dict(scope='head_and_tail',backbone_tail=saved))
        x=torch.tensor([[1.,2.,3.]])
        self.assertTrue(torch.equal(a.final_ln(a.lastlayer(x)),b.final_ln(b.lastlayer(x))))

    def test_no_silent_drop_or_partial_load(self):
        a=self.net();saved=tail_state(a);saved['final_ln']['weight']=torch.ones(4)
        before=a.lastlayer.weight.detach().clone()
        for checkpoint in (dict(scope='head_only',backbone_tail=saved),dict(scope='head_and_tail'),
                           dict(scope='head_and_tail',backbone_tail=saved)):
            with self.assertRaises(ValueError):load_tail(a,checkpoint)
        self.assertTrue(torch.equal(before,a.lastlayer.weight))

    def test_legacy_head_checkpoint_is_unchanged(self):
        a=self.net();before=a.lastlayer.weight.detach().clone()
        self.assertFalse(load_tail(a,dict(scope='head_only')))
        self.assertTrue(torch.equal(before,a.lastlayer.weight))
