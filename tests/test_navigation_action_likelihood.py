import importlib.util
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Torch unavailable')
class DeliveredActionProbabilityTests(unittest.TestCase):
    def setUp(self):
        from minecraftrl.navigation_action_likelihood import DeliveredActionLikelihood
        # One physical button action has off/on encodings; another only has on.
        self.likelihood=DeliveredActionLikelihood(SimpleNamespace(
            BUTTON_IDX_TO_FACTORED=[[1,0],[1,0],[0,1]],
            BUTTON_IDX_TO_CAMERA_META_OFF=[True,False,False],camera_null_idx=1))
        self.pd={'buttons':torch.tensor([[[[.3,.2,.5]]]]).log(),
                 'camera':torch.tensor([[[[.2,.1,.7]]]]).log()}

    def action(self,b,c):return {'buttons':torch.tensor([[[b]]]),'camera':torch.tensor([[[c]]])}

    def probability(self,b,c):return self.likelihood(self.pd,self.action(b,c)).exp().item()

    def test_all_gated_camera_outputs_and_on_null_are_the_same_delivered_action(self):
        # Canonical latent probability is .3*.1=.03; delivered probability is .3+.2*.1=.32.
        for b,c in ((0,0),(0,1),(0,2),(1,1)):
            self.assertAlmostEqual(self.probability(b,c),.32,places=6)
        self.assertTrue(self.likelihood.matches(self.action(0,2),self.action(1,1)).item())

    def test_nonzero_camera_still_requires_both_on_button_and_camera_bin(self):
        self.assertAlmostEqual(self.probability(1,0),.2*.2,places=6)
        self.assertAlmostEqual(self.probability(1,2),.2*.7,places=6)
        self.assertFalse(self.likelihood.matches(self.action(1,0),self.action(0,0)).item())

    def test_probability_sums_to_one_across_distinct_delivered_actions(self):
        unique=((0,1),(1,0),(1,2),(2,0),(2,1),(2,2))
        self.assertAlmostEqual(sum(self.probability(b,c) for b,c in unique),1.,places=6)

    def test_mapper_singleton_on_is_not_assumed_to_suppress_camera(self):
        self.assertAlmostEqual(self.probability(2,0),.5*.2,places=6)
        self.assertFalse(self.likelihood.matches(self.action(2,0),self.action(2,1)).item())

    def test_gradients_are_finite_with_missing_off_partner(self):
        logits={k:v.clone().requires_grad_() for k,v in self.pd.items()}
        pd={k:torch.log_softmax(v,-1) for k,v in logits.items()}
        (-self.likelihood(pd,self.action(2,1)).mean()).backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in logits.values()))

    def test_equivalent_delivered_targets_have_identical_training_gradients(self):
        gradients=[]
        for action in (self.action(0,0),self.action(0,2),self.action(1,1)):
            logits={k:v.clone().requires_grad_() for k,v in self.pd.items()}
            pd={k:torch.log_softmax(v,-1) for k,v in logits.items()}
            (-self.likelihood(pd,action).mean()).backward()
            gradients.append({k:v.grad for k,v in logits.items()})
        for actual in gradients[1:]:
            self.assertTrue(all(torch.equal(actual[k],v) for k,v in gradients[0].items()))

    def test_physical_likelihood_matches_explicit_latent_marginal_gradient(self):
        first={k:v.clone().requires_grad_() for k,v in self.pd.items()}
        second={k:v.clone().requires_grad_() for k,v in self.pd.items()}
        pd={k:torch.log_softmax(v,-1) for k,v in first.items()}
        (-self.likelihood(pd,self.action(0,2)).mean()).backward()
        probabilities={k:torch.softmax(v,-1) for k,v in second.items()}
        pb,pc=probabilities['buttons'],probabilities['camera']
        # Explicitly sum every camera-bin encoding for the gated-off command.
        marginal=(pb[...,0:1]*pc).sum(-1)+pb[...,1]*pc[...,1]
        (-marginal.log().mean()).backward()
        for key in first:self.assertTrue(torch.allclose(first[key].grad,second[key].grad,atol=1e-7))

    def test_delivered_mode_aggregates_gated_aliases_before_selection(self):
        pd={'buttons':torch.tensor([[[[.3,.2,.5]]]]).log(),
            'camera':torch.tensor([[[[.3,.3,.4]]]]).log()}
        # Independent latent modes choose the second physical button (.5*.4=.2),
        # but the first button with no camera is more probable (.3+.2*.3=.36).
        selected=self.likelihood.mode(pd)
        self.assertTrue(self.likelihood.matches(selected,self.action(0,1)).item())
        self.assertAlmostEqual(self.likelihood(pd,selected).exp().item(),.36,places=6)

    def test_delivered_mode_matches_exhaustive_marginalization_in_multiple_rows(self):
        generator=torch.Generator().manual_seed(17801)
        pd={k:torch.log_softmax(torch.randn(2,3,1,n,generator=generator),-1)
            for k,n in (('buttons',3),('camera',3))}
        selected=self.likelihood.mode(pd)
        self.assertEqual(selected['buttons'].shape,(2,3,1))
        for b in range(2):
            for t in range(3):
                masses={}
                for ib,buttons in enumerate(((1,0),(1,0),(0,1))):
                    for ic in range(3):
                        key=(buttons,1 if ib==0 else ic)
                        masses[key]=masses.get(key,0.)+float(pd['buttons'][b,t,0,ib].exp()*pd['camera'][b,t,0,ic].exp())
                ib=int(selected['buttons'][b,t,0]);ic=int(selected['camera'][b,t,0])
                key=(((1,0),(1,0),(0,1))[ib],1 if ib==0 else ic)
                self.assertAlmostEqual(masses[key],max(masses.values()),places=6)

    def test_delivered_mode_handles_zero_probabilities_and_singleton_on_group(self):
        pd={'buttons':torch.tensor([[[[0.,0.,1.]]]]).log(),
            'camera':torch.tensor([[[[.1,.2,.7]]]]).log()}
        self.assertTrue(self.likelihood.matches(self.likelihood.mode(pd),self.action(2,2)).item())

    def test_delivered_mode_ties_repeat_without_consuming_randomness(self):
        pd={k:torch.log_softmax(torch.zeros_like(v),-1) for k,v in self.pd.items()}
        before=torch.random.get_rng_state().clone()
        first=self.likelihood.mode(pd);second=self.likelihood.mode(pd)
        self.assertTrue(all(torch.equal(first[k],second[k]) for k in first))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_delivered_mode_rejects_distribution_without_possible_commands(self):
        pd={k:torch.full_like(v,float('-inf')) for k,v in self.pd.items()}
        with self.assertRaises(ValueError):self.likelihood.mode(pd)

    @unittest.skipIf(importlib.util.find_spec('minerl') is None,'official WSL MineRL required')
    def test_official_delivered_mode_matches_sparse_exhaustive_decoded_commands(self):
        import sys
        from pathlib import Path
        import numpy as np
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/steve1'))
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from minecraftrl.navigation_action_likelihood import DeliveredActionLikelihood
        mapper=CameraHierarchicalMapping(n_camera_bins=11)
        likelihood=DeliveredActionLikelihood(mapper)
        pb=torch.zeros(len(mapper.BUTTON_IDX_TO_FACTORED),dtype=torch.float64)
        pc=torch.zeros(121,dtype=torch.float64)
        bi=[0,1,2,3,len(pb)-1];ci=[0,7,mapper.camera_null_idx,97,120]
        pb[bi]=torch.tensor([.3,.1,.2,.15,.25],dtype=torch.float64)
        pc[ci]=torch.tensor([.2,.1,.3,.15,.25],dtype=torch.float64)
        pd={'buttons':pb.log().view(1,1,1,-1),'camera':pc.log().view(1,1,1,-1)}
        def key(b,c):
            decoded=mapper.to_factored(dict(buttons=np.array([[b]]),camera=np.array([[c]])))
            return tuple(decoded['buttons'].reshape(-1)),tuple(decoded['camera'].reshape(-1))
        mass={}
        for b in bi:
            for c in ci:mass[key(b,c)]=mass.get(key(b,c),0.)+float(pb[b]*pc[c])
        selected=likelihood.mode(pd)
        self.assertAlmostEqual(mass[key(selected['buttons'].item(),selected['camera'].item())],max(mass.values()),places=12)
        if torch.cuda.is_available():
            gpu=likelihood.cuda().mode({k:v.cuda() for k,v in pd.items()})
            self.assertTrue(all(torch.equal(gpu[k].cpu(),selected[k]) for k in selected))

    @unittest.skipIf(importlib.util.find_spec('minerl') is None,'official WSL MineRL required')
    def test_official_mapper_aliases_match_decoded_buttons_and_camera(self):
        import sys
        from pathlib import Path
        import numpy as np
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/steve1'))
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from minecraftrl.navigation_action_likelihood import DeliveredActionLikelihood
        mapper=CameraHierarchicalMapping(n_camera_bins=11)
        likelihood=DeliveredActionLikelihood(mapper)
        for b in (0,1,2,3,len(mapper.BUTTON_IDX_TO_FACTORED)-1):
            for c in (0,mapper.camera_null_idx,120):
                a=self.action(b,c)
                decoded=mapper.to_factored({k:v[:,0].numpy() for k,v in a.items()})
                canonical=mapper.from_factored(decoded)
                again=mapper.to_factored(canonical)
                # Inventory can be canonicalized to a different actual camera by from_factored.
                expected=all(np.array_equal(decoded[k],again[k]) for k in decoded)
                target={k:torch.as_tensor(v).unsqueeze(1) for k,v in canonical.items()}
                self.assertEqual(likelihood.matches(a,target).item(),expected)

    @unittest.skipIf(importlib.util.find_spec('minerl') is None,'official WSL MineRL required')
    def test_real_vpt_head_shape_and_multirow_greedy_commands(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.action_head import make_action_head
        from minecraftrl.navigation_action_likelihood import DeliveredActionLikelihood
        mapper=CameraHierarchicalMapping(n_camera_bins=11)
        head=make_action_head(DictType(**mapper.get_action_space_update()),8)
        likelihood=DeliveredActionLikelihood(mapper)
        pd=head(torch.randn(2,3,8))
        actions={'buttons':torch.zeros(2,3,1,dtype=torch.long),
                 'camera':torch.full((2,3,1),mapper.camera_null_idx,dtype=torch.long)}
        actual=likelihood(pd,actions);latent=head.logprob(actions,pd)
        self.assertEqual(actual.shape,(2,3))
        self.assertTrue(torch.all(actual>=latent-1e-6))
        greedy={k:v.argmax(-1) for k,v in pd.items()}
        self.assertEqual(likelihood.matches(greedy,actions).shape,(2,3))
        (-actual.mean()).backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in head.parameters()))


if __name__=='__main__':unittest.main()
