"""Behavioral checks for the optional, initially neutral visual residual."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import test_navigation_conditioning as conditioning
try:
    import torch
    from minecraftrl import navigation_model as model
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Torch unavailable')
class NavigationVisualAdapterTests(unittest.TestCase):
    def setUp(self):
        conditioning.NavigationConditioningTests.setUp(self)
        self.model=model

    head=conditioning.NavigationConditioningTests.head

    def adapted(self):
        head=self.head('mlp')
        model.set_root_context_mode(head,'ignored')
        model.attach_visual_adapter(head)
        return head

    def test_attachment_preserves_initial_function_state_and_rng(self):
        head=self.head('mlp');model.set_root_context_mode(head,'ignored')
        before=copy.deepcopy(head.state_dict());expected=head(self.x)
        rng=torch.get_rng_state().clone()
        receipt=model.attach_visual_adapter(head)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(receipt['heads'],2)
        self.assertGreater(receipt['additional_parameters'],0)
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(torch.equal(v,head(self.x)[k]) for k,v in expected.items()))

    def test_disabled_branch_matches_legacy_keys_values_and_rng(self):
        rng=torch.get_rng_state().clone()
        default=model.refinement_head(self.prior,'mlp',16,'ignored')
        after=torch.get_rng_state().clone();torch.set_rng_state(rng)
        explicit=model.refinement_head(self.prior,'mlp',16,'ignored',0)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertEqual(set(default.state_dict()),set(explicit.state_dict()))
        self.assertFalse(any('visual_adapter' in k for k in default.state_dict()))
        self.assertTrue(all(torch.equal(v,explicit(self.x)[k]) for k,v in default(self.x).items()))

    def test_normalized_branch_learns_without_mutating_frozen_source(self):
        head=self.adapted()
        receipt=model.configure_refinement_training(head,'visual-adapter-only')
        self.assertEqual(receipt['trainable_parameters'],sum(p.numel() for n,p in head.named_parameters() if 'visual_adapter' in n))
        frozen={k:v.clone() for k,v in head.state_dict().items() if 'visual_adapter' not in k}
        modules=[m for m in head.modules() if isinstance(m,model.RefinedLinear)]
        optimizer=torch.optim.SGD([p for p in head.parameters() if p.requires_grad],lr=.01)
        for step in range(2):
            optimizer.zero_grad();loss=sum(torch.nn.functional.cross_entropy(v.flatten(0,1),torch.zeros(32,dtype=torch.long)) for v in head(self.x).values())
            self.assertTrue(torch.isfinite(loss));loss.backward()
            for m in modules:
                for p in m.visual_adapter.parameters():self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(m.visual_adapter.network[-1].weight.grad.abs().sum().item(),0)
                if step:self.assertGreater(m.visual_adapter.network[0].weight.grad.abs().sum().item(),0)
            optimizer.step()
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in frozen.items()))

    def test_training_scope_preserves_function_rng_and_legacy_parameter_set(self):
        head=self.adapted();before=copy.deepcopy(head.state_dict());expected=head(self.x)
        rng=torch.get_rng_state().clone()
        model.configure_refinement_training(head,'visual-adapter-only')
        self.assertTrue(all(('visual_adapter' in name)==p.requires_grad for name,p in head.named_parameters()))
        model.configure_refinement_training(head)
        self.assertTrue(all(p.requires_grad==any(part in name for part in ('residual','visual_projection','visual_adapter'))
                            for name,p in head.named_parameters()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(torch.equal(v,head(self.x)[k]) for k,v in expected.items()))

    def test_invalid_scope_rejects_without_mutating_gradient_flags(self):
        head=self.head('mlp');before=[p.requires_grad for p in head.parameters()]
        for scope in ('visual-adapter-only','invalid'):
            with self.assertRaises(ValueError):model.configure_refinement_training(head,scope)
            self.assertEqual(before,[p.requires_grad for p in head.parameters()])

    def test_trained_adapter_ignores_rm_and_root_path_stays_exact(self):
        head=self.adapted()
        with torch.no_grad():
            for m in head.modules():
                if isinstance(m,model.RefinedLinear):m.visual_adapter.network[-1].weight.normal_(0,.1)
        changed=self.x.clone();changed[...,2048:2054]=1000
        missing=torch.cat([self.x[...,:2048],self.x[...,2054:]],-1)
        for k,v in head(self.x).items():
            self.assertTrue(torch.equal(v,head(changed)[k]))
            self.assertTrue(torch.equal(v,head(missing)[k]))
        switch=model.NavigationSwitch(self.parent,head,True)
        features=self.x[...,:2054];expected=self.parent(features)
        for goal in (None,self.x[...,-10:],None):
            switch.goal=goal;actual=switch(features)
            if goal is None:self.assertTrue(all(torch.equal(v,actual[k]) for k,v in expected.items()))

    def test_distance_rescale_preserves_trained_adapter(self):
        head=self.adapted()
        with torch.no_grad():
            for m in head.modules():
                if isinstance(m,model.RefinedLinear):m.visual_adapter.network[-1].weight.normal_(0,.1)
        old=copy.deepcopy(head.state_dict());expected=head(self.x)
        model.rescale_distance_input(head,16)
        self.assertTrue(all(torch.equal(v,head(self.x)[k]) for k,v in expected.items()))
        model.rescale_distance_input(head,1)
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in old.items()))

    def test_spatial_training_scale_preserves_function_rng_and_folds_for_export(self):
        head=self.adapted();model.rescale_distance_input(head,16)
        with torch.no_grad():
            for m in head.modules():
                if isinstance(m,model.RefinedLinear):m.visual_adapter.network[-1].weight.normal_(0,.1)
        old=copy.deepcopy(head.state_dict());expected=head(self.x);rng=torch.get_rng_state().clone()
        receipt=model.rescale_adapter_spatial_input(head,8)
        self.assertEqual(receipt['previous_scale'],1)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,head(self.x)[k]) for k,v in expected.items()))
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in old.items() if 'visual_adapter' not in k))
        model.rescale_adapter_spatial_input(head,1)
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in old.items()))
        saved=dict(schema='navigation_refinement_v4',kind='mlp',root_context_mode='ignored',
            residual_distance_scale=16,visual_adapter_width=64,actor=head.state_dict())
        restored,_=model.load_navigation(self.parent,saved)
        self.assertTrue(all(torch.equal(v,restored(self.x)[k]) for k,v in head(self.x).items()))

    def test_spatial_scale_changes_only_selected_first_layer_gradients_then_exports(self):
        ordinary=self.adapted()
        with torch.no_grad():
            for m in ordinary.modules():
                if isinstance(m,model.RefinedLinear):m.visual_adapter.network[-1].weight.normal_(0,.1)
        scaled=copy.deepcopy(ordinary)
        model.configure_refinement_training(ordinary,'visual-adapter-only')
        model.configure_refinement_training(scaled,'visual-adapter-only')
        model.rescale_adapter_spatial_input(scaled,8)
        for head in (ordinary,scaled):
            sum(v.square().mean() for v in head(self.x).values()).backward()
        for (name,p),(other,q) in zip(ordinary.named_parameters(),scaled.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is None:self.assertIsNone(q.grad);continue
            expected=p.grad.clone()
            if name.endswith('visual_adapter.network.0.weight'):expected[:,2048:2052]*=8
            self.assertTrue(torch.equal(expected,q.grad),name)
        optimizer=torch.optim.SGD([p for p in scaled.parameters() if p.requires_grad],lr=.001)
        optimizer.step();expected=scaled(self.x)
        model.rescale_adapter_spatial_input(scaled,1)
        self.assertTrue(all(torch.equal(v,scaled(self.x)[k]) for k,v in expected.items()))

    def test_invalid_spatial_scale_and_partial_head_reject_before_mutation(self):
        for scale in (0,2,4,16,True,8.,'8',None):
            head=self.adapted();old=copy.deepcopy(head.state_dict())
            with self.assertRaises(ValueError):model.rescale_adapter_spatial_input(head,scale)
            self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in old.items()))
        head=self.adapted();modules=[m for m in head.modules() if isinstance(m,model.RefinedLinear)]
        modules[-1].visual_adapter_width=0;old=copy.deepcopy(head.state_dict())
        with self.assertRaises(ValueError):model.rescale_adapter_spatial_input(head,8)
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in old.items()))

    def test_joint_training_with_spatial_scale_preserves_parent_and_exports(self):
        head=self.adapted();model.rescale_distance_input(head,16)
        model.configure_refinement_training(head,'all-residuals');model.rescale_adapter_spatial_input(head,8)
        before=copy.deepcopy(head.state_dict())
        frozen={name:p.clone() for name,p in head.named_parameters() if not p.requires_grad}
        optimizer=torch.optim.SGD([p for p in head.parameters() if p.requires_grad],lr=.001)
        for _ in range(2):
            optimizer.zero_grad();sum(v.square().mean() for v in head(self.x).values()).backward();optimizer.step()
        self.assertTrue(all(torch.equal(value,head.state_dict()[name]) for name,value in frozen.items()))
        for branch in ('residual.','visual_adapter.'):
            self.assertTrue(any(not torch.equal(value,head.state_dict()[name]) for name,value in before.items() if branch in name))
        for name,value in before.items():
            if name.endswith('visual_projection.0.weight'):
                self.assertTrue(torch.equal(value[:,2048:],head.state_dict()[name][:,2048:]))
        expected=head(self.x);model.rescale_adapter_spatial_input(head,1)
        self.assertTrue(all(torch.equal(value,head(self.x)[key]) for key,value in expected.items()))

    def test_v4_roundtrip_and_metadata_rejection(self):
        head=self.adapted();model.rescale_distance_input(head,16)
        with torch.no_grad():
            for m in head.modules():
                if isinstance(m,model.RefinedLinear):m.visual_adapter.network[-1].weight.normal_(0,.1)
        saved=dict(schema='navigation_refinement_v4',kind='mlp',root_context_mode='ignored',
            residual_distance_scale=16,visual_adapter_width=64,actor=head.state_dict())
        restored,refined=model.load_navigation(self.parent,saved);self.assertTrue(refined)
        self.assertTrue(all(torch.equal(v,restored(self.x)[k]) for k,v in head(self.x).items()))
        for key in ('root_context_mode','residual_distance_scale','visual_adapter_width'):
            invalid=dict(saved);del invalid[key]
            with self.assertRaises(ValueError):model.load_navigation(self.parent,invalid)
        for change in (dict(schema='navigation_refinement_v3'),dict(kind='linear'),dict(root_context_mode='observed'),
                       dict(visual_adapter_width=0),dict(visual_adapter_width=True),dict(visual_adapter_width=64.)):
            with self.assertRaises(ValueError):model.load_navigation(self.parent,saved|change)

    def test_invalid_attachment_rejects_before_mutation(self):
        for kind,mode in (('linear','ignored'),('mlp','observed')):
            head=self.head(kind);model.set_root_context_mode(head,mode);old=copy.deepcopy(head.state_dict())
            with self.assertRaises(ValueError):model.attach_visual_adapter(head)
            self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in old.items()))
        head=self.adapted();old=copy.deepcopy(head.state_dict())
        with self.assertRaises(ValueError):model.attach_visual_adapter(head)
        with self.assertRaises(ValueError):model.set_root_context_mode(head,'observed')
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in old.items()))
        for value in (-1,32,None,True,64.,'64'):
            with self.assertRaises(ValueError):model.checked_visual_adapter_width(value)

    def test_runtime_binding_preserves_root_then_dispatches_v4(self):
        from minecraftrl.rm_navigation import RMNavigationController
        from minecraftrl.world_store import digest
        head=self.adapted()
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);parent=folder/'parent';parent.write_bytes(b'parent')
            embed=folder/'embed';embed.write_bytes(b'embed');path=folder/'actor.pt'
            torch.save(dict(schema='navigation_refinement_v4',kind='mlp',motion_dim=10,visual_adapter_width=64,
                root_context_mode='ignored',residual_distance_scale=1,actor=head.state_dict(),
                pose_source='client',coordinate_source='privileged_client',parent_sha256=digest(parent),embedding_sha256=digest(embed)),path)
            controller=object.__new__(RMNavigationController);controller.navigation_path=path
            controller.student=SimpleNamespace(torch=torch,checkpoint=parent,embed_path=embed,
                agent=SimpleNamespace(policy=SimpleNamespace(pi_head=self.parent)))
            controller.attach_navigation();features=self.x[...,:2054]
            self.assertTrue(all(torch.equal(v,controller.nav_head(features)[k]) for k,v in self.parent(features).items()))
            controller.nav_head.goal=self.x[...,-10:]
            self.assertTrue(all(torch.equal(v,controller.nav_head(features)[k]) for k,v in head(self.x).items()))

    def test_human_loader_accepts_v4_then_checks_provenance(self):
        from minecraftrl import human_navigation
        contract=dict(schema='human_navigation_reference_bc_v1',supervision='far_reference_only',minimum_distance=1.,feature_report={})
        actor=dict(schema='navigation_refinement_v4',kind='mlp',visual_adapter_width=64,root_context_mode='ignored')
        with patch.object(human_navigation,'checked_file',side_effect=RuntimeError('reached provenance check')):
            with self.assertRaisesRegex(RuntimeError,'reached provenance'):human_navigation.load_reference_dataset(Path('.'),contract,actor,None,None)
            with self.assertRaises(ValueError):human_navigation.load_reference_dataset(Path('.'),contract,actor|dict(visual_adapter_width=32),None,None)


if __name__=='__main__':unittest.main()
