import copy,unittest
import test_navigation_conditioning as conditioning
try:
    import torch
    from minecraftrl import navigation_model as staged
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Torch unavailable')
class NavigationRootContextTests(unittest.TestCase):
    def setUp(self):
        conditioning.NavigationConditioningTests.setUp(self);self.model=staged

    head=conditioning.NavigationConditioningTests.head

    def test_legacy_function_is_bitwise_identical(self):
        for kind in ('linear','mlp'):
            head=self.head(kind);staged.rescale_distance_input(head,16)
            visual,goal=self.x[...,:2054],self.x[...,2054:]
            conditioned=torch.cat([goal[...,:2],goal[...,2:3]*16,goal[...,3:]],-1)
            actual=head(self.x)
            for key in ('buttons','camera'):
                module=getattr(head,key)
                inputs=conditioned if kind=='linear' else torch.cat([module.visual_projection(visual),conditioned],-1)
                expected=module.base(torch.cat([visual,goal[...,:6]],-1))+module.residual(inputs)
                self.assertTrue(torch.equal(expected,actual[key]))

    def test_ignored_rm_invariance_and_missing_context_equivalence(self):
        for kind in ('linear','mlp'):
            head=self.head(kind);staged.set_root_context_mode(head,'ignored')
            changed=self.x.clone();changed[...,2048:2054]=1000*torch.randn_like(changed[...,2048:2054])
            missing=torch.cat([self.x[...,:2048],self.x[...,2054:]],-1)
            for key,value in head(self.x).items():
                self.assertTrue(torch.equal(value,head(changed)[key]))
                self.assertTrue(torch.equal(value,head(missing)[key]))
            # Both input and omitted parameter columns receive no RM gradient.
            head.requires_grad_(True);x=self.x.clone().requires_grad_(True)
            sum(v.sum() for v in head(x).values()).backward()
            self.assertEqual(x.grad[...,2048:2054].abs().max().item(),0)
            for module in head.modules():
                if isinstance(module,staged.RefinedLinear):
                    self.assertEqual(module.base.visual.weight.grad[:,2048:].abs().max().item(),0)
                    if kind=='mlp':self.assertEqual(module.visual_projection[0].weight.grad[:,2048:].abs().max().item(),0)

    def test_root_switch_path_is_unmodified(self):
        head=self.head('mlp');staged.set_root_context_mode(head,'ignored')
        switch=staged.NavigationSwitch(self.parent,head,True);features=self.x[...,:2054]
        expected=self.parent(features)
        for goal in (None,torch.zeros(4,8,10),None):
            switch.goal=goal;actual=switch(features)
            if goal is None:
                for key,value in expected.items():self.assertTrue(torch.equal(value,actual[key]))
        self.assertTrue(all(torch.equal(v,self.parent.state_dict()[k]) for k,v in switch.root.state_dict().items()))

    def test_versioned_reload_and_invalid_metadata(self):
        head=self.head('mlp');staged.rescale_distance_input(head,16);staged.set_root_context_mode(head,'ignored')
        saved=dict(schema='navigation_refinement_v3',root_context_mode='ignored',residual_distance_scale=16,
                   kind='mlp',actor=head.state_dict())
        restored,_=staged.load_navigation(self.parent,saved)
        for key,value in head(self.x).items():self.assertTrue(torch.equal(value,restored(self.x)[key]))
        for key in ('root_context_mode','residual_distance_scale'):
            invalid=dict(saved);del invalid[key]
            with self.assertRaises(ValueError):staged.load_navigation(self.parent,invalid)
        for schema in ('waypoint_actor_v1','navigation_refinement_v1','navigation_refinement_v2'):
            with self.assertRaises(ValueError):staged.load_navigation(self.parent,dict(saved,schema=schema))
        for mode in ('unknown','',None,False,0):
            with self.assertRaises(ValueError):staged.load_navigation(self.parent,dict(saved,root_context_mode=mode))

    def test_invalid_conversion_and_input_fail(self):
        head=self.head('mlp');before=copy.deepcopy(head.state_dict())
        with self.assertRaises(ValueError):staged.set_root_context_mode(head,'bad')
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in before.items()))
        short=torch.cat([self.x[...,:2048],self.x[...,2054:]],-1)
        with self.assertRaises(ValueError):head(short)
        staged.set_root_context_mode(head,'ignored')
        with self.assertRaises(ValueError):head(torch.randn(1,1,2057))


if __name__=='__main__':unittest.main()
