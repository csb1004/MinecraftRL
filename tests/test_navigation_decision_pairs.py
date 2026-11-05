import copy
import math
import random
import unittest
import torch
from minecraftrl.navigation_decision_pairs import annotate_views, decision_views, DecisionPairSampler, DecisionPairBC, pair_rows


def fixture(views=3):
    rows=[]
    # Aliases 0/1 neither, 2/3 forward, 4/5 forward+jump.
    bits=torch.tensor([[0,0],[0,0],[0,1],[0,1],[1,1],[1,1]],dtype=torch.bool)
    for view in range(views):
        features=torch.full((2,1,4),float(view))
        subset=[]
        for i,action in enumerate([0,1,2,3,4,5]):
            goal=[0.,1.,(.1+i)/16,0.,0.,1.,0.,0.,0.,.4]
            subset.append(dict(features=features,motion_goal=goal,
                bc_source='procedural_step',bc_step_goal_query=i>0,
                bc_action={'buttons':torch.tensor([[[action]]])}))
        annotate_views(subset,[{'row':view}]*6,'results/train/run',split='train')
        rows.extend(subset)
    return rows,bits


class DecisionPairTests(unittest.TestCase):
    def test_same_view_pairs_and_full_coverage_preserve_global_random_state(self):
        rows,bits=fixture(7);pools,receipt=decision_views(rows,bits)
        self.assertEqual(receipt['views'],{'jump':7,'forward':7})
        before=torch.get_rng_state();python_before=random.getstate()
        sampler=DecisionPairSampler(pools);first=[sampler.draw() for _ in range(7)]
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(python_before,random.getstate())
        for offset,button in [(0,0),(2,1)]:
            self.assertEqual(len({rows[r[offset]]['bc_step_view'] for r in first}),7)
            for selected in first:
                a,b=(rows[i] for i in selected[offset:offset+2])
                self.assertEqual(a['bc_step_view'],b['bc_step_view'])
                self.assertTrue(bits[a['bc_action']['buttons'].item(),button])
                self.assertFalse(bits[b['bc_action']['buttons'].item(),button])
        again=DecisionPairSampler(pools)
        self.assertEqual(first,[again.draw() for _ in range(7)])
        self.assertEqual(sampler.report()['draws'],7)

    def test_nearest_first_and_goal_combinations_cycle(self):
        rows,bits=fixture(1);pools,_=decision_views(rows,bits);s=DecisionPairSampler(pools)
        draws=[s.draw() for _ in range(8)]
        self.assertEqual(draws[0],[4,0,2,0])
        self.assertEqual(len({tuple(r[:2]) for r in draws}),8)
        self.assertEqual(len({tuple(r[2:]) for r in draws}),8)

    def test_pair_budget_only_changes_shallow_supervision_copy(self):
        rows,bits=fixture(1);original=copy.deepcopy(rows)
        pairs=pair_rows(rows,[4,0,2,0])
        self.assertTrue(all(r['bc_motion_goal'][-1]==1 for r in pairs))
        self.assertTrue(all(r['motion_goal'][-1]==.4 for r in rows))
        self.assertNotIn('bc_motion_goal',rows[4])
        self.assertIs(pairs[0]['features'],rows[4]['features'])
        self.assertTrue(torch.equal(pairs[0]['bc_action']['buttons'],original[4]['bc_action']['buttons']))

    def test_reject_nontrain_missing_labels_or_mixed_views(self):
        rows,bits=fixture(1)
        with self.assertRaises(ValueError):annotate_views(rows,[{'row':0}]*6,'dev',split='development')
        bad=copy.deepcopy(rows);bad[0]['bc_step_split']='development'
        with self.assertRaises(ValueError):decision_views(bad,bits)
        with self.assertRaises(ValueError):decision_views(rows[:5],bits)
        bad=list(rows);bad[0]=dict(rows[0],features=rows[0]['features'].clone())
        with self.assertRaises(ValueError):decision_views(bad,bits)
        bad=list(rows);bad[0]=dict(rows[0],bc_action={'buttons':torch.tensor([[[9]]])})
        with self.assertRaises(ValueError):decision_views(bad,bits)

    def test_alias_summed_probability_and_gradient(self):
        _,bits=fixture(1);loss=DecisionPairBC(bits)
        x=torch.tensor([[.1,.2,.3,.4,.5,.6]]*4,dtype=torch.float64,requires_grad=True)
        y=x.detach().clone().requires_grad_();p=y.softmax(-1)
        wanted=-(p[0,4:].sum().log()+p[1,:4].sum().log()+p[2,2:].sum().log()+p[3,:2].sum().log())/4
        actual=loss(x);self.assertAlmostEqual(actual.item(),wanted.item(),places=12)
        actual.backward();wanted.backward();self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-12,rtol=0))
        self.assertLess(x.grad[0,4].item(),0);self.assertGreater(x.grad[1,4].item(),0)

    def test_extreme_logits_finite_and_shapes_checked(self):
        _,bits=fixture(1);loss=DecisionPairBC(bits)
        logits=torch.tensor([[1000.,-1000.]*3]*4,requires_grad=True)
        value=loss(logits);value.backward();self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(logits.grad).all())
        for x in [torch.zeros(3,6),torch.zeros(4,5),torch.full((4,6),float('nan'))]:
            with self.assertRaises(ValueError):loss(x)
        with self.assertRaises(ValueError):DecisionPairBC(torch.zeros(6,2))

    def test_ranking_alias_probability_formula_and_gradient(self):
        _,bits=fixture(1);loss=DecisionPairBC(bits,'conditional-ranking')
        x=torch.tensor([[.1,.2,.3,.4,.5,.6],[.6,.5,.4,.3,.2,.1],
                        [-.2,.1,.4,.5,.9,.2],[.1,.2,-.3,.4,.5,.8]],dtype=torch.float64,requires_grad=True)
        y=x.detach().clone().requires_grad_();p=y.softmax(-1);terms=[]
        for column in range(2):
            marginal=p[2*column:2*column+2,bits[:,column]].sum(-1)
            odds=marginal/(1-marginal)
            terms.append(torch.log1p(odds[1]/odds[0])/2)
        wanted=torch.stack(terms).mean();actual=loss(x)
        self.assertAlmostEqual(actual.item(),wanted.item(),places=12)
        actual.backward();wanted.backward()
        self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-12,rtol=0))
        self.assertLess(x.grad[0,4].item(),0);self.assertGreater(x.grad[1,4].item(),0)

    def test_ranking_pair_common_logodds_shift_invariance(self):
        _,bits=fixture(1);loss=DecisionPairBC(bits,'conditional-ranking')
        x=torch.arange(24,dtype=torch.float64).reshape(4,6).sin().requires_grad_()
        y=x.detach().clone()
        # Shift each decision's two rows only; joint-button cross-effects are not invariant.
        y[:2,bits[:,0]]+=7.;y[2:,bits[:,1]]-=3.
        y.requires_grad_();a=loss(x);b=loss(y)
        self.assertAlmostEqual(a.item(),b.item(),places=12)
        a.backward();b.backward();self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-12,rtol=0))

    def test_ranking_goal_order_and_neutral_gradient_normalization(self):
        bits=torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.bool)
        ranking=DecisionPairBC(bits,'conditional-ranking');bce=DecisionPairBC(bits)
        x=torch.zeros(4,4,dtype=torch.float64,requires_grad=True)
        y=x.detach().clone().requires_grad_();ranking(x).backward();bce(y).backward()
        self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-12,rtol=0))
        good=torch.zeros_like(x)
        for column in range(2):
            good[2*column,bits[:,column]]=2.;good[2*column+1,bits[:,column]]=-2.
        self.assertLess(ranking(good).item(),ranking(torch.zeros_like(x)).item())
        self.assertGreater(ranking(good[[1,0,3,2]]).item(),ranking(torch.zeros_like(x)).item())

    def test_ranking_extremes_and_invalid_objective(self):
        _,bits=fixture(1);loss=DecisionPairBC(bits,'conditional-ranking')
        x=torch.tensor([[1000.,-1000.]*3]*4,requires_grad=True)
        value=loss(x);value.backward()
        self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(x.grad).all())
        for bad in [torch.zeros(3,6),torch.zeros(4,5),torch.full((4,6),float('inf'))]:
            with self.assertRaises(ValueError):loss(bad)
        with self.assertRaises(ValueError):DecisionPairBC(bits,'unknown')


if __name__=='__main__':unittest.main()
