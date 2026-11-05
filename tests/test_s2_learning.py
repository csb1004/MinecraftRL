import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'isolated tensor environment required')
class S2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def head(self, variant):
        from minecraftrl.conditioned_head import ConditionedHead
        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(5,3)
            def forward(self,x):return self.linear(x)
        return ConditionedHead(Base(),5,variant)

    def test_all_initial_variants_match_and_off_is_invariant_after_update(self):
        from minecraftrl.conditioned_head import VARIANTS
        from minecraftrl.semantic_goal import CONTEXT_DIM
        x=torch.randn(2,1,5+CONTEXT_DIM)
        initial=self.head('linear-on').base.state_dict()
        expected=None
        for variant in VARIANTS:
            h=self.head(variant);h.base.load_state_dict(initial)
            y=h(x)
            if expected is None:expected=y.detach()
            torch.testing.assert_close(y,expected,rtol=0,atol=0)
            y.square().sum().backward()
            with torch.no_grad():
                for p in h.parameters():
                    if p.grad is not None:p.add_(p.grad,alpha=-.01)
            if variant.endswith('-off'):
                changed=x.clone();changed[...,5:]=torch.randn_like(changed[...,5:])
                torch.testing.assert_close(h(x),h(changed),rtol=0,atol=0)

    def test_nonlinear_adapter_can_change_visual_context_interaction(self):
        from minecraftrl.semantic_goal import CONTEXT_DIM
        h=self.head('nonlinear-on')
        with torch.no_grad(): h.adapter[-1].weight.normal_(0,.1)
        x=torch.randn(2,1,5+CONTEXT_DIM);x[1]=x[0];x[1,:,:5]+=2
        other=x.clone();other[:,:,5:]+=1
        delta=h(other)-h(x)
        self.assertGreater((delta[0]-delta[1]).abs().max().item(),1e-5)

    def policy_env(self, action_budget=1200, stalled=False):
        from test_rm_program import ProgramTests
        from test_rm_logo_conditioning import FakeGame
        from minecraftrl.rm_program import ProgramPolicy
        from minecraftrl.s2_learning import SemanticEnvironment
        from minecraftrl.semantic_goal import ResultMonitor
        f=ProgramTests();p=f.policy([f.wood()]);b=f.baseline()
        next_state=dict(copy.deepcopy(b),server_tick=700 if stalled else 101)
        class Recording(ProgramPolicy):
            def commit(self,*args):
                super().commit(*args);self.learning_rows.append(dict(self.rows[-1]))
        p.__class__=Recording
        p.monitor=ResultMonitor();p.monitor_events=[];p.learning_rows=[]
        p.action_budget=action_budget;p.failure=None;p.end_reason=None
        env=SemanticEnvironment(FakeGame([b,next_state]),p);env.reset()
        return p,env

    def test_budget_truncation_keeps_bootstrap_boundary(self):
        from minecraftrl.policies import Steve1Policy
        from test_rm_conditioning import RMConditioningTests
        p,env=self.policy_env(action_budget=1)
        with patch.object(Steve1Policy,'act',RMConditioningTests.infer):
            _,_,done,info=env.step(p.act(None))
        self.assertTrue(done);self.assertFalse(env.terminal)
        self.assertFalse(p.learning_rows[-1]['terminal'])
        self.assertEqual(info['s2_end_reason'],'action_budget')

    def test_stagnation_is_finite_task_failure_and_keeps_failure_evidence(self):
        from minecraftrl.policies import Steve1Policy
        from test_rm_conditioning import RMConditioningTests
        p,env=self.policy_env(stalled=True)
        with patch.object(Steve1Policy,'act',RMConditioningTests.infer):
            _,reward,done,info=env.step(p.act(None))
        self.assertTrue(done);self.assertTrue(env.terminal);self.assertEqual(reward,0)
        self.assertTrue(p.learning_rows[-1]['terminal'])
        self.assertFalse(p.failure['executed'])
        self.assertEqual(p.queue.records['wood']['reason'],'no_confirmed_progress')
        with self.assertRaises(RuntimeError):env.step({'attack':[1]})

    def test_stop_before_inference(self):
        from minecraftrl.s2_learning import SemanticPolicy
        p=SemanticPolicy.__new__(SemanticPolicy);p.is_stopped=lambda:True
        with self.assertRaises(InterruptedError):p.act(None)

    def test_checkpoint_outside_results_is_rejected_before_model_loading(self):
        from types import SimpleNamespace
        from minecraftrl.rm_conditioning import validate_args
        args=SimpleNamespace(root=Path(__file__).resolve().parents[1],s2_variant='linear-on',
            s2_checkpoint=Path(__file__).resolve(),rm_conditioning='goal-queue')
        with self.assertRaisesRegex(ValueError,'project-local S2'):validate_args(args)

    def test_trainer_rejects_evaluation_cancellation_and_disconnected_context(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from train_s2_conditions import validate
        from minecraftrl.s2_learning import SCOPE
        from minecraftrl.semantic_goal import CONTEXT_DIM, CONTEXT_FIELDS
        from minecraftrl.rl_policy import state_fingerprint
        heads={'actor':{'base.weight':torch.zeros(2,5+CONTEXT_DIM)},'value':{'weight':torch.zeros(1)}}
        row=dict(step=1,terminal=True,phase_context=[0.]*CONTEXT_DIM,phase='collect',
                 features=torch.zeros(2,1,5+CONTEXT_DIM),old_logp=-1.,value=0.,reward=1.)
        data=dict(scope=SCOPE,split='train',eligible=True,variant='linear-on',checkpoint_sha256=None,
            status='target_reached',rows=[row],context_fields=list(CONTEXT_FIELDS),context_dim=CONTEXT_DIM,
            feature_width=5+CONTEXT_DIM,recurrent_commit='dispatch_only',initial_heads=heads,
            initial_heads_sha256=state_fingerprint(heads),terminal=True,bootstrap=0.)
        summary=dict(execution_valid=True,worker_exit_code=0,status='target_reached',steps=1)
        validate(data,summary)
        for changes in (dict(split='evaluation'),dict(status='cancelled'),dict(checkpoint_sha256='old')):
            with self.assertRaises(ValueError):validate(dict(data,**changes),summary)
        broken=copy.deepcopy(data);broken['rows'][0]['features'][0,0,-1]=1
        with self.assertRaises(ValueError):validate(broken,summary)


if __name__=='__main__':unittest.main()
