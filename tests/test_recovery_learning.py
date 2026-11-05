import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import test_rm_program


class RecoveryContextTests(unittest.TestCase):
    def state(self, skill):
        from minecraftrl.goal_queue import GoalQueue
        from minecraftrl.rm_program import CAPABILITIES
        f=test_rm_program.ProgramTests(); goal=f.wood()
        goal.update(skill=skill,items=list(CAPABILITIES[skill]))
        q=GoalQueue(dict(schema='goal_queue_v1',goals=[goal]),CAPABILITIES)
        q.observe(f.baseline());q.select()
        return q

    def test_unknown_dirt_is_not_confirmed_interaction(self):
        from minecraftrl.recovery_learning import recovery_context, CONTEXT_FIELDS
        values,desc=recovery_context(self.state('dirt_fixed'),'collect',None,{},task='collection')
        row=dict(zip(CONTEXT_FIELDS,values))
        self.assertEqual(sum(values[:4]),0)
        self.assertEqual(row['phase_valid'],0)
        self.assertEqual(row['pending_valid'],0)
        self.assertEqual(desc['stage'],'unknown')

    def test_pickup_intent_never_claims_entity_location(self):
        from minecraftrl.recovery_learning import recovery_context, CONTEXT_FIELDS
        values,desc=recovery_context(self.state('wood_memory'),'collect',None,
            dict(target_known=True,target_reachable=True),task='pickup')
        row=dict(zip(CONTEXT_FIELDS,values))
        self.assertEqual(row['recover_result'],1)
        self.assertEqual(row['phase_valid'],1)
        self.assertEqual(row['pending_valid'],0)
        self.assertEqual(row['known_valid'],0)
        self.assertEqual(row['reachable_valid'],0)
        self.assertEqual(desc['stage_source'],'requested_skill')

    def test_harvest_evidence_is_not_drop_location(self):
        from minecraftrl.recovery_learning import recovery_context, CONTEXT_FIELDS
        values,_=recovery_context(self.state('wood_memory'),'collect',
            dict(pending_result_since=99,last_progress_tick=99),
            dict(target_known=True,target_reachable=True),task='collection')
        row=dict(zip(CONTEXT_FIELDS,values))
        self.assertEqual(row['pending_result'],1)
        self.assertEqual(row['pending_valid'],1)
        self.assertEqual(row['known_valid'],0)

    def test_cross_mode_and_external_world_rejected(self):
        from minecraftrl.recovery_learning import validate_args
        root=Path(__file__).resolve().parents[1]
        args=SimpleNamespace(root=root,recovery_variant='nonlinear-on',recovery_world=Path(__file__),
            recovery_prepare=False,rm_conditioning='goal-queue',s2_variant='linear-on')
        with self.assertRaises(ValueError): validate_args(args)
        args.s2_variant=None
        with self.assertRaisesRegex(ValueError,'project-local'):validate_args(args)
        args.recovery_variant=None;args.recovery_world=None;args.recovery_sampling_seed=123
        with self.assertRaisesRegex(ValueError,'explicit recovery'):validate_args(args)


class RecoveryHeadTests(unittest.TestCase):
    def test_replay_admission_rejects_old_candidate_and_development(self):
        import sys
        import torch
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from train_recovery_adapter import validate
        from minecraftrl.recovery_learning import SCOPE,CONTEXT_FIELDS
        from minecraftrl.rl_policy import state_fingerprint
        dim=len(CONTEXT_FIELDS);heads={'actor':{'w':torch.zeros(1)},'value':{'w':torch.zeros(1)}}
        row=dict(step=1,terminal=True,reward=1.,value=0.,old_logp=-1.,
            phase_context=[0.]*dim,features=torch.zeros(2,1,5+dim))
        data=dict(scope=SCOPE,split='train',eligible=True,checkpoint_sha256=None,status='target_reached',
            variant='nonlinear-on',context_fields=list(CONTEXT_FIELDS),recurrent_commit='dispatch_only',
            initial_heads=heads,initial_heads_sha256=state_fingerprint(heads),rows=[row],feature_width=5+dim,
            terminal=True,bootstrap=0.)
        summary=dict(status='target_reached',execution_valid=True,worker_exit_code=0,steps=1)
        validate(data,summary,'nonlinear-off',None)
        for changes in (dict(split='development'),dict(status='cancelled'),dict(checkpoint_sha256='wrong')):
            with self.assertRaises(ValueError):validate(dict(data,**changes),summary,'nonlinear-on',None)
        with self.assertRaises(ValueError):validate(dict(data,checkpoint_sha256='parent'),summary,'nonlinear-off','parent')
        broken=copy.deepcopy(data);broken['rows'][0]['features'][0,0,-1]=1
        with self.assertRaises(ValueError):validate(broken,summary,'nonlinear-on',None)

    def test_frozen_motor_head_and_off_invariance(self):
        import torch
        from minecraftrl.recovery_learning import RecoveryHead, CONTEXT_FIELDS
        torch.set_num_threads(2)
        class Base(torch.nn.Module):
            def __init__(self):super().__init__();self.linear=torch.nn.Linear(5,3)
            def forward(self,x):return self.linear(x)
        base=Base(); expected=copy.deepcopy(base)
        head=RecoveryHead(base,5,'nonlinear-off')
        x=torch.randn(2,1,5+len(CONTEXT_FIELDS))
        # Wider GEMM includes zero context columns; allow float32 summation noise.
        torch.testing.assert_close(head(x),expected(x[...,:5]),rtol=1e-6,atol=1e-6)
        before={k:v.clone() for k,v in head.base.state_dict().items()}
        optimizer=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.01)
        head(x).square().sum().backward();optimizer.step()
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in head.base.state_dict().items()))
        y=x.clone();y[...,5:]=torch.randn_like(y[...,5:])
        torch.testing.assert_close(head(x),head(y),rtol=0,atol=0)
        self.assertFalse(torch.equal(head(x),expected(x[...,:5])))


class RecoveryEnvironmentTests(unittest.TestCase):
    def test_real_queue_pickup_completion_stops_and_does_not_require_harvest(self):
        import tempfile
        from unittest.mock import patch
        from minecraftrl.recovery_learning import RecoveryPolicy,RecoveryEnvironment
        from minecraftrl.semantic_goal import ResultMonitor
        from minecraftrl.rm_conditioning import ConditioningPolicy
        from minecraftrl.policies import Steve1Policy
        from test_rm_logo_conditioning import FakeGame
        import test_rm_conditioning as fixtures
        f=test_rm_program.ProgramTests();b=f.baseline();after=copy.deepcopy(b)
        after['server_tick']+=1;after['server_inventory']['oak_log']=1
        p=f.policy([f.wood()]);p.__class__=RecoveryPolicy
        p.task='pickup';p.variant='nonlinear-on';p.split='development'
        p.monitor=ResultMonitor();p.monitor_events=[];p.learning_rows=[];p.action_budget=400
        p.failure=p.end_reason=None;p.learning_pending={}
        p.prompts['pickup_wood']='pick up wood';p.embeddings['pickup_wood']=fixtures.FakeEmbedding('pickup_wood')
        p.bank_metadata['embeddings']['pickup_wood']=dict(vector_sha256='pickup_wood')
        game=FakeGame([b,after]);game.recovery_world=dict(sha256='fixture',experimental_split='development')
        with tempfile.TemporaryDirectory() as directory:
            p.output=Path(directory);env=RecoveryEnvironment(game,p);env.reset()
            self.assertIsNone(p.progress)
            with patch.object(Steve1Policy,'act',fixtures.RMConditioningTests.infer):
                action=ConditioningPolicy.act(p,None);obs,reward,done,_=env.step(action)
            self.assertEqual(reward,1.)
            self.assertTrue(env.terminal);self.assertTrue(p.queue.completed)
            self.assertTrue(p.learning_rows[-1]['terminal'])
            self.assertEqual(after['server_log_harvests'],b['server_log_harvests'])
            with self.assertRaises(RuntimeError):env.step(action)
            with self.assertRaises(InterruptedError):p.act(None)


if __name__=='__main__':unittest.main()
