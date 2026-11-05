import copy
import math
import unittest
from test_execution_contract import spec, server, queue


class ExecutionReturnTests(unittest.TestCase):
    def test_raw_server_audit_rejects_relabelled_inventory(self):
        from minecraftrl.execution_returns import check_committed_evidence
        before=server();after=server(wood=1,tick=101)
        row=dict(step=1,env_action={'attack':1},reward=1.,
            execution_before=dict(tick=100,inventory=before['server_inventory']),
            execution_after=dict(tick=101,inventory=after['server_inventory']))
        event=dict(step=1,action={'attack':1},reward=1.,info=dict(workshop=after))
        check_committed_evidence(row,before,event)
        bad=copy.deepcopy(row);bad['execution_after']['inventory']['oak_log']=2
        with self.assertRaises(ValueError):check_committed_evidence(bad,before,event)

    def test_trace_replays_outcomes_and_rejects_changed_learning_targets(self):
        import hashlib
        import struct
        from minecraftrl.execution_returns import (EXTRA_FIELDS,REWARD_PROFILE,state_record,context_tail,
            program_potential,learning_reward,validate_execution_trace)
        from minecraftrl.execution_contract import execution_request
        from minecraftrl.goal_spec import policy_condition
        q=queue(spec(quantity=2));rows=[];vector=[0.]*512
        sha=hashlib.sha256(struct.pack('<512f',*vector)).hexdigest()
        fields=['quantity_log2',*EXTRA_FIELDS]
        for i,n in enumerate((1,2)):
            before=state_record(q);phi=program_potential(q);binding=policy_condition(q,'mine')
            req=execution_request(q,'mine','collection');ctx=[math.log2(3)/math.log2(65.),*context_tail(q,vector)]
            reward=q.observe(server(wood=n,tick=101+i));q.select()
            rows.append(dict(step=i+1,goal_id='first',phase='mine',goal_key=binding['goal_key'],goal_binding=binding,
                embedding_sha256=sha,phase_context=ctx,execution_request=req,execution_before=before,
                execution_after=state_record(q),reward=reward,program_reward=float(q.completed),
                potential_before=phi,potential_after=program_potential(q),terminal=q.completed,
                learning_reward=learning_reward(phi,program_potential(q),q.completed,q.completed)))
        data=dict(rows=rows,goal_spec=q.goal_spec,program=q.program,program_sha256=q.program_sha256,
            context_fields=fields,variant='nonlinear-on',recovery_task='collection',reward_profile=REWARD_PROFILE,
            bank_metadata=dict(prompts={'root':binding['prompt']},embeddings={'root':dict(vector_sha256=sha)}))
        validate_execution_trace(data)
        for key,value in [('learning_reward',1.),('learning_reward',float('nan')),('program_reward',1.),('terminal',True)]:
            bad=copy.deepcopy(data);bad['rows'][0][key]=value
            with self.assertRaises(ValueError):validate_execution_trace(bad)
        bad=copy.deepcopy(data);bad['rows'][0]['phase_context'][-1]=.2
        with self.assertRaisesRegex(ValueError,'vector'):validate_execution_trace(bad)

    def test_root_return_crosses_intermediate_acquisition(self):
        from minecraftrl.execution_returns import program_returns
        run=dict(rows=[dict(value=0.,learning_reward=r,terminal=t,skill_terminal=True)
                       for r,t in ((0.,False),(0.,False),(1.,True))],terminal=True,bootstrap=0.)
        self.assertEqual(program_returns(run,gamma=.5)[1],[.25,.5,1.])

    def test_potential_telescopes_and_terminal_does_not_bootstrap(self):
        from minecraftrl.execution_returns import learning_reward
        a=learning_reward(0.,.5,False,False)
        b=learning_reward(.5,1.,True,True)
        self.assertAlmostEqual(a+.999*b,.999)
        self.assertAlmostEqual(learning_reward(.8,.5,False,True),-.16)
        self.assertNotEqual(learning_reward(.8,.5,False,False),learning_reward(.8,.5,False,True))

    def test_truncation_bootstraps_whole_program(self):
        from minecraftrl.execution_returns import program_returns
        run=dict(rows=[dict(value=0.,learning_reward=0.,terminal=False)],terminal=False,bootstrap=2.)
        self.assertEqual(program_returns(run,gamma=.5)[1],[1.])

    def test_explicit_target_vector_and_protected_inventory_in_context(self):
        from minecraftrl.execution_returns import context_tail, EXTRA_FIELDS
        q=queue(spec(preserve=[dict(target='dirt',quantity=2)]),server(dirt=3))
        vector=[0.]*512;vector[7]=.25
        values=context_tail(q,vector)
        self.assertEqual(values[EXTRA_FIELDS.index('goal_vector_007')],.25)
        self.assertEqual(values[EXTRA_FIELDS.index('effect_acquire')],1.)
        self.assertEqual(values[EXTRA_FIELDS.index('effect_possess')],0.)
        self.assertGreater(values[EXTRA_FIELDS.index('preserve_dirt_min')],0.)
        changed=vector.copy();changed[7]=-.25
        self.assertNotEqual(values,context_tail(q,changed))
        with self.assertRaises(ValueError):context_tail(q,[float('nan')]*512)

    def test_root_potential_tracks_whole_program_without_reset_at_handoff(self):
        from minecraftrl.execution_returns import program_potential
        s=spec(quantity=1)
        other=copy.deepcopy(s['goals'][0]);other.update(id='second',target='dirt',depends_on=['first'])
        s['goals'].append(other);q=queue(s)
        self.assertEqual(program_potential(q),0.)
        q.observe(server(wood=1,tick=101));q.select()
        self.assertEqual(program_potential(q),.5)
        q.observe(server(wood=1,dirt=1,tick=102))
        self.assertEqual(program_potential(q),1.)

    def test_future_goal_preservation_does_not_add_policy_success(self):
        from minecraftrl.execution_returns import learning_reward,program_potential
        q=queue(spec(quantity=1,preserve=[dict(target='dirt',quantity=1)]),server(dirt=1))
        before=program_potential(q)
        q.observe(server(wood=1,dirt=0,tick=101))
        self.assertEqual(learning_reward(before,program_potential(q),q.completed,True),0.)

    def test_actor_and_critic_receive_same_explicit_context(self):
        try:import torch
        except ImportError:self.skipTest('isolated WSL torch check')
        from minecraftrl.execution_learning import ExecutionHead,CONTEXT_FIELDS
        from minecraftrl.conditioned_head import ConditionedValue
        class Base(torch.nn.Module):
            def __init__(self):super().__init__();self.linear=torch.nn.Linear(5,3)
            def forward(self,x):return self.linear(x)
        base=Base();old=copy.deepcopy(base);head=ExecutionHead(base,5,'nonlinear-on')
        critic=ConditionedValue(5,'nonlinear-on',context_dim=len(CONTEXT_FIELDS))
        x=torch.randn(2,1,5+len(CONTEXT_FIELDS))
        torch.testing.assert_close(head(x),old(x[...,:5]),rtol=1e-6,atol=1e-6)
        self.assertEqual(tuple(critic(x[0:1]).shape),(1,1,1))
        opt=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.01)
        head(x).square().mean().backward();opt.step()
        altered=x.clone();altered[...,-512:]=0
        self.assertGreater(float((head(x)-head(altered)).abs().max()),1e-5)
        self.assertTrue(all(not p.requires_grad for p in head.base.parameters()))

    def test_real_environment_adapter_handles_handoff_and_instant_successor(self):
        try:import torch
        except ImportError:self.skipTest('isolated WSL runtime check')
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from minecraftrl.execution_learning import ExecutionPolicy,ExecutionEnvironment
        from minecraftrl.execution_contract import BoundaryUpdates,execution_request
        from minecraftrl.execution_returns import state_record,program_potential
        from minecraftrl.goal_queue import GoalQueue
        from minecraftrl.rm_program import CAPABILITIES
        from minecraftrl.semantic_goal import ResultMonitor
        from minecraftrl.goal_spec import expected_prompt
        from minecraftrl.rm_conditioning import ConditioningPolicy
        from minecraftrl.policies import Steve1Policy
        import test_rm_program as pf
        import test_rm_conditioning as cf
        from test_rm_logo_conditioning import FakeGame
        for immediate in (False,True):
            with self.subTest(immediate=immediate),tempfile.TemporaryDirectory() as folder:
                s=spec(quantity=1)
                other=copy.deepcopy(s['goals'][0]);other.update(id='second',target='dirt',depends_on=['first'],
                    effect='possess' if immediate else 'acquire')
                s['goals'].append(other)
                f=pf.ProgramTests();baseline=f.baseline();baseline['server_inventory']['dirt']=int(immediate)
                states=[baseline]
                for logs,dirt in ((1,int(immediate)),(1,1)):
                    state=copy.deepcopy(states[-1]);state['server_tick']+=1
                    state['server_inventory'].update(oak_log=logs,dirt=dirt);states.append(state)
                p=f.policy([f.wood()]);p.__class__=ExecutionPolicy;p.queue=GoalQueue(s,CAPABILITIES)
                p.boundaries=BoundaryUpdates();p.last_boundary_updates=[]
                p.task='collection';p.split='development';p.variant='nonlinear-on'
                p.monitor=ResultMonitor();p.monitor_events=[];p.learning_rows=[];p.action_budget=400
                p.failure=p.end_reason=None;p.output=Path(folder)
                p.prompts={k:expected_prompt(k) for k in p.prompts}
                game=FakeGame(states);game.recovery_world=dict(sha256='test',experimental_split='development')
                env=ExecutionEnvironment(game,p);env.reset()
                with patch.object(Steve1Policy,'act',cf.RMConditioningTests.infer):
                    while not p.queue.completed:
                        p.learning_pending=dict(execution_request=execution_request(p.queue,p.selector.current,p.task),
                            execution_before=state_record(p.queue),potential_before=program_potential(p.queue))
                        env.step(ConditioningPolicy.act(p,None))
                self.assertEqual(len(p.learning_rows),1 if immediate else 2)
                self.assertEqual(sum(r['program_reward'] for r in p.learning_rows),1.)
                self.assertTrue(p.learning_rows[-1]['terminal'])
                self.assertEqual(p.learning_rows[0]['execution_request']['goal_id'],'first')
                self.assertTrue(env.terminal)


if __name__=='__main__':unittest.main()
