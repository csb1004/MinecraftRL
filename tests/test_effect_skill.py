import copy
import unittest

import test_goal_spec as gf
specification = gf.specification


class EffectTests(unittest.TestCase):
    def setup_state(self, quantity=3):
        from minecraftrl.effect_skill import EffectLedger
        q, server = gf.GoalSpecTests().queue(specification(quantity))
        return q, server, EffectLedger()

    def advance(self, q, server, ledger, count):
        request = ledger.begin(q)
        server = copy.deepcopy(server)
        server['server_tick'] += 1
        server['server_inventory']['oak_log'] = count
        reward = q.observe(server)
        outcome = ledger.commit(request, server, terminal=q.completed)
        return server, reward, outcome

    def test_partial_credit_preserves_root_requirement(self):
        q, s, ledger = self.setup_state()
        s, reward, result = self.advance(q, s, ledger, 9)
        self.assertEqual(reward, 0)
        self.assertEqual(result['skill_reward'], 1)
        self.assertTrue(result['skill_terminal'])
        self.assertFalse(q.completed)
        s, reward, result = self.advance(q, s, ledger, 10)
        self.assertEqual(reward, 0)
        self.assertEqual(result['skill_reward'], 1)
        s, reward, result = self.advance(q, s, ledger, 11)
        self.assertEqual(reward, 1)
        self.assertTrue(q.completed)

    def test_drop_repick_has_no_repeated_credit(self):
        q, s, ledger = self.setup_state()
        rewards = []
        for count in (9, 8, 9, 8, 9, 10):
            s, _, result = self.advance(q, s, ledger, count)
            rewards.append(result['skill_reward'])
        self.assertEqual(rewards, [1, 0, 0, 0, 0, 1])

    def test_multi_item_event_is_one_skill_and_advances_peak(self):
        q, s, ledger = self.setup_state(5)
        s, _, result = self.advance(q, s, ledger, 11)
        self.assertEqual(result['skill_reward'], 1)
        request = ledger.begin(q)
        self.assertEqual(request['credited_peak'], 11)
        self.assertEqual(request['threshold'], 12)
        self.assertEqual(request['quantity'], 5)

    def test_stale_request_duplicate_commit_and_revision_rejected(self):
        q, s, ledger = self.setup_state()
        request = ledger.begin(q)
        q.reprioritize(q.active, 60, 1)
        with self.assertRaisesRegex(ValueError, 'revision'):
            ledger.begin(q)
        q.records[q.active]['revision'] = 1
        s, _, _ = self.advance(q, s, ledger, 9)
        with self.assertRaises(ValueError):
            ledger.commit(request, s, terminal=False)

    def test_uncommitted_request_does_not_advance_peak(self):
        q, _, ledger = self.setup_state()
        before = ledger.begin(q)
        self.assertEqual(ledger.begin(q), before)
        self.assertEqual(before['credited_peak'], 8)

    def test_server_numeric_telemetry_normalizes_integer_ticks(self):
        q,s,ledger=self.setup_state()
        request=ledger.begin(q)
        s['server_tick']=float(s['server_tick']+1)
        s['server_inventory']['oak_log']=9.0
        out=ledger.commit(request,s,terminal=False)
        self.assertIs(type(out['after_tick']),int)
        self.assertIs(type(out['after_count']),int)

    def test_training_recomputes_credit_and_condition_from_evidence(self):
        from minecraftrl.effect_skill import validate_effect_trace, effect_context, EXTRA_FIELDS
        q, s, ledger = self.setup_state()
        rows=[]
        for count in (9,8,9,10,11):
            request=ledger.begin(q)
            ctx=effect_context(request, ledger.last_count)
            s, reward, outcome=self.advance(q,s,ledger,count)
            rows.append(dict(effect_request=request,effect_outcome=outcome,goal_id='first',
                reward=reward,terminal=q.completed,skill_reward=outcome['skill_reward'],
                skill_terminal=outcome['skill_terminal'],phase_context=ctx))
        data=dict(rows=rows,goal_spec=q.goal_spec,context_fields=list(EXTRA_FIELDS))
        validate_effect_trace(data)
        for key, value in (('skill_reward',1.), ('skill_terminal',True), ('reward',1.)):
            bad=copy.deepcopy(data);bad['rows'][1][key]=value
            with self.assertRaises(ValueError):validate_effect_trace(bad)
        bad=copy.deepcopy(data);bad['rows'][1]['effect_request']['threshold']=1
        with self.assertRaises(ValueError):validate_effect_trace(bad)
        bad=copy.deepcopy(data);bad['rows'][1]['phase_context'][0]=.5
        with self.assertRaises(ValueError):validate_effect_trace(bad)

    def test_actor_zero_initialization_and_request_condition_training(self):
        import torch
        from minecraftrl.effect_learning import EffectHead, CONTEXT_FIELDS
        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.linear=torch.nn.Linear(5,3)
            def forward(self,x):return self.linear(x)
        torch.set_num_threads(2)
        base=Base();original=copy.deepcopy(base)
        rng=torch.get_rng_state().clone()
        head=EffectHead(base,5,'nonlinear-on')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        x=torch.randn(2,1,5+len(CONTEXT_FIELDS)); y=x.clone(); y[...,5:]=0
        torch.testing.assert_close(head(x),original(x[...,:5]),rtol=1e-6,atol=1e-6)
        frozen={k:v.clone() for k,v in head.base.state_dict().items()}
        optimizer=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.01)
        head(x).square().sum().backward();optimizer.step()
        self.assertTrue(all(torch.equal(v,frozen[k]) for k,v in head.base.state_dict().items()))
        self.assertGreater(float((head(x)-head(y)).abs().max()),1e-5)

    def test_segmented_returns_do_not_leak_next_skill_rewards(self):
        from minecraftrl.effect_skill import skill_returns
        rows = [dict(value=0., skill_reward=r, skill_terminal=t) for r,t in
                ((0., False), (1., True), (0., False), (1., True))]
        a, returns = skill_returns(dict(rows=rows, terminal=True, bootstrap=0.), gamma=.5)
        self.assertEqual(returns, [.5, 1., .5, 1.])
        self.assertEqual(a, returns)
        rows[-1].update(skill_reward=0., skill_terminal=False)
        _, returns = skill_returns(dict(rows=rows, terminal=False, bootstrap=2.), gamma=.5)
        self.assertEqual(returns, [.5, 1., .5, 1.])

    def test_effect_runtime_uses_committed_old_goal_on_root_switch(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from minecraftrl.effect_learning import EffectPolicy, EffectEnvironment
        from minecraftrl.effect_skill import EffectLedger
        from minecraftrl.goal_queue import GoalQueue
        from minecraftrl.rm_program import CAPABILITIES
        from minecraftrl.semantic_goal import ResultMonitor
        from minecraftrl.rm_conditioning import ConditioningPolicy
        from minecraftrl.policies import Steve1Policy
        import test_rm_program as pf
        import test_rm_conditioning as cf
        from test_rm_logo_conditioning import FakeGame
        source = specification(2)
        source['goals'].append(dict(id='second', target='dirt', quantity=1, priority=50,
                                    depends_on=['first'], requires=[]))
        f = pf.ProgramTests(); baseline = f.baseline()
        states = [baseline]
        for logs, dirt in ((1, 0), (2, 0), (2, 1)):
            state = copy.deepcopy(states[-1]); state['server_tick'] += 1
            state['server_inventory'].update(oak_log=logs, dirt=dirt); states.append(state)
        p = f.policy([f.wood()]); p.__class__ = EffectPolicy
        p.queue = GoalQueue(source, CAPABILITIES)
        p.effects = EffectLedger(); p.task='collection'; p.split='development'; p.variant='nonlinear-on'
        p.monitor=ResultMonitor(); p.monitor_events=[]; p.learning_rows=[]; p.action_budget=400
        p.failure=p.end_reason=None
        # Stub only motor inference; exercise real queue/environment/commit paths.
        from minecraftrl.goal_spec import expected_prompt
        p.prompts={k:expected_prompt(k) for k in p.prompts}
        game = FakeGame(states)
        game.recovery_world=dict(sha256='test', experimental_split='development')
        with tempfile.TemporaryDirectory() as directory:
            p.output=Path(directory); env=EffectEnvironment(game, p); env.reset()
            with patch.object(Steve1Policy,'act',cf.RMConditioningTests.infer):
                for _ in range(3):
                    p.current_request=p.effects.begin(p.queue)
                    p.learning_pending={'effect_request':copy.deepcopy(p.current_request)}
                    env.step(ConditioningPolicy.act(p, None))
            rows=p.learning_rows
            self.assertEqual([r['reward'] for r in rows], [0., 1., 1.])
            self.assertEqual([r['skill_reward'] for r in rows], [1., 1., 1.])
            self.assertEqual([r['terminal'] for r in rows], [False, False, True])
            self.assertEqual([r['effect_request']['goal_id'] for r in rows], ['first','first','second'])
            self.assertTrue(p.queue.completed)
            self.assertEqual(len((p.output/'skill-events.jsonl').read_text().splitlines()), 6)


if __name__ == '__main__':
    unittest.main()
