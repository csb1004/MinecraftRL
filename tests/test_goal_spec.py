import copy
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from minecraftrl.goal_contract import read_json
from minecraftrl.goal_queue import GoalQueue
from minecraftrl.goal_spec import compile_spec, compile_response, expected_prompt, policy_condition, request
from minecraftrl.rm_program import CAPABILITIES, ProgramEnvironment, ProgramPolicy
from minecraftrl.semantic_goal import context, CONTEXT_FIELDS
import test_rm_program as program_fixtures
import test_rm_conditioning as condition_fixtures
from test_rm_conditioning import FakeEmbedding
from test_rm_logo_conditioning import FakeGame

ROOT = Path(__file__).resolve().parents[1]


def specification(quantity=3, target='wood'):
    return dict(schema='goal_spec_v1', goals=[dict(id='first', target=target, quantity=quantity,
        priority=50, depends_on=[], requires=[])])


class GoalSpecTests(unittest.TestCase):
    def queue(self, source=None):
        q = GoalQueue(source or specification(), CAPABILITIES)
        baseline = program_fixtures.ProgramTests().baseline()
        baseline['server_inventory']['oak_log'] = 8
        q.observe(baseline); q.select()
        return q, baseline

    def test_quantity_controls_reward_and_features_without_reencoding_goal(self):
        keys, hashes = [], []
        for amount in (1, 3, 64):
            q, before = self.queue(specification(amount))
            binding = policy_condition(q, 'search')
            values, desc = context(q, 'search', None, {})
            keys.append(binding['goal_key']); hashes.append(binding['sha256'])
            self.assertEqual(binding['quantity'], desc['quantity'])
            self.assertEqual(binding['items'], desc['target'])
            self.assertEqual(binding['sha256'], desc['goal_contract_sha256'])
            self.assertAlmostEqual(values[CONTEXT_FIELDS.index('quantity_log2')], math.log2(1+amount)/math.log2(65))
            almost = copy.deepcopy(before); almost['server_tick'] += 1
            almost['server_inventory']['oak_log'] += amount-1
            self.assertEqual(q.observe(almost), 0.)
            done = copy.deepcopy(almost); done['server_tick'] += 1; done['server_inventory']['oak_log'] += 1
            self.assertEqual(q.observe(done), 1.)
            self.assertTrue(q.completed)
            done['server_tick'] += 1
            self.assertEqual(q.observe(done), 0.)
        self.assertEqual(keys, ['seek']*3)
        self.assertEqual(len(set(hashes)), 3)

    def test_target_changes_both_inventory_and_policy_binding(self):
        q, _ = self.queue(specification(target='dirt'))
        condition = policy_condition(q, 'collect')
        self.assertEqual(condition['items'], ['dirt'])
        self.assertEqual(condition['goal_key'], 'dirt')
        pickup = policy_condition(q, 'collect', 'pickup')
        self.assertEqual(pickup['goal_key'], 'pickup_dirt')
        self.assertEqual(pickup['prompt'], 'pick up dirt')
        self.assertEqual(condition['sha256'], pickup['sha256'])

    def test_rejects_unsupported_injected_and_invalid_definitions(self):
        for change in ({'target':'stone'}, {'target':[]}, {'quantity':True}, {'quantity':0},
                       {'quantity':65}, {'skill':'wood_memory'}, {'prompt':'collect dirt'},
                       {'success':'always'}, {'objective':'collect dirt'}, {'depends_on':['first']}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                source=specification();source['goals'][0].update(change);compile_spec(source)
        source=specification();source['constraints']=['never attack']
        with self.assertRaises(ValueError):compile_spec(source)
        source=specification();source['goals'] *= 2
        with self.assertRaises(ValueError):compile_spec(source)

    def test_prerequisite_is_held_inventory_and_not_completion(self):
        source=specification(target='dirt');source['goals'][0]['requires']=[dict(target='wood',quantity=9)]
        q,before=self.queue(source)
        self.assertIsNone(q.active)
        self.assertEqual(q.records['first']['reason'],'missing_prerequisite_items')
        before['server_tick']+=1;before['server_inventory']['oak_log']=9
        self.assertEqual(q.observe(before),0.)
        q.select();self.assertEqual(q.active,'first');self.assertEqual(q.records['first']['progress'],0)

    def test_spec_is_copied_and_reprioritization_preserves_identity(self):
        source=specification();q,_=self.queue(source);source['goals'][0]['quantity']=64
        original=policy_condition(q,'search')
        q.reprioritize('first',90,expected_revision=1)
        self.assertEqual(policy_condition(q,'search')['sha256'],original['sha256'])
        self.assertEqual(q.records['first']['revision'],2)
        self.assertEqual(q.goals['first']['count'],3)

    def test_drift_is_rejected_before_reward_or_condition(self):
        for field,value in [('count',1),('items',['dirt']),('depends_on',['other'])]:
            q,before=self.queue();q.goals['first'][field]=value
            with self.assertRaisesRegex(ValueError,'drift'):policy_condition(q,'search')
            before['server_tick']+=1;before['server_inventory']['oak_log']=99
            with self.assertRaisesRegex(ValueError,'drift'):q.observe(before)
            self.assertNotEqual(q.records['first']['status'],'completed')

    def test_human_and_llm_use_identical_compiler_and_can_report_unsupported(self):
        source=specification()
        self.assertEqual(compile_spec(source),compile_response(dict(status='candidate',reason='',spec=source)))
        result=compile_response(dict(status='unsupported',reason='No supported crafting skill',spec=None))
        self.assertFalse(result['execution_authorized']);self.assertNotIn('program',result)
        with self.assertRaises(ValueError):
            compile_response(dict(status='unsupported',reason='unsupported',spec=source))
        self.assertEqual(request()['response_schema']['properties']['status']['enum'],['candidate','unsupported'])

    def test_cli_rejects_missing_or_disabled_structured_input_before_loading(self):
        from minecraftrl.rm_conditioning import validate_args
        args=SimpleNamespace(root=ROOT,rm_program=ROOT/'configs/unified-goals-wood-then-dirt.json',
            rm_conditioning='goal-queue',rm_goal_bank=ROOT/'results/bank',policy='steve1',condition='wood',operation='episode')
        for variant in (None,'linear-off','nonlinear-off'):
            args.s2_variant=variant
            with self.assertRaisesRegex(ValueError,'structured policy context'):validate_args(args)
        args.s2_variant='nonlinear-on';validate_args(args)
        args.s2_variant=None;args.recovery_variant='nonlinear-on'
        with self.assertRaisesRegex(ValueError,'exactly one goal'):validate_args(args)

    def test_existing_queue_input_remains_legacy(self):
        q=GoalQueue(compile_spec(specification())['program'],CAPABILITIES)
        self.assertIsNone(q.goal_spec)
        self.assertNotIn('goal_spec',q.snapshot())


class GoalSpecRuntimeTests(unittest.TestCase):
    def policy(self,source):
        fixture=program_fixtures.ProgramTests()
        p=fixture.policy([fixture.wood()])
        p.queue=GoalQueue(source,CAPABILITIES)
        p.prompts={k:expected_prompt(k) for k in ('root','seek','dirt','pickup_wood','pickup_dirt')}
        p.embeddings={k:FakeEmbedding(k) for k in p.prompts}
        p.bank_metadata={'prompts':p.prompts.copy(),'embeddings':{k:dict(vector_sha256=k) for k in p.prompts}}
        return p

    def test_real_queue_adapter_and_learning_context_share_goal_across_handoff(self):
        from minecraftrl.policies import Steve1Policy
        source=read_json(ROOT/'configs/unified-goals-wood-then-dirt.json')
        b=program_fixtures.ProgramTests().baseline();states=[b]
        for n in (1,2,3):
            next_state=copy.deepcopy(b);next_state['server_tick']+=n;next_state['server_inventory']['oak_log']=n;states.append(next_state)
        final=copy.deepcopy(states[-1]);final['server_tick']+=1;final['server_inventory']['dirt']=1;states.append(final)
        p=self.policy(source);game=FakeGame(states);env=ProgramEnvironment(game,p);env.reset()
        rewards=[]
        with patch.object(Steve1Policy,'act',condition_fixtures.RMConditioningTests.infer):
            for step in range(4):
                values,description=context(p.queue,p.selector.current,p.phase_snapshot,{})
                action=p.act(None)
                binding=p.pending['goal_binding']
                self.assertEqual(binding['quantity'],description['quantity'])
                self.assertEqual(binding['items'],description['target'])
                self.assertEqual(binding['sha256'],description['goal_contract_sha256'])
                if step==1:self.assertAlmostEqual(values[5],1/3)
                rewards.append(env.step(action)[1])
        self.assertEqual(rewards,[0.,0.,1.,1.])
        self.assertTrue(p.queue.completed)
        self.assertEqual([r['goal_binding']['target_ref'] for r in p.rows],['wood']*3+['dirt'])
        with patch.object(Steve1Policy,'act') as infer:
            with self.assertRaises(InterruptedError):p.act(None)
            infer.assert_not_called()
        with self.assertRaises(RuntimeError):env.step({'attack':1})

    def test_wrong_bank_text_is_rejected_before_inference(self):
        from minecraftrl.policies import Steve1Policy
        p=self.policy(specification())
        env=ProgramEnvironment(FakeGame([program_fixtures.ProgramTests().baseline()]),p);env.reset()
        p.prompts['seek']='collect dirt'
        with patch.object(Steve1Policy,'act') as infer:
            with self.assertRaisesRegex(ValueError,'disagree'):p.act(None)
            infer.assert_not_called()


class GoalSpecLearningTests(unittest.TestCase):
    def test_committed_tensor_trace_keeps_contract_and_trainer_rejects_drift(self):
        try:
            import torch
        except ImportError:
            self.skipTest('isolated tensor environment required')
        from minecraftrl.s2_learning import SemanticPolicy, SemanticEnvironment
        from minecraftrl.conditioned_head import ConditionedHead, ConditionedValue
        from minecraftrl.semantic_goal import ResultMonitor
        from minecraftrl.rl_policy import cpu_tree, state_fingerprint
        sys.path.insert(0,str(ROOT/'scripts'))
        from train_s2_conditions import validate

        class Motor(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(2,2)
            def forward(self,x):return {'attack':self.linear(x)}
            def sample(self,pd,deterministic=False):
                return {'attack':torch.distributions.Categorical(logits=pd['attack']).sample()}
            def logprob(self,actions,pd):
                return torch.distributions.Categorical(logits=pd['attack']).log_prob(actions['attack'])

        torch.set_num_threads(2)
        p=GoalSpecRuntimeTests().policy(specification());p.__class__=SemanticPolicy
        p.torch=torch;p.variant='linear-on';p.split='train';p.action_budget=10
        p.monitor=ResultMonitor();p.monitor_events=[];p.failure=p.end_reason=None
        p.learning_rows=[];p.learning_pending=None;p.seed=1
        p.agent.policy=SimpleNamespace(pi_head=ConditionedHead(Motor(),2,'linear-on'))
        p.agent._agent_action_to_env=lambda action:{'attack':[int(action['attack'].item())]}
        p.value=ConditionedValue(2,'linear-on');p.feature_width=2+len(CONTEXT_FIELDS)
        p.base_sha256='test-base';p.checkpoint_sha256=None
        p.initial_heads=cpu_tree(dict(actor=p.agent.policy.pi_head.state_dict(),value=p.value.state_dict()))
        p.initial_heads_sha256=state_fingerprint(p.initial_heads)
        def features(screen):
            values,p.current_contract=context(p.queue,p.selector.current,p.phase_snapshot,{})
            f=torch.tensor([0.,0.]+values,dtype=torch.float32).view(1,1,-1).expand(2,1,-1)
            return f,'next-state',values
        p.features=features  # Only the visual network is replaced; real heads/commit/archive/admission run.
        b=program_fixtures.ProgramTests().baseline();after=copy.deepcopy(b)
        after['server_tick']+=1;after['server_inventory']['oak_log']=3
        with tempfile.TemporaryDirectory() as folder:
            p.output=Path(folder);env=SemanticEnvironment(FakeGame([b,after]),p);env.reset()
            self.assertEqual(env.step(p.act(None))[1],1.)
            self.assertEqual(p.learning_rows[0]['goal_binding'],p.rows[0]['goal_binding'])
            with patch.object(ProgramPolicy,'close'):
                p.close()
            report=dict(execution_valid=True,status='target_reached',worker_exit_code=0,steps=1)
            p.archive_learning(report)
            saved=torch.load(p.output/'s2-trajectory.pt',map_location='cpu')
            validate(saved,report)
            for change in ('quantity','embedding','source','binding','context'):
                broken=copy.deepcopy(saved)
                if change=='quantity':broken['goal_spec']['source']['goals'][0]['quantity']=1
                if change=='embedding':broken['rows'][0]['embedding_sha256']='wrong'
                if change=='source':broken.pop('goal_spec')
                if change=='binding':broken['rows'][0].pop('goal_binding')
                if change=='context':
                    broken['rows'][0]['phase_context'][7]=0.
                    broken['rows'][0]['features'][...,-len(CONTEXT_FIELDS)+7]=0.
                with self.subTest(change=change),self.assertRaises(ValueError):validate(broken,report)
            self.assertEqual(saved['rows'][0]['goal_binding']['quantity'],3)
        with self.assertRaises(RuntimeError):env.step({'attack':1})

if __name__=='__main__':unittest.main()
