import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from minecraftrl.goal_contract import read_json
from minecraftrl.goal_queue import GoalQueue
from minecraftrl.logo_wood import admit, acceptance_cases
from minecraftrl.policies import Steve1Policy
from minecraftrl.rm_conditioning import validate_args
from minecraftrl.rm_program import ProgramPolicy, ProgramEnvironment, CAPABILITIES
from test_goal_queue import goal
import test_rm_conditioning as fixtures
from test_rm_logo_conditioning import FakeGame


class ProgramTests(unittest.TestCase):
    def policy(self, goals):
        p=fixtures.RMConditioningTests().policy()
        p.__class__=ProgramPolicy
        p.queue=GoalQueue(dict(schema='goal_queue_v1',goals=goals),CAPABILITIES)
        p.goal_id=None;p.progress=None;p.phase_snapshot=None
        p.validators,p.validator_review=admit(read_json(Path(__file__).resolve().parents[1]/'configs/logo-wood-validators-v1.json'))
        p.prompts=dict(root='gather wood',seek='find a tree',dirt='collect dirt')
        p.embeddings={k:fixtures.FakeEmbedding(k) for k in p.prompts}
        p.bank_metadata={'embeddings':{k:dict(vector_sha256=k) for k in p.prompts}}
        return p

    def wood(self, **kw):return goal('wood',skill='wood_memory',items=list(CAPABILITIES['wood_memory']),**kw)

    def baseline(self):
        b=copy.deepcopy(acceptance_cases()[0]['baseline']);b['server_inventory']['dirt']=0
        return b

    def test_two_goals_switch_conditions_without_world_or_recurrent_reset(self):
        b=self.baseline();wood=copy.deepcopy(b);wood['server_tick']=101;wood['server_inventory']['oak_log']=1
        dirt=copy.deepcopy(wood);dirt['server_tick']=102;dirt['server_inventory']['dirt']=1
        p=self.policy([self.wood(),goal('dirt',items=['dirt'],skill='dirt_fixed',depends_on=['wood'])])
        env=ProgramEnvironment(FakeGame([b,wood,dirt]),p);env.reset()
        with patch.object(Steve1Policy,'act',fixtures.RMConditioningTests.infer):
            action=p.act(None);self.assertEqual(env.step(action)[1],1.)
            self.assertEqual(p.queue.active,'dirt');self.assertFalse(env.reward.completed)
            self.assertEqual(p.agent.hidden_state,'next-seek')
            self.assertFalse(p.rows[-1]['terminal']);self.assertEqual(p.rows[-1]['goal_id'],'wood')
            action=p.act(None);self.assertEqual(env.step(action)[1],1.)
        self.assertTrue(env.reward.completed);self.assertTrue(p.rows[-1]['terminal'])
        self.assertEqual([r['goal_key'] for r in p.rows],['seek','dirt'])
        self.assertEqual([r['goal_id'] for r in p.rows],['wood','dirt'])
        with self.assertRaises(RuntimeError):env.step({'forward':1})

    def test_pending_result_survives_observer_expiry_in_actual_adapter(self):
        b=self.baseline();broken=dict(copy.deepcopy(b),server_tick=101,server_log_harvests=8,wood_v3_kind=2)
        missing=dict(copy.deepcopy(broken),server_tick=145,wood_v3_kind=0)
        servers=[b,broken,dict(broken,server_tick=102),dict(broken,server_tick=103),missing]
        p=self.policy([self.wood()]);env=ProgramEnvironment(FakeGame(servers),p);env.reset()
        with patch.object(Steve1Policy,'act',fixtures.RMConditioningTests.infer):
            for _ in range(4):env.step(p.act(None))
        self.assertEqual(p.progress.stage,'collect');self.assertEqual(p.goal_key(),'root')
        self.assertEqual(p.queue.active,'wood');self.assertFalse(p.queue.completed)

    def test_unsupported_successor_pauses_before_any_new_policy_inference(self):
        b=self.baseline();w=copy.deepcopy(b);w['server_tick']=101;w['server_inventory']['oak_log']=1
        p=self.policy([self.wood(),goal('stone',items=['cobblestone'],skill='stone_collect',depends_on=['wood'])])
        env=ProgramEnvironment(FakeGame([b,w]),p);env.reset()
        with patch.object(Steve1Policy,'act',fixtures.RMConditioningTests.infer):env.step(p.act(None))
        reasons=[];p.prepare_control(request_stop=reasons.append,emit=lambda *a,**k:None)
        self.assertEqual(reasons,['planning_paused'])
        with patch.object(Steve1Policy,'act') as infer:
            with self.assertRaises(InterruptedError):p.act(None)
            infer.assert_not_called()
        self.assertEqual(p.queue.records['wood']['status'],'completed')
        self.assertEqual(p.queue.records['stone']['reason'],'unsupported_skill')

    def test_cli_requires_explicit_program_and_isolation(self):
        root=Path(__file__).resolve().parents[1]
        args=SimpleNamespace(rm_conditioning='goal-queue',rm_goal_bank=root/'results/bank',
            rm_program=root/'configs/goals-wood-then-stone.json',root=root,policy='steve1',condition='wood',operation='episode')
        validate_args(args)
        args.rm_conditioning='logo-fixed'
        with self.assertRaises(ValueError):validate_args(args)
        args.rm_conditioning='goal-queue';args.rm_program=None
        with self.assertRaises(ValueError):validate_args(args)


if __name__=='__main__':unittest.main()
