import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.kb import KnowledgeBase
from minecraftrl.pipeline.openai_client import ResponsesClient
from minecraftrl.pipeline.remedy import choose, curate, outcome, retrieve, situation, store, trigger_key
from minecraftrl.pipeline.steps import Step

from test_pipeline_knowledge import FakeTransport

CTX = {'game_version': '1.16.5'}
CLIMB = Step('climb-2', 'climb_out_text', 'policy', 'rise', 3, params={'until': 'rise', 'expect': 'rise'})
IRON = Step('mine-iron_ore', 'mine_iron_text', 'policy', 'iron_ore', 1, block='iron_ore')
SKILLS = {'mine_iron_text': {'target_items': ['iron_ore']}, 'break_iron_text': {'target_items': ['iron_ore']},
          'climb_out_text': {'target_items': [], 'source': 'text_prompt:climb out of the hole'},
          'wood_visual': {'target_items': ['log']}}


def sit(step=CLIMB, held='oak_planks', inventory=None, kind='stagnation'):
    return situation(step, kind, inventory if inventory is not None else {'oak_planks': 3, 'dirt': 2}, held, SKILLS)


def remedy(action='equip', arg='dirt', risk='jumping with planks in a 1-wide pit without placing them under the feet',
           trigger=None):
    return dict(action=action, arg=arg, risk_pattern=risk,
                trigger=trigger or dict(failure_type='stagnation', skill='climb_out_text', climbing=True),
                reason='the policy places dirt more reliably')


def reply(candidate):
    return {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(candidate)}]}],
            'usage': {'input_tokens': 500, 'output_tokens': 80}}


class CuratorTests(unittest.TestCase):
    def test_accepts_an_executable_specific_remedy(self):
        self.assertEqual(curate(remedy(), sit(), set()), (True, None))
        self.assertEqual(curate(remedy('switch_skill', 'break_iron_text', 'the text prompt keeps the view on the floor',
                                       dict(failure_type='stagnation', skill='mine_iron_text')), sit(IRON, held='stone_pickaxe'), set()),
                         (True, None))

    def test_the_five_checks_reject(self):
        # MineEvolve eq. 6: schema, match, exec, specificity, conflict.
        self.assertEqual(curate(dict(remedy(), action='try_again'), sit(), set())[1], 'schema')
        self.assertEqual(curate(remedy(trigger=dict(failure_type='stagnation', weather='rain')), sit(), set())[1], 'match')
        self.assertEqual(curate(remedy(trigger=dict(failure_type='stagnation', skill='wood_visual')), sit(), set())[1], 'match')
        self.assertEqual(curate(remedy(arg='cobblestone'), sit(), set())[1], 'exec')        # not in the inventory
        self.assertEqual(curate(remedy('switch_skill', 'climb_out_text'), sit(), set())[1], 'exec')   # no other climb skill
        self.assertEqual(curate(remedy('clear_above', None, trigger=dict(failure_type='stagnation', skill='mine_iron_text')),
                                sit(IRON, held='stone_pickaxe'), set())[1], 'exec')        # not a climb
        self.assertEqual(curate(remedy(arg='stick'), sit(inventory={'oak_planks': 3, 'stick': 2}), set())[1], 'exec')   # no pillar block
        mining = dict(failure_type='stagnation', skill='mine_iron_text')
        self.assertEqual(curate(remedy(arg='dirt', trigger=mining), sit(IRON, held='stone_pickaxe'), set())[1], 'exec')  # not a tool
        self.assertEqual(curate(remedy(arg='stone_pickaxe', trigger=mining),
                                sit(IRON, held='air', inventory={'stone_pickaxe': 1}), set()), (True, None))
        self.assertEqual(curate(remedy(arg='oak_planks'), sit(), set())[1], 'spec')          # already in hand
        self.assertEqual(curate(remedy('replan', None), sit(), set())[1], 'exec')            # the planner does not plan out of a pit
        self.assertEqual(curate(remedy(risk='stuck'), sit(), set())[1], 'spec')
        self.assertEqual(curate(remedy(), sit(), {('equip', 'dirt')})[1], 'conflict')


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.kb = KnowledgeBase(Path(self.temp.name) / 'kb.sqlite')

    def tearDown(self): self.kb.close(); self.temp.cleanup()

    def test_stored_remedies_are_retrieved_by_trigger_and_confidence(self):
        key = trigger_key('stagnation', 'climb_out_text')
        weak = store(self.kb, key, remedy('turn_away', None), CTX, model='m', run_id='r')
        strong = store(self.kb, key, remedy(), CTX, model='m', run_id='r')
        outcome(self.kb, strong, True, 'r:climb-2', run_id='r')
        claim = retrieve(self.kb, key, CTX, sit())
        self.assertEqual((claim.id, claim.status, claim.object['action']), (strong, 'supported', 'equip'))
        self.assertIsNone(retrieve(self.kb, trigger_key('stagnation', 'wood_visual'), CTX, sit()))
        # Not executable now (no dirt): the next usable one.
        self.assertEqual(retrieve(self.kb, key, CTX, sit(inventory={'oak_planks': 3})).id, weak)

    def test_a_remedy_with_game_support_is_not_dropped_by_one_failure(self):
        # Curator usage history (MineEvolve): disputed only when the game refuted it more often than it supported it.
        key = trigger_key('stagnation', 'climb_out_text')
        rid = store(self.kb, key, remedy(), CTX, model='m', run_id='r')
        for _ in range(2): outcome(self.kb, rid, True, 'r:climb-2', run_id='r')
        outcome(self.kb, rid, False, 'r:climb-2', run_id='r')
        self.assertEqual(retrieve(self.kb, key, CTX, sit()).id, rid)
        for _ in range(2): outcome(self.kb, rid, False, 'r:climb-2', run_id='r')
        self.assertIsNone(retrieve(self.kb, key, CTX, sit()))

    def test_a_stored_equip_of_the_item_already_in_hand_is_skipped(self):
        key = trigger_key('stagnation', 'climb_out_text')
        store(self.kb, key, remedy(), CTX, model='m', run_id='r')
        self.assertIsNone(retrieve(self.kb, key, CTX, sit(held='dirt')))

    def test_a_failed_remedy_is_disputed_and_then_a_conflict(self):
        key = trigger_key('stagnation', 'climb_out_text')
        rid = store(self.kb, key, remedy(), CTX, model='m', run_id='r')
        outcome(self.kb, rid, False, 'r:climb-2', run_id='r')
        self.assertIsNone(retrieve(self.kb, key, CTX, sit()))
        self.assertEqual(choose(self.kb, None, CTX, sit(), [], run_id='r')['remedy']['action'], 'turn_away')
        llm = ResponsesClient('gpt-test', 'k', allow_paid=True, transport=FakeTransport(reply(remedy())), max_calls=4)
        chosen = choose(self.kb, llm, CTX, sit(), [], run_id='r')
        self.assertEqual((chosen['source'], chosen['rejected']), ('temporary', 'conflict'))


class ChooseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.kb = KnowledgeBase(Path(self.temp.name) / 'kb.sqlite')

    def tearDown(self): self.kb.close(); self.temp.cleanup()

    def test_knowledge_first_then_the_inducer_then_a_temporary_remedy(self):
        transport = FakeTransport(reply(remedy()))
        llm = ResponsesClient('gpt-test', 'k', allow_paid=True, transport=transport, max_calls=4)
        feedback = [dict(step_id='climb-2', success=False, failure_type='stagnation', progress=.02, stagnant=True)]
        induced = choose(self.kb, llm, CTX, sit(), feedback, run_id='r')
        self.assertEqual((induced['source'], induced['remedy']['action'], induced['remedy']['arg']), ('induced', 'equip', 'dirt'))
        self.assertIsNotNone(induced['claim_id']); self.assertEqual(len(transport.requests), 1)
        sent = json.loads(transport.requests[0]['input'][1]['content'])
        self.assertEqual(sent['feedback'], feedback); self.assertIn('equip', sent['actions'])
        again = choose(self.kb, llm, CTX, sit(), feedback, run_id='r')
        self.assertEqual((again['source'], again['claim_id']), ('knowledge', induced['claim_id'])); self.assertEqual(len(transport.requests), 1)
        self.assertEqual(sent['situation']['skill_prompt'], 'text_prompt:climb out of the hole')
        # No approved LLM and nothing stored for this trigger: turn away, not stored.
        other = choose(self.kb, None, CTX, sit(IRON, held='stone_pickaxe'), feedback, run_id='r')
        self.assertEqual((other['source'], other['remedy']['action'], other['claim_id']), ('temporary', 'turn_away', None))


    def test_the_last_calls_of_the_budget_are_kept_for_knowledge(self):
        # A stall remedy must not use up the calls the run needs to get out of a pit or past a ceiling (check E2 used all 8).
        transport = FakeTransport(reply(remedy()))
        llm = ResponsesClient('gpt-test', 'k', allow_paid=True, transport=transport, max_calls=2)
        chosen = choose(self.kb, llm, CTX, sit(), [], run_id='r')
        self.assertEqual((chosen['source'], chosen['rejected'], len(transport.requests)), ('temporary', 'llm_budget_reserved', 0))


class AdaptorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.kb = KnowledgeBase(Path(self.temp.name) / 'kb.sqlite')

    def tearDown(self): self.kb.close(); self.temp.cleanup()

    def test_an_equip_remedy_inserts_the_repair_before_the_same_step_and_a_new_stall_refutes_it(self):
        from minecraftrl.pipeline.oversight import Oversight
        key = trigger_key('stagnation', 'climb_out_text')
        rid = store(self.kb, key, remedy(), CTX, model='m', run_id='seed')
        o = Oversight(self.kb, CTX, 'iron_ore', [CLIMB], run_id='r', skills=SKILLS); events = []
        o.emit = lambda kind, **fields: events.append((kind, fields))
        stalled = dict(step_id='climb-2', skill='climb_out_text', success=False, failure_type='stagnation', progress=.0, stagnant=True)
        event = dict(kind='stagnation', hold=True, evidence=dict(p=0., remaining=3.))
        out = o.slow_path(event, {'oak_planks': 3, 'dirt': 2}, CLIMB, attempt=stalled, held='oak_planks')
        self.assertEqual((out['route'], out['resolution'], out['remedy_source'], out['carry']), ('remedy', 'insert', 'knowledge', True))
        self.assertEqual([s.id for s in out['plan_steps']], ['equip-dirt', 'climb-2'])
        self.assertEqual(out['plan_steps'][0].params, {'goals': ['dirt'], 'equip': 'dirt', 'equip_only': True})
        self.assertIn('plan', [k for k, _ in events])
        o.record_attempt(CLIMB, stalled)                       # the held attempt itself says nothing about the remedy
        self.assertEqual(self.kb.query(key, 'remedy', context=CTX)[0].status, 'proposed')
        o.slow_path(event, {'oak_planks': 3}, CLIMB, attempt=stalled, held='dirt')   # the repaired attempt stalled again
        self.assertEqual(self.kb.query(key, 'remedy', context=CTX), [])
        self.assertEqual([c.status for c in self.kb.query(key, 'remedy', context=CTX, usable_only=False) if c.id == rid], ['disputed'])


    def test_a_policy_hold_without_a_decision_gets_a_remedy_instead_of_stopping(self):
        # Check F3 (2026-11-25): the dirt ran out 1.25 levels below the surface (climb_without_block, too shallow for the
        # escape replan); the slow path had no decision and stopped with oak planks in the inventory.
        from minecraftrl.pipeline.oversight import Oversight
        event = dict(kind='climb_without_block', hold=True, evidence=dict(depth=1.25, held='air', remaining=1.25))
        held = dict(step_id='climb-2', skill='climb_out_text', success=False, failure_type='climb_without_block', progress=.6, stagnant=False)
        planks = remedy(arg='oak_planks', risk='the hand is empty below the surface and a pillar needs a block',
                        trigger=dict(failure_type='climb_without_block', skill='climb_out_text', climbing=True, held='air'))
        llm = ResponsesClient('gpt-test', 'k', allow_paid=True, transport=FakeTransport(reply(planks)), max_calls=4)
        o = Oversight(self.kb, CTX, 'iron_ore', [CLIMB], run_id='r', skills=SKILLS, llm=llm)
        out = o.slow_path(event, {'oak_planks': 3}, CLIMB, attempt=held, held='air')
        self.assertEqual((out['route'], out['resolution'], out['remedy_source'], out['llm']), ('remedy', 'insert', 'induced', 'called'))
        self.assertEqual([s.id for s in out['plan_steps']], ['equip-oak_planks', 'climb-2'])
        # The next run has no approved LLM: the stored remedy is used as it is.
        free = Oversight(self.kb, CTX, 'iron_ore', [CLIMB], run_id='r2', skills=SKILLS); free.max_retries = 2
        again = free.slow_path(event, {'oak_planks': 3}, CLIMB, attempt=held, held='air')
        self.assertEqual((again['remedy_source'], again['llm'], again['resolution']), ('knowledge', 'not_needed', 'insert'))
        # Nothing stored for this skill and no LLM: a temporary turn-away and another try, bounded by the retry limit.
        other = Step('climb-3', 'wood_visual', 'policy', 'rise', 3, params={'until': 'rise', 'expect': 'rise'})
        outs = [free.slow_path(event, {'oak_planks': 3}, other, held='air') for _ in range(3)]
        self.assertEqual([(x['route'], x['resolution']) for x in outs], [('remedy', 'retry'), ('remedy', 'retry'), ('retry_limit', 'stop')])
        self.assertEqual(outs[0]['remedy_source'], 'temporary')
        # Low health has no repair: it still stops.
        low = free.slow_path(dict(kind='low_health', hold=True, evidence=dict(health=4.)), {'oak_planks': 3}, CLIMB, held='air')
        self.assertEqual((low['route'], low['resolution']), ('ask_llm', 'stop'))
        # A rule step's failure keeps its own default.
        craft = Step('craft-x', 'craft', 'rule', 'stone_pickaxe', 1, params={'goals': ['stone_pickaxe']})
        rule = free.slow_path(dict(kind='rule_skill_failed', hold=True, evidence=dict(reason='boom')), {}, craft)
        self.assertEqual((rule['route'], rule['resolution']), ('retry_stage', 'retry'))


    def test_a_clear_above_remedy_is_recorded_like_the_ceiling_replan(self):
        from minecraftrl.pipeline.kb import Evidence
        from minecraftrl.pipeline.oversight import Oversight
        self.kb.add_claim('solid_block', 'blocks_movement', True, {}, [Evidence('web', 'support', 'https://minecraft.wiki')], run_id='seed')
        key = trigger_key('stagnation', 'climb_out_text')
        store(self.kb, key, remedy('clear_above', None, 'the head keeps hitting a block over the pit'), CTX, model='m', run_id='seed')
        o = Oversight(self.kb, CTX, 'iron_ore', [CLIMB], run_id='r', skills=SKILLS); events = []
        o.emit = lambda kind, **fields: events.append((kind, fields))
        out = o.slow_path(dict(kind='stagnation', hold=True, evidence=dict(p=0., remaining=2.4)), {'oak_planks': 3}, CLIMB, held='oak_planks')
        self.assertEqual([(s.id, s.count) for s in out['plan_steps']], [('clear-above', 1), ('climb-2-after-clear', 3)])
        self.assertEqual(out['resolution'], 'insert')
        self.assertEqual([k for k, _ in events], ['remedy', 'plan', 'learning_request'])
        self.assertEqual(len(self.kb.dependents(out['plan_steps'][0].claim_ids[0])), 1)   # the plan is recorded with its claims

    def test_a_replan_that_gives_the_same_step_is_a_turn_away_not_a_plain_retry(self):
        from minecraftrl.pipeline.oversight import Oversight
        from test_pipeline_planner import iron_kb
        iron_kb(self.kb)
        key = trigger_key('stagnation', 'mine_iron_text')
        store(self.kb, key, remedy('replan', None, 'the search drifted away from the ore face', dict(failure_type='stagnation')),
              CTX, model='m', run_id='seed')
        o = Oversight(self.kb, CTX, 'iron_ore', [IRON], run_id='r', skills=SKILLS, selected={'iron_ore': 'mine_iron_text'})
        out = o.slow_path(dict(kind='stagnation', hold=True, evidence=dict(p=0.)), {'stone_pickaxe': 1}, IRON, held='stone_pickaxe')
        self.assertEqual((out['resolution'], out['prelude'], out['plan_steps']), ('retry', 'turn_away', None))
        self.assertIn('same step', out['reason'])


if __name__ == '__main__':
    unittest.main()
