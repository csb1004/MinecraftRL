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
          'climb_out_text': {'target_items': []}, 'wood_visual': {'target_items': ['log']}}


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
        feedback = [dict(step='climb-2', success=False, failure_type='stagnation', progress=.02, stagnant=True)]
        induced = choose(self.kb, llm, CTX, sit(), feedback, run_id='r')
        self.assertEqual((induced['source'], induced['remedy']['action'], induced['remedy']['arg']), ('induced', 'equip', 'dirt'))
        self.assertIsNotNone(induced['claim_id']); self.assertEqual(len(transport.requests), 1)
        sent = json.loads(transport.requests[0]['input'][1]['content'])
        self.assertEqual(sent['feedback'], feedback); self.assertIn('equip', sent['actions'])
        again = choose(self.kb, llm, CTX, sit(), feedback, run_id='r')
        self.assertEqual((again['source'], again['claim_id']), ('knowledge', induced['claim_id'])); self.assertEqual(len(transport.requests), 1)
        # No approved LLM and nothing stored for this trigger: turn away, not stored.
        other = choose(self.kb, None, CTX, sit(IRON, held='stone_pickaxe'), feedback, run_id='r')
        self.assertEqual((other['source'], other['remedy']['action'], other['claim_id']), ('temporary', 'turn_away', None))


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
        stalled = dict(step='climb-2', skill='climb_out_text', success=False, failure_type='stagnation', progress=.0, stagnant=True)
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


if __name__ == '__main__':
    unittest.main()
