import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.kb import KnowledgeBase
from minecraftrl.pipeline.openai_client import ResponsesClient
from minecraftrl.pipeline.oversight import Oversight
from minecraftrl.pipeline.planner import plan

from test_pipeline_knowledge import candidate, response
from test_pipeline_planner import iron_kb

CTX = {'game_version': '1.16.5'}
SELECTED = {'log': 'wood_visual', 'cobblestone': 'mine_stone_text', 'iron_ore': 'mine_iron_text'}
R = lambda subject, ingredients, grid, count=1: candidate(subject=subject, relation='recipe',
                                                           obj={'ingredients': ingredients, 'count': count, 'grid': grid})
RECIPES = [R('stone_pickaxe', {'cobblestone': 3, 'stick': 2}, 3), R('wooden_pickaxe', {'planks': 3, 'stick': 2}, 3),
           R('planks', {'log': 1}, 2, 4), R('stick', {'planks': 2}, 2, 4), R('crafting_table', {'planks': 4}, 2)]
BLOCKS = [candidate(), candidate(relation='drops', obj='iron_ore'),
          candidate(subject='stone', obj={'tool_class': 'pickaxe', 'min_tier': 'wooden'}),
          candidate(subject='stone', relation='drops', obj='cobblestone')]
LOG = candidate(subject='log', obj={'tool_class': None, 'min_tier': 'hand'})


class PlanAheadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.kb = KnowledgeBase(Path(self.temp.name) / 'kb.sqlite'); self.events = []

    def tearDown(self): self.kb.close(); self.temp.cleanup()

    def oversight(self, replies=None):
        self.requests = []
        def reply(request, timeout): self.requests.append(json.loads(request.data)); return replies[len(self.requests) - 1]
        llm = ResponsesClient('gpt-test', 'k', allow_paid=True, transport=reply, max_calls=8) if replies is not None else None
        initial = plan('iron_ore', 1, {}, self.kb, CTX)
        o = Oversight(self.kb, CTX, 'iron_ore', initial.steps, run_id='r', llm=llm, selected=SELECTED)
        o.emit = lambda kind, **fields: self.events.append((kind, fields))
        return o, initial

    def asked(self, i): return json.loads(self.requests[i]['input'][1]['content'])

    def test_an_empty_store_is_filled_before_the_game_and_the_first_plan_is_the_whole_chain(self):
        # User request 2026-11-25: the first plan was the naive one-step plan and the knowledge came only after the
        # guard's hold in the game (checks E1-E3). Now the goal is planned with knowledge asked for up front.
        o, initial = self.oversight([response(BLOCKS + [LOG] + RECIPES)])
        self.assertTrue(initial.naive)
        out = o.plan_ahead(initial)
        self.assertEqual((out['resolution'], out['llm'], out['rounds'], out['missing']), ('replan', 'called', 1, []))
        self.assertEqual([s.id for s in out['plan_steps']],
                         ['logs', 'craft-wooden_pickaxe', 'mine-cobblestone', 'craft-stone_pickaxe', 'mine-iron_ore'])
        self.assertEqual([s.skill for s in out['plan_steps']], ['wood_visual', 'craft', 'mine_stone_text', 'craft', 'mine_iron_text'])
        self.assertEqual(self.asked(0)['event'], 'goal_received'); self.assertNotIn('missing', self.asked(0))
        self.assertIn('stone', o.guarded)   # the oversight follows the new plan
        self.assertEqual([k for k, _ in self.events], ['llm_request', 'kb_change', 'plan'])

    def test_missing_links_and_unverified_blocks_are_asked_for_by_name(self):
        # Check E2 (2026-11-24): the first answer had no log knowledge, so the guard held again at the first log.
        o, initial = self.oversight([response(BLOCKS + RECIPES[:1]), response(RECIPES[1:]), response([LOG])])
        out = o.plan_ahead(initial)
        self.assertEqual((out['resolution'], out['rounds'], out['missing']), ('replan', 3, []))
        self.assertEqual(self.asked(1)['missing'], ['wooden_pickaxe', 'stick', 'crafting_table'])
        self.assertEqual(self.asked(2)['missing'], ['log'])
        self.assertEqual(len(out['plan_steps']), 5)

    def test_a_complete_store_asks_nothing(self):
        iron_kb(self.kb)
        o, initial = self.oversight([])
        out = o.plan_ahead(initial)
        self.assertEqual((out['resolution'], out['llm'], out['rounds'], out['plan_steps'], self.requests), ('keep', 'not_needed', 0, None, []))

    def test_without_an_approved_llm_the_first_plan_stays_and_the_gaps_are_named(self):
        o, initial = self.oversight(None)
        out = o.plan_ahead(initial)
        self.assertEqual((out['resolution'], out['llm'], out['plan_steps'], out['missing']), ('keep', 'not_called', None, ['iron_ore']))

    def test_the_run_starts_from_the_plan_made_ahead_and_records_it(self):
        from minecraftrl.pipeline.cli import plan_ahead_steps, runtime_needs
        o, initial = self.oversight([response(BLOCKS + [LOG] + RECIPES)]); seen = []
        steps = plan_ahead_steps(o, initial, list(initial.steps), lambda kind, **fields: seen.append((kind, fields)))
        self.assertEqual(len(steps), 5)
        kinds = [k for k, _ in seen]; self.assertEqual(kinds, ['llm_request', 'kb_change', 'plan', 'plan_ahead'])
        record = seen[-1][1]
        self.assertEqual((record['resolution'], record['rounds'], record['missing']), ('replan', 1, []))
        self.assertNotIn('plan_steps', record)
        # Nothing to add: the first steps stay.
        again = plan_ahead_steps(o, plan('iron_ore', 1, {}, self.kb, CTX), ['first'], lambda kind, **fields: None)
        self.assertEqual(again, ['first'])
        # A remedy may switch to another registered skill for the same target: those embeddings are loaded too.
        config = {'selected': {'iron_ore': 'mine_iron_text', 'escape': 'climb_out_text'},
                  'skills': {'mine_iron_text': {'target_items': ['iron_ore']}, 'break_iron_text': {'target_items': ['iron_ore']},
                             'climb_out_text': {'target_items': []}, 'wood_visual': {'target_items': ['log']}}}
        self.assertEqual(runtime_needs([], config, oversight=True)[0], ['break_iron_text', 'climb_out_text', 'mine_iron_text'])
        self.assertEqual(runtime_needs([], config, oversight=False)[0], [])

    def test_planning_ahead_stops_asking_when_it_takes_too_long(self):
        # The supervisor's startup limit (600 s) runs until the session starts: slow calls must not use it up.
        o, initial = self.oversight([response(BLOCKS + RECIPES[:1]), response(RECIPES[1:]), response([LOG])])
        ticks = iter([0., 50., 130., 200., 300.]); o.clock = lambda: next(ticks)
        out = o.plan_ahead(initial)
        self.assertEqual((out['rounds'], out['resolution'], out['missing']), (2, 'replan', ['log']))
        self.assertIn('time limit', out['reason'])

    def test_knowledge_that_still_leaves_gaps_keeps_the_first_plan(self):
        o, initial = self.oversight([response(BLOCKS)] + [response([])] * 3)
        out = o.plan_ahead(initial)
        self.assertEqual((out['resolution'], out['rounds'], out['plan_steps']), ('keep', 4, None))
        self.assertEqual(out['missing'], ['stone_pickaxe'])


if __name__ == '__main__':
    unittest.main()
