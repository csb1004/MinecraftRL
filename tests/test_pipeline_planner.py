import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.kb import Claim, Evidence, KnowledgeBase
from minecraftrl.pipeline.planner import (block_family, clear_plan, concrete_skill, escape_plan, harvest_verdict, oversight, placeable,
                                          plan, tool_of, validate_plan)
from minecraftrl.pipeline.steps import Step

CTX = {'game_version': '1.16.5'}
PIT_INVENTORY = {'dirt': 3, 'cobblestone': 3, 'oak_planks': 3, 'oak_log': 1, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stick': 2,
                 'wheat_seeds': 1}
WEB = [Evidence('web', 'support', 'https://minecraft.wiki')]


def req(tool, tier): return dict(tool_class=tool, min_tier=tier)


IRON_REQ_STONE = Claim(1, 'iron_ore', 'harvest_requires', req('pickaxe', 'stone'), {}, 'supported', 1, 1, 0)
IRON_REQ_WOODEN = Claim(2, 'iron_ore', 'harvest_requires', req('pickaxe', 'wooden'), {}, 'proposed', 1, 1, 0)
LOG_REQ_HAND = Claim(3, 'log', 'harvest_requires', req(None, 'hand'), {}, 'supported', 1, 1, 0)


def recipe(ingredients, count, grid): return dict(ingredients=ingredients, count=count, grid=grid)


def iron_kb(kb):
    add = lambda s, r, o, c={}: kb.add_claim(s, r, o, c, WEB, run_id='fixture')
    add('planks', 'recipe', recipe({'log': 1}, 4, 2)); add('stick', 'recipe', recipe({'planks': 2}, 4, 2))
    add('crafting_table', 'recipe', recipe({'planks': 4}, 1, 2))
    add('wooden_pickaxe', 'recipe', recipe({'planks': 3, 'stick': 2}, 1, 3))
    add('stone_pickaxe', 'recipe', recipe({'cobblestone': 3, 'stick': 2}, 1, 3))
    add('stone', 'drops', 'cobblestone'); add('log', 'harvest_requires', req(None, 'hand'))
    add('stone', 'harvest_requires', req('pickaxe', 'wooden')); add('iron_ore', 'harvest_requires', req('pickaxe', 'stone'))
    add('iron_ore', 'drops', 'iron_ore', {'game_version': '1.16.5'})
    return kb


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); root = Path(self.temp.name)
        self.empty = KnowledgeBase(root / 'empty.sqlite'); self.kb = iron_kb(KnowledgeBase(root / 'iron.sqlite'))

    def tearDown(self):
        self.empty.close(); self.kb.close(); self.temp.cleanup()

    def test_empty_kb_gives_naive_single_step(self):
        p = plan('iron_ore', 1, {}, self.empty, CTX)
        self.assertTrue(p.naive); self.assertEqual([s.skill for s in p.steps], ['mine_iron_ore'])
        self.assertEqual(p.steps[0].params['precondition'], 'unverified'); self.assertEqual(p.guarded, {'iron_ore'})
        self.assertEqual(p.needs, {'iron_ore'})

    def test_empty_kb_cobblestone_goal_uses_the_registered_skill_effect(self):
        # Second task (2026-11-05): with no knowledge the goal item is still the declared effect of a registered skill
        # (mine stone -> cobblestone); its harvest condition stays unverified and the guard protects stone.
        p = plan('cobblestone', 3, {}, self.empty, CTX)
        self.assertTrue(p.naive)
        self.assertEqual([(s.skill, s.target, s.count, s.block, s.params) for s in p.steps],
                         [('mine_cobblestone', 'cobblestone', 3, 'stone', {'precondition': 'unverified', 'source': 'skill_effect'})])
        self.assertIn('stone', p.guarded); self.assertIn('stone', p.missing)
        full = plan('cobblestone', 3, {}, self.kb, CTX)
        self.assertEqual([s.skill for s in full.steps], ['collect_log', 'craft', 'mine_cobblestone'])

    def test_intermediate_item_uses_the_registered_skill_effect_without_drops_knowledge(self):
        # Round 3c (2026-11-05): asked about "cobblestone", gpt-5.4 answered "cobblestone drops cobblestone" four rounds in a
        # row; no claim said which mineable block yields it, so the plan never formed. The stone skill declares cobblestone.
        claim, = self.kb.query('stone', 'drops', context=CTX); self.kb.set_status(claim.id, 'retracted', 'test', run_id='t')
        p = plan('iron_ore', 1, {}, self.kb, CTX)
        self.assertEqual(p.problems, [])
        self.assertIn(('mine_cobblestone', 'cobblestone', 'stone'), [(s.skill, s.target, s.block) for s in p.steps])

    def test_harvest_verdict_unknown_violation_ok(self):
        self.assertEqual(harvest_verdict([], 'air')[0], 'unknown')
        self.assertEqual(harvest_verdict([IRON_REQ_STONE], 'wooden_pickaxe')[0], 'violation')
        self.assertEqual(harvest_verdict([IRON_REQ_STONE], 'stone_pickaxe')[0], 'ok')
        self.assertEqual(harvest_verdict([LOG_REQ_HAND], 'air')[0], 'ok')
        self.assertEqual(harvest_verdict([IRON_REQ_WOODEN, IRON_REQ_STONE], 'wooden_pickaxe')[0], 'violation')

    def test_block_family_and_tool_of(self):
        self.assertEqual(block_family('dark_oak_log'), 'log'); self.assertEqual(block_family('iron_ore'), 'iron_ore')
        self.assertEqual(tool_of('stick'), (None, 'hand')); self.assertEqual(tool_of('stone_axe'), ('axe', 'stone'))
        self.assertEqual(tool_of('air'), (None, 'hand'))

    def test_full_chain_from_web_knowledge(self):
        p = plan('iron_ore', 1, {}, self.kb, CTX)
        self.assertEqual([(s.skill, s.target, s.count) for s in p.steps], [
            ('collect_log', 'log', 4), ('craft', 'wooden_pickaxe', 1), ('mine_cobblestone', 'cobblestone', 3),
            ('craft', 'stone_pickaxe', 1), ('mine_iron_ore', 'iron_ore', 1)])
        self.assertTrue(all(s.claim_ids for s in p.steps[1:])); self.assertEqual(p.guarded, {'log', 'stone', 'iron_ore'})
        self.assertEqual(p.needs, {'log', 'wooden_pickaxe', 'cobblestone', 'stone_pickaxe', 'iron_ore'})
        self.assertFalse(p.naive); self.assertEqual(p.problems, [])

    def test_existing_inventory_skips_steps(self):
        p = plan('iron_ore', 1, {'stone_pickaxe': 1}, self.kb, CTX)
        self.assertEqual([s.skill for s in p.steps], ['mine_iron_ore']); self.assertEqual(p.needs, {'stone_pickaxe', 'iron_ore'})

    def test_disputed_and_retracted_claims_ignored(self):
        claim, = self.kb.query('iron_ore', 'harvest_requires', context=CTX)
        self.kb.set_status(claim.id, 'disputed', 'test', run_id='t')
        p = plan('iron_ore', 1, {}, self.kb, CTX)
        self.assertTrue(p.naive); self.assertTrue(p.problems)

    def test_version_conditional_drops_selects_context(self):
        self.kb.add_claim('iron_ore', 'drops', 'raw_iron', {'game_version': {'gte': '1.17'}}, WEB, run_id='t')
        self.assertEqual(plan('iron_ore', 1, {}, self.kb, CTX).steps[-1].target, 'iron_ore')

    def test_unmapped_requirement_reports_problem(self):
        self.kb.add_claim('iron_ore', 'harvest_requires', req('pickaxe', 'diamond'), {}, WEB, run_id='t')
        p = plan('iron_ore', 1, {}, self.kb, CTX)
        self.assertTrue(any('diamond_pickaxe' in x for x in p.problems)); self.assertNotIn('diamond_pickaxe', [s.target for s in p.steps])

    def test_missing_intermediate_recipe_reports_problem(self):
        # Real gpt-5.4 answer (2026-11-04) dropped the stick recipe as uncited; unknown is not "free".
        claim, = self.kb.query('stick', 'recipe', context=CTX)
        self.kb.set_status(claim.id, 'retracted', 'test', run_id='t')
        p = plan('iron_ore', 1, {}, self.kb, CTX)
        self.assertTrue(any('stick' in x for x in p.problems))

    def test_validate_plan_rejects_unknown_skill(self):
        self.assertTrue(validate_plan([Step('x', 'teleport', 'policy', 'iron_ore', 1)], self.kb, CTX))
        self.assertEqual(validate_plan(plan('iron_ore', 1, {}, self.kb, CTX).steps, self.kb, CTX), [])

    def test_oversight_follows_goal_not_behaviour(self):
        # User rule (2026-11-04): iron is protected only when the goal needs it.
        steps = [Step('logs', 'collect_log', 'policy', 'log', 3, block='log')]
        self.assertEqual(oversight('log', steps), (frozenset({'log'}), frozenset({'log'})))
        guarded, needs = oversight('iron_ore', [Step('iron', 'mine_iron_ore', 'policy', 'iron_ore', 1, block='iron_ore')])
        self.assertIn('iron_ore', guarded)

    def test_concrete_skill_uses_selected_prompts(self):
        selected = {'log': 'wood_visual', 'cobblestone': 'mine_stone_text', 'iron_ore': 'mine_iron_text', 'escape': 'climb_out_text'}
        self.assertEqual(concrete_skill('mine_iron_ore', selected), 'mine_iron_text')
        self.assertEqual(concrete_skill('craft', selected), 'craft'); self.assertEqual(concrete_skill('equip', selected), 'equip')
        self.assertEqual(concrete_skill('escape_pit', selected), 'climb_out_text')

    def pillar_kb(self, method='pillar_up', per_level=1):
        return self.kb.add_claim('vertical_pit', 'escape_by', dict(method=method, blocks_per_level=per_level), {}, WEB, run_id='t')

    def test_plan_keeps_what_the_rest_consumes(self):
        p = plan('iron_ore', 1, PIT_INVENTORY, self.kb, CTX)
        self.assertEqual([s.target for s in p.steps], ['stone_pickaxe', 'iron_ore'])
        self.assertEqual(p.keep, {'cobblestone': 3, 'stick': 2, 'crafting_table': 1})

    def test_escape_plan_climbs_with_spare_stacks_then_resumes_the_goal(self):
        # Real post-cobblestone inventory (run B, 2026-11-04) at the bottom of the 6-deep pit: cobblestone, sticks and the
        # table stay for the stone pickaxe; dirt 3 + planks 3 are spare, one equip and one climb per stack.
        claim = self.pillar_kb()
        p = escape_plan('iron_ore', 1, PIT_INVENTORY, 6, self.kb, CTX)
        self.assertEqual([(s.skill, s.kind, s.target, s.count) for s in p.steps], [
            ('equip', 'rule', 'dirt', 1), ('escape_pit', 'policy', 'rise', 3), ('equip', 'rule', 'oak_planks', 1),
            ('escape_pit', 'policy', 'rise', 3), ('craft', 'rule', 'stone_pickaxe', 1), ('mine_iron_ore', 'policy', 'iron_ore', 1)])
        self.assertEqual(p.steps[0].params, {'goals': ['dirt'], 'equip': 'dirt', 'equip_only': True})
        self.assertEqual(p.steps[1].params, {'until': 'rise', 'expect': 'rise', 'to_surface': True}); self.assertEqual(p.steps[1].claim_ids, (claim,))
        self.assertEqual(len({s.id for s in p.steps}), 6)
        self.assertFalse(p.naive); self.assertEqual(p.problems, [])
        self.assertEqual(validate_plan(p.steps, self.kb, CTX), [])
        self.assertFalse({'dirt', 'oak_planks', 'rise'} & p.needs)  # blocks spent on the pillar are not goal resources

    def test_escape_never_spends_blocks_the_goal_still_needs(self):
        self.pillar_kb()
        p = escape_plan('iron_ore', 1, {'dirt': 2, 'cobblestone': 9, 'stick': 2, 'crafting_table': 1, 'wooden_pickaxe': 1}, 6,
                        self.kb, CTX)
        self.assertTrue(any('placeable' in x for x in p.problems))
        self.assertNotIn('cobblestone', [s.target for s in p.steps if s.skill == 'equip'])

    def test_escape_without_knowledge_is_missing_not_assumed(self):
        p = escape_plan('iron_ore', 1, PIT_INVENTORY, 6, self.kb, CTX)
        self.assertIn('vertical_pit', p.missing); self.assertTrue(p.problems)
        self.assertNotIn('escape_pit', [s.skill for s in p.steps])

    def test_escape_method_without_a_skill_is_a_problem(self):
        self.pillar_kb(method='staircase', per_level=0)
        p = escape_plan('iron_ore', 1, PIT_INVENTORY, 6, self.kb, CTX)
        self.assertTrue(any('staircase' in x for x in p.problems)); self.assertNotIn('vertical_pit', p.missing)

    def test_shallow_pit_needs_no_blocks(self):
        self.pillar_kb()
        p = escape_plan('iron_ore', 1, PIT_INVENTORY, 1, self.kb, CTX)
        self.assertEqual([(s.skill, s.count) for s in p.steps[:1]], [('escape_pit', 1)])

    def test_clear_plan_needs_movement_knowledge(self):
        steps, problems, missing = clear_plan(self.kb, CTX)
        self.assertEqual((steps, missing), ([], ('solid_block',))); self.assertTrue(problems)
        claim = self.kb.add_claim('solid_block', 'blocks_movement', True, {}, WEB, run_id='t')
        steps, problems, missing = clear_plan(self.kb, CTX)
        self.assertEqual([(s.skill, s.kind, s.claim_ids) for s in steps], [('clear_above', 'rule', (claim,))])
        self.assertEqual((problems, missing), ([], ()))
        self.assertEqual(validate_plan(steps, self.kb, CTX), [])
        self.assertEqual(concrete_skill('clear_above', {}), 'clear_above')
        self.assertEqual(concrete_skill('climb_out_text', {'escape': 'climb_out_text'}), 'climb_out_text')  # already concrete

    def test_placeable(self):
        self.assertTrue(all(map(placeable, ('dirt', 'cobblestone', 'oak_planks', 'birch_log'))))
        self.assertFalse(any(map(placeable, ('air', 'stick', 'wooden_pickaxe', 'crafting_table', 'iron_ore'))))


if __name__ == '__main__':
    unittest.main()
