import copy
import unittest
from pathlib import Path

from minecraftrl.goal_contract import digest, read_json
from minecraftrl.goal_actions import compile_action_conditions
from minecraftrl.logo_validators import request, ValidatorSet, SuccessMonitor
from minecraftrl.logo_wood import context, acceptance_cases, admit, action_candidate, prompts, stage

PROPOSAL = Path(__file__).resolve().parents[1]/'configs/logo-wood-validators-v1.json'


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.proposal = read_json(PROPOSAL)
        self.validators, self.review = admit(self.proposal)
        self.baseline = acceptance_cases()[0]['baseline']

    def current(self, **changes):
        result = copy.deepcopy(self.baseline)
        result.update(server_tick=101)
        result.update(changes)
        return result

    def test_request_proposal_and_caller_owned_cases(self):
        packet = request(context())
        self.assertEqual(packet['context_sha256'], self.proposal['context_sha256'])
        self.assertTrue(self.review['passed'])
        self.assertEqual(set(self.validators.functions), set(context()['validators']))
        self.assertTrue(callable(self.validators.functions['item_acquired']))

    def test_generated_python_unknown_fact_and_stale_response_rejected(self):
        for key, value in [('context_sha256', 'old'), ('python', '__import__("os").system("whoami")')]:
            bad = copy.deepcopy(self.proposal)
            bad[key] = value
            with self.assertRaises(ValueError):
                ValidatorSet(context(), bad)
        bad = copy.deepcopy(self.proposal)
        bad['validators'][0]['expression']['fact'] = 'tree_visible_now'
        with self.assertRaises(ValueError):
            ValidatorSet(context(), bad)

    def test_model_cannot_weaken_authoritative_root(self):
        bad = copy.deepcopy(self.proposal)
        bad['validators'][-1]['expression'] = dict(op='ge', fact='log_breaks', mode='delta', value=1)
        with self.assertRaisesRegex(ValueError, 'root cannot'):
            ValidatorSet(context(), bad)

    def test_schema_valid_wrong_semantics_get_revision_feedback(self):
        bad = copy.deepcopy(self.proposal)
        bad['validators'][2]['expression']['mode'] = 'current'
        review = ValidatorSet(context(), bad).verify(acceptance_cases())
        self.assertFalse(review['passed'])
        self.assertTrue(any(case['validator'] == 'log_broken' for case in review['failures']))
        with self.assertRaisesRegex(ValueError, 'acceptance'):
            admit(bad)

    def test_unknown_evidence_is_not_success_or_ordinary_false(self):
        fn = self.validators.functions['item_acquired']
        for inventory in ({}, None, {'oak_log':1}):
            self.assertIsNone(fn(self.current(server_inventory=inventory), self.baseline))
        for value in (True, -1, 0.5, float('nan'), float('inf')):
            inventory = dict(self.baseline['server_inventory'], oak_log=value)
            self.assertIsNone(fn(self.current(server_inventory=inventory), self.baseline))
        for changes in (dict(server_available=0), dict(server_available=True), dict(error=1), dict(server_tick=99)):
            self.assertTrue(all(value is None for value in self.validators.observe(self.current(**changes), self.baseline).values()))

    def test_same_generic_function_supports_different_item_and_quantity(self):
        ctx = context()
        ctx['facts']['resource_count']['items'] = ['dirt']
        ctx['root_requirement']['value'] = 64
        proposal = copy.deepcopy(self.proposal)
        proposal['context_sha256'] = digest(ctx)
        proposal['validators'][-1]['expression']['value'] = 64
        validators = ValidatorSet(ctx, proposal)
        baseline = dict(self.baseline, server_inventory={'dirt':5})
        self.assertFalse(validators.functions['item_acquired'](self.current(server_inventory={'dirt':68}), baseline))
        self.assertTrue(validators.functions['item_acquired'](self.current(server_inventory={'dirt':69}), baseline))

    def test_actual_minerl_float_encoded_flags_and_counts(self):
        baseline = copy.deepcopy(self.baseline)
        for key, value in baseline.items():
            baseline[key] = {k:float(v) for k,v in value.items()} if isinstance(value, dict) else float(value)
        monitor = SuccessMonitor(self.validators, baseline)
        current = copy.deepcopy(baseline)
        current['server_tick'] += 1.
        current['server_inventory']['oak_log'] = 1.
        self.assertEqual(monitor.advance(current), 1.)

    def test_stale_missing_observations_and_reward_absorption(self):
        monitor = SuccessMonitor(self.validators, self.baseline)
        with self.assertRaisesRegex(ValueError, 'stale'):
            monitor.advance(self.baseline)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            monitor.advance(self.current(server_inventory={}))
        self.assertEqual(monitor.last_tick, 100)
        self.assertEqual(monitor.advance(self.current(server_log_harvests=8)), 0.)
        inventory = dict(self.baseline['server_inventory'], oak_log=1)
        self.assertEqual(monitor.advance(self.current(server_tick=102, server_inventory=inventory)), 1.)
        self.assertTrue(monitor.completed)
        with self.assertRaisesRegex(RuntimeError, 'accepting'):
            monitor.advance(self.current(server_tick=103, server_inventory=inventory))

    def test_no_drop_or_mining_proxy_can_complete_or_reward_task(self):
        for changes in (dict(wood_server_mining=1), dict(wood_v3_kind=2), dict(server_log_harvests=8, wood_v3_kind=2)):
            monitor = SuccessMonitor(self.validators, self.baseline)
            server = self.current(**changes)
            self.assertEqual(monitor.advance(server), 0)
            self.assertNotEqual(stage(monitor.values, server), 'complete')
        monitor = SuccessMonitor(self.validators, self.baseline)
        monitor.advance(self.current(server_log_harvests=8, wood_v3_kind=2))
        self.assertEqual(stage(monitor.values, self.current(wood_v3_kind=2)), 'collect')
        monitor.advance(self.current(server_tick=102, server_log_harvests=8, wood_v3_kind=0))
        self.assertEqual(stage(monitor.values, self.current()), 'search')

    def test_authoritative_root_completion_precedes_missing_subgoal_hint(self):
        monitor = SuccessMonitor(self.validators, self.baseline)
        current = self.current(server_inventory=dict(self.baseline['server_inventory'], oak_log=1))
        del current['wood_v3_kind']
        self.assertEqual(monitor.advance(current), 1.)
        self.assertIsNone(monitor.values['target_known'])
        self.assertEqual(stage(monitor.values, current), 'complete')

    def test_expression_size_unknown_operator_and_empty_cases_rejected(self):
        bad = copy.deepcopy(self.proposal)
        expr = bad['validators'][0]['expression']
        for _ in range(8):
            expr = dict(op='all', args=[expr])
        bad['validators'][0]['expression'] = expr
        with self.assertRaisesRegex(ValueError, 'oversized'):
            ValidatorSet(context(), bad)
        self.assertFalse(self.validators.verify([])['passed'])
        for field in ('op', 'fact'):
            malformed = copy.deepcopy(self.proposal)
            malformed['validators'][0]['expression'][field] = []
            with self.assertRaises(ValueError):
                ValidatorSet(context(), malformed)


class ActionRenderingTests(unittest.TestCase):
    def test_target_purpose_and_paper_action_verbs_without_metadata_prose(self):
        self.assertEqual(prompts()['root'], 'chop down the tree, gather wood, pick up wood')
        self.assertIn('find a tree', prompts()['seek'])
        self.assertIn('gather wood', prompts()['seek'])
        self.assertNotIn('Expected result:', prompts()['root'])
        self.assertEqual(action_candidate()['conditions']['gather']['predicate_ref'], 'item_acquired')
        self.assertFalse(action_candidate()['capability_verified'])

    def test_targetless_or_purposeless_rendering_rejected(self):
        candidate = action_candidate()
        for key in ('seek', 'gather'):
            rendering = copy.deepcopy(candidate['rendering'])
            rendering['clauses'][key] = [dict(action_ref='pick', target_ref='resource')]
            with self.assertRaises(ValueError):
                compile_action_conditions(candidate['context'], candidate['proposal'], rendering)

    def test_general_renderer_preserves_changed_target_purpose_and_constraints(self):
        candidate = action_candidate()
        ctx, proposal, rendering = (copy.deepcopy(candidate[key]) for key in ('context','proposal','rendering'))
        ctx.update(root_goal='collect coal', active_goal='collect coal', constraints=['stay above water'])
        ctx['actions']['chop'] = 'mine'
        ctx['targets'].update(source='coal ore', tree='coal ore', resource='coal')
        for item in (proposal, rendering):
            item['context_sha256'] = digest(ctx)
        result = compile_action_conditions(ctx, proposal, rendering)
        text = result['conditions']['seek']['prompt']
        self.assertIn('find coal ore', text)
        self.assertIn('gather coal', text)
        self.assertIn('stay above water', text)


if __name__ == '__main__':
    unittest.main()
