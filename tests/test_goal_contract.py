import copy
import json
from pathlib import Path
import tempfile
import unittest

from minecraftrl.goal_contract import compile_conditions, digest, main, read_json, request


def context(target='a tree', goal='collect wood'):
    return dict(schema=1, root_goal=goal, active_goal=goal, constraints=[],
        actions={'find': 'find', 'approach': 'approach'}, targets={'resource': target},
        outcomes={'located': {'description': target + ' is observed', 'predicate_ref': 'target_visible'}},
        states=['search', 'approach'], terminal_states=['complete'])


def proposal(ctx):
    return dict(schema=1, context_sha256=digest(ctx), status='candidate', reason='',
        conditions=[dict(id='seek', action_ref='find', target_ref='resource', outcome_ref='located')],
        state_bindings=[dict(state=state, condition_id='seek') for state in ctx['states']])


class GoalContractTests(unittest.TestCase):
    def test_search_preserves_target_and_purpose_across_unrelated_domains(self):
        for target, goal in [('a tree', 'collect wood'), ('iron ore', 'obtain iron'),
                             ('the designated base', 'return home')]:
            with self.subTest(target=target):
                ctx = context(target, goal)
                result = compile_conditions(ctx, proposal(ctx))
                prompt = result['conditions']['seek']['prompt']
                self.assertIn('find ' + target, prompt)
                self.assertIn('Purpose: ' + goal, prompt)
                self.assertIn(ctx['outcomes']['located']['description'], prompt)
                self.assertFalse(result['execution_authorized'])
                self.assertFalse(result['capability_verified'])
                self.assertFalse(result['tokenizer_checked'])

    def test_parent_goal_constraints_and_predicate_are_not_llm_editable(self):
        ctx = context()
        ctx.update(root_goal='craft a tool', constraints=['avoid lava', 'do not discard materials'])
        p = proposal(ctx)
        result = compile_conditions(ctx, p)
        condition = result['conditions']['seek']
        for value in [ctx['root_goal'], ctx['active_goal'], *ctx['constraints']]:
            self.assertIn(value, condition['prompt'])
        self.assertEqual(condition['predicate_ref'], 'target_visible')
        for field in ('prompt', 'root_goal', 'constraints', 'predicate_ref', 'capability_verified'):
            tampered = copy.deepcopy(p)
            tampered['conditions'][0][field] = 'go explore'
            with self.subTest(field=field), self.assertRaises(ValueError):
                compile_conditions(ctx, tampered)

    def test_stage_change_can_keep_exact_same_condition(self):
        ctx = context()
        result = compile_conditions(ctx, proposal(ctx))
        self.assertEqual(result['state_bindings'], {'search': 'seek', 'approach': 'seek'})
        self.assertEqual(len(result['conditions']), 1)
        self.assertNotIn('complete', result['state_bindings'])

    def test_movement_and_interaction_keep_destination_or_affected_object(self):
        cases = [('move to', 'the designated base', 'store the collected resources', 'the base is reached'),
                 ('fill', 'the empty bucket', 'obtain water', 'a water bucket is present in inventory')]
        for action, target, goal, outcome in cases:
            ctx = context(target, goal)
            ctx['actions']['find'] = action
            ctx['outcomes']['located']['description'] = outcome
            result = compile_conditions(ctx, proposal(ctx))
            prompt = result['conditions']['seek']['prompt']
            for required in (action, target, goal, outcome):
                self.assertIn(required, prompt)

    def test_context_revision_change_rejects_stale_proposal(self):
        ctx = context()
        p = proposal(ctx)
        ctx['active_goal'] = 'obtain stone'
        with self.assertRaisesRegex(ValueError, 'revision'):
            compile_conditions(ctx, p)

    def test_missing_or_unknown_target_action_and_outcome_are_rejected(self):
        ctx = context()
        for field in ('action_ref', 'target_ref', 'outcome_ref'):
            for value in (None, '', 'invented', []):
                p = proposal(ctx)
                if value is None:
                    del p['conditions'][0][field]
                else:
                    p['conditions'][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    compile_conditions(ctx, p)

    def test_bindings_must_be_total_unique_and_nonterminal(self):
        ctx = context()
        for bindings in ([], [{'state': 'search', 'condition_id': 'seek'}],
                         [{'state': 'complete', 'condition_id': 'seek'}],
                         [{'state': 'search', 'condition_id': 'missing'}],
                         [{'state': 'search', 'condition_id': 'seek'}] * 2):
            p = proposal(ctx)
            p['state_bindings'] = bindings
            with self.subTest(bindings=bindings), self.assertRaises(ValueError):
                compile_conditions(ctx, p)

    def test_duplicate_and_unused_conditions_are_rejected(self):
        ctx = context()
        for name in ('seek', 'unused'):
            p = proposal(ctx)
            p['conditions'].append(dict(p['conditions'][0], id=name))
            with self.assertRaises(ValueError):
                compile_conditions(ctx, p)

    def test_unsupported_is_explicit_and_has_no_fallback(self):
        ctx = context()
        p = dict(proposal(ctx), status='unsupported', reason='No admitted target observation.',
                 conditions=[], state_bindings=[])
        self.assertEqual(compile_conditions(ctx, p)['conditions'], {})
        p['reason'] = ''
        with self.assertRaises(ValueError):
            compile_conditions(ctx, p)
        p.update(reason='missing capability', conditions=proposal(ctx)['conditions'])
        with self.assertRaises(ValueError):
            compile_conditions(ctx, p)

    def test_overlong_prompt_is_rejected_instead_of_dropping_context(self):
        ctx = context()
        with self.assertRaisesRegex(ValueError, 'never truncate'):
            compile_conditions(ctx, proposal(ctx), max_prompt_chars=10)

    def test_invalid_context_cannot_form_an_llm_request(self):
        for field, value in [('active_goal', ''), ('root_goal', '\n'), ('schema', True),
                             ('targets', {}), ('states', ['search', 'search']),
                             ('terminal_states', ['search']), ('constraints', 'avoid lava')]:
            ctx = context()
            ctx[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                request(ctx)

    def test_request_schema_is_bound_and_does_not_mutate_inputs(self):
        ctx = context()
        original = copy.deepcopy(ctx)
        packet = request(ctx)
        self.assertEqual(packet['context_sha256'], digest(ctx))
        self.assertFalse(packet['response_schema']['additionalProperties'])
        compile_conditions(ctx, proposal(ctx))
        packet['context']['targets']['resource'] = 'different'
        self.assertEqual(ctx, original)

    def test_cli_round_trip_no_overwrite_and_invalid_input_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            c, p, r, out = [folder/name for name in ('context.json', 'proposal.json', 'request.json', 'out.json')]
            ctx = context('a water source', 'fill a bucket')
            c.write_text(json.dumps(ctx), encoding='utf-8')
            main(['request', '--context', str(c), '--output', str(r)])
            packet = read_json(r)
            candidate = proposal(ctx)
            candidate['context_sha256'] = packet['context_sha256']
            p.write_text(json.dumps(candidate), encoding='utf-8')
            argv = ['compile', '--context', str(c), '--proposal', str(p), '--output', str(out)]
            main(argv)
            saved = out.read_bytes()
            self.assertIn('a water source', read_json(out)['conditions']['seek']['prompt'])
            with self.assertRaises(FileExistsError):
                main(argv)
            self.assertEqual(out.read_bytes(), saved)
            candidate['conditions'][0]['target_ref'] = 'unknown'
            p.write_text(json.dumps(candidate), encoding='utf-8')
            invalid = folder/'invalid.json'
            with self.assertRaises(ValueError):
                main(argv[:-1] + [str(invalid)])
            self.assertFalse(invalid.exists())

    def test_duplicate_json_keys_are_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'input.json'
            path.write_text('{"target_ref":"resource","target_ref":"other"}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                read_json(path)


if __name__ == '__main__':
    unittest.main()
