import copy
import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.monitor_spec import SpecError, constraint_priority, load_spec, parse_spec

ROOT = Path(__file__).resolve().parents[1]
VALID = {
    'version': 'test-v1', 'author': 'human', 'margin': 0.2,
    'subgoals': {
        'mine_iron_text': {'ending': 'until_found', 'unit': 'item', 'found': 'iron_ore',
                           'anomaly': ['no_progress', 'stationary_repetition'], 'waste': ['revisiting_area']},
        'craft': {'ending': 'bounded', 'unit': 'step'},
    },
    'constraints': [
        {'id': 'health', 'form': 'always', 'scope': 'goal', 'source': 'system', 'strength': 'hard',
         'predicate': {'fact': 'health', 'op': 'ge', 'value': 6.01}, 'on_violation': 'hold'},
    ],
}


def edited(change):
    raw = copy.deepcopy(VALID); change(raw); return raw


def reasons(raw):
    try: parse_spec(raw)
    except SpecError as e: return e.reasons
    return []


def constraint(**fields):
    c = {'id': 'c1', 'form': 'always', 'scope': 'goal', 'source': 'human', 'strength': 'hard',
         'predicate': {'fact': 'health', 'op': 'ge', 'value': 10}, 'on_violation': 'warn'}
    c.update(fields); return c


def with_constraints(*cs):
    return edited(lambda r: r.__setitem__('constraints', list(cs)))


class MonitorSpecTest(unittest.TestCase):
    def assertRejected(self, raw, fragment):
        found = reasons(raw)
        self.assertTrue(any(fragment in r for r in found), f'{fragment!r} not in {found}')

    def test_valid_spec_parses(self):
        spec = parse_spec(VALID)
        self.assertEqual(spec.for_step('mine_iron_text', 'iron_ore').ending, 'until_found')
        self.assertEqual(spec.for_step('craft', 'stick').key, 'craft')
        raw = edited(lambda r: r['subgoals'].__setitem__('craft:stone_pickaxe', {'ending': 'bounded', 'unit': 'step'}))
        self.assertEqual(parse_spec(raw).for_step('craft', 'stone_pickaxe').key, 'craft:stone_pickaxe')
        self.assertIsNone(spec.for_step('equip', 'stone_pickaxe'))

    def test_all_reasons_collected(self):
        def change(r):
            r['subgoals']['craft']['ending'] = 'forever'
            r['subgoals']['mine_iron_text']['anomaly'] = ['teleported']
        self.assertGreaterEqual(len(reasons(edited(change))), 2)

    def test_unknown_signal_rejected(self):
        self.assertRejected(edited(lambda r: r['subgoals']['mine_iron_text'].__setitem__('anomaly', ['teleported'])),
                            'unknown signal')

    def test_role_mismatch_rejected(self):
        self.assertRejected(edited(lambda r: r['subgoals']['mine_iron_text'].__setitem__('waste', ['no_progress'])),
                            'is a anomaly signal')

    def test_signal_not_allowed_for_ending(self):
        self.assertRejected(edited(lambda r: r['subgoals']['craft'].__setitem__('anomaly', ['repeated_cycle'])),
                            'not allowed for bounded')

    def test_until_found_requires_found(self):
        self.assertRejected(edited(lambda r: r['subgoals']['mine_iron_text'].pop('found')), 'requires found')

    def test_bounded_forbids_found(self):
        self.assertRejected(edited(lambda r: r['subgoals']['craft'].__setitem__('found', 'log')), 'bounded has no found')

    def test_time_numbers_rejected(self):
        for field in ('best_s', 'check_s', 'expected_s', 'max_s'):
            with self.subTest(field=field):
                self.assertRejected(edited(lambda r: r['subgoals']['craft'].__setitem__(field, 30)),
                                    'times come from records')

    def test_success_fields_rejected(self):
        for field in ('done', 'success'):
            with self.subTest(field=field):
                self.assertRejected(edited(lambda r: r['subgoals']['craft'].__setitem__(field, 'inventory >= 1')),
                                    'caller owns success')

    def test_llm_constraint_rejected(self):
        self.assertRejected(with_constraints(constraint(source='llm')), '2026-11-05')

    def test_unobservable_fact_unsupported(self):
        self.assertRejected(with_constraints(constraint(predicate={'fact': 'noise', 'op': 'le', 'value': 0})),
                            'unsupported')

    def test_deadline_requires_by(self):
        self.assertRejected(with_constraints(constraint(form='deadline')), 'deadline requires by')
        by = {'fact': 'day_time', 'op': 'le', 'value': 12000}
        self.assertEqual(reasons(with_constraints(constraint(form='deadline', by=by))), [])

    def test_conditional_requires_during(self):
        self.assertRejected(with_constraints(constraint(form='conditional')), 'conditional requires during')
        during = {'fact': 'inventory', 'item': 'dirt', 'op': 'ge', 'value': 1}
        self.assertEqual(reasons(with_constraints(constraint(form='conditional', during=during))), [])

    def test_inventory_fact_requires_item(self):
        self.assertRejected(with_constraints(constraint(predicate={'fact': 'inventory', 'op': 'ge', 'value': 1})),
                            'inventory requires item')

    def test_contradicting_hard_constraints_rejected(self):
        at_least = constraint(id='a', predicate={'fact': 'inventory', 'item': 'iron_ore', 'op': 'ge', 'value': 1})
        at_most = constraint(id='b', predicate={'fact': 'inventory', 'item': 'iron_ore', 'op': 'le', 'value': 0})
        self.assertRejected(with_constraints(at_least, at_most), 'contradict')
        self.assertEqual(reasons(with_constraints(at_least, dict(at_most, strength='soft'))), [])

    def test_system_and_plan_constraints_must_be_hard(self):
        for source in ('system', 'plan'):
            with self.subTest(source=source):
                self.assertRejected(with_constraints(constraint(source=source, strength='soft')), 'must be hard')

    def test_duplicate_json_key_rejected(self):
        text = '{"version": "a", "version": "b", "author": "human", "margin": 0.2, "subgoals": {}, "constraints": []}'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'spec.json'; path.write_text(text, encoding='utf-8')
            with self.assertRaises(SpecError) as caught: load_spec(path)
        self.assertTrue(any('duplicate key' in r for r in caught.exception.reasons))

    def test_same_checks_for_human_and_llm_author(self):
        bad = edited(lambda r: r['subgoals']['craft'].__setitem__('best_s', 30))
        for raw in (VALID, bad):
            human = edited(lambda r: r.update(raw, author='human'))
            llm = edited(lambda r: r.update(raw, author='llm:gpt-5.4'))
            self.assertEqual(reasons(human), reasons(llm))
        self.assertEqual(reasons(edited(lambda r: r.update(author='llm:gpt-5.4'))), [])

    def test_priority_order(self):
        cs = [constraint(id='soft', strength='soft'), constraint(id='plan', source='plan'),
              constraint(id='human'), constraint(id='system', source='system')]
        spec = parse_spec(with_constraints(*cs))
        self.assertEqual([c.id for c in sorted(spec.constraints, key=constraint_priority)],
                         ['system', 'human', 'plan', 'soft'])

    def test_margin_range(self):
        for margin in (-0.1, 5.1, True, 'wide'):
            with self.subTest(margin=margin):
                self.assertRejected(edited(lambda r: r.__setitem__('margin', margin)), 'margin')
        for margin in (0, 5):
            self.assertEqual(reasons(edited(lambda r: r.__setitem__('margin', margin))), [])

    def test_shipped_iron_spec_parses(self):
        spec = load_spec(ROOT / 'configs' / 'monitor-specs' / 'iron-demo-v1.json')
        self.assertEqual(spec.for_step('wood_visual', 'log').ending, 'until_found')
        self.assertEqual(spec.for_step('mine_iron_text', 'iron_ore').ending, 'until_found')
        self.assertEqual(spec.for_step('craft', 'wooden_pickaxe').key, 'craft:wooden_pickaxe')
        self.assertEqual(spec.for_step('climb_out_text', 'height').unit, 'block')
        self.assertEqual(spec.margin, 0.2)


if __name__ == '__main__':
    unittest.main()
