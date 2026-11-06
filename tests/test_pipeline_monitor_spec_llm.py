import json
import unittest

from minecraftrl.pipeline.monitor_spec import SIGNALS, parse_spec
from minecraftrl.pipeline.monitor_spec_llm import (PROPOSAL_SCHEMA, SubgoalFact, check_facts, proposal_payload,
                                                   propose_spec, spec_from_proposal)
from minecraftrl.pipeline.openai_client import ResponsesClient

FACTS = (
    SubgoalFact('mine_iron_text', 'policy', 'iron_ore', 'iron_ore', 'unknown', 'mine one iron ore block'),
    SubgoalFact('craft:stone_pickaxe', 'rule', 'stone_pickaxe', None, 'known', 'craft a stone pickaxe'),
)
SYSTEM = [{'id': 'system-health', 'form': 'always', 'scope': 'goal', 'source': 'system', 'strength': 'hard',
           'predicate': {'fact': 'health', 'op': 'ge', 'value': 6.01}, 'on_violation': 'hold'}]


def subgoal(key, ending, unit, found=None, progress=(), anomaly=(), waste=()):
    return dict(key=key, ending=ending, unit=unit, found=found, progress=list(progress), anomaly=list(anomaly),
                waste=list(waste), reason='short reason')


GOOD = [subgoal('mine_iron_text', 'until_found', 'item', 'iron_ore', ['new_position_cell'], ['repeated_cycle'],
                ['moving_but_stuck']),
        subgoal('craft:stone_pickaxe', 'bounded', 'step')]
WRONG_ROLE = [dict(GOOD[0], anomaly=['looking_down']), GOOD[1]]


def reply(subgoals):
    text = json.dumps({'subgoals': subgoals})
    return {'status': 'completed', 'usage': {'input_tokens': 900, 'output_tokens': 200},
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}]}


class Transport:
    def __init__(self, *responses): self.responses = list(responses); self.requests = []
    def __call__(self, request, timeout):
        self.requests.append(json.loads(request.data)); return self.responses.pop(0)


def client(*responses):
    transport = Transport(*responses)
    return ResponsesClient('gpt-test', 'k', allow_paid=True, transport=transport, max_calls=4), transport


def propose(llm):
    return propose_spec(llm, 'iron_ore x1', FACTS, version='llm-test-1', margin=.2, constraints=SYSTEM)


class PayloadTest(unittest.TestCase):
    def test_payload_lists_every_signal_and_fact_without_tools(self):
        payload = proposal_payload('iron_ore x1', FACTS)
        text = json.dumps(payload, ensure_ascii=False)
        for name, (role, endings) in SIGNALS.items():
            self.assertIn(name, text)
        for fact in FACTS:
            self.assertIn(fact.key, text)
        self.assertNotIn('tools', payload)
        fmt = payload['text']['format']
        self.assertEqual((fmt['type'], fmt['strict']), ('json_schema', True))

    def test_schema_has_no_time_success_or_constraint_fields(self):
        item = PROPOSAL_SCHEMA['properties']['subgoals']['items']
        self.assertEqual(set(item['required']), set(item['properties']))
        for field in ('best_s', 'check_s', 'max_s', 'done', 'success', 'constraints', 'margin'):
            self.assertNotIn(field, item['properties'])
        self.assertNotIn('constraints', PROPOSAL_SCHEMA['properties'])


class ProposeTest(unittest.TestCase):
    def test_valid_proposal_becomes_checked_spec(self):
        llm, transport = client(reply(GOOD))
        result = propose(llm)
        self.assertTrue(result['ok'])
        spec = parse_spec(result['spec'])
        self.assertEqual((spec.version, spec.author, spec.margin), ('llm-test-1', 'llm:gpt-test', .2))
        self.assertEqual(spec.for_step('mine_iron_text', 'iron_ore').found, 'iron_ore')
        self.assertEqual(result['spec']['constraints'], SYSTEM)
        self.assertNotIn('found', result['spec']['subgoals']['craft:stone_pickaxe'])
        self.assertEqual(len(result['attempts']), 1)
        self.assertEqual(len(transport.requests), 1)

    def test_one_repair_round_sends_the_checker_reasons(self):
        llm, transport = client(reply(WRONG_ROLE), reply(GOOD))
        result = propose(llm)
        self.assertTrue(result['ok'])
        self.assertEqual([bool(a['reasons']) for a in result['attempts']], [True, False])
        self.assertIn('looking_down is a waste signal, not anomaly', json.dumps(transport.requests[1]))

    def test_gives_up_after_the_repair_round(self):
        llm, transport = client(reply(WRONG_ROLE), reply(WRONG_ROLE))
        result = propose(llm)
        self.assertFalse(result['ok'])
        self.assertIsNone(result['spec'])
        self.assertTrue(result['reasons'])
        self.assertEqual(len(transport.requests), 2)


class FactRulesTest(unittest.TestCase):
    def raw(self, subgoals):
        """The proposal as the LLM returns it (a list keeps duplicate keys visible)."""
        return {'subgoals': subgoals}

    def test_location_decides_the_ending(self):
        bounded_search = [subgoal('mine_iron_text', 'bounded', 'item'), GOOD[1]]
        self.assertIn('subgoal mine_iron_text: target location unknown, ending must be until_found',
                      check_facts(self.raw(bounded_search), FACTS))
        search_craft = [GOOD[0], subgoal('craft:stone_pickaxe', 'until_found', 'step', 'stone_pickaxe')]
        self.assertIn('subgoal craft:stone_pickaxe: target location known, ending must be bounded',
                      check_facts(self.raw(search_craft), FACTS))

    def test_found_must_be_the_target_block(self):
        other = [dict(GOOD[0], found='stone'), GOOD[1]]
        self.assertIn('subgoal mine_iron_text: found must be iron_ore', check_facts(self.raw(other), FACTS))

    def test_missing_extra_and_duplicate_keys(self):
        reasons = check_facts(self.raw([GOOD[0], GOOD[0], subgoal('teleport', 'bounded', 'step')]), FACTS)
        for text in ('missing subgoal craft:stone_pickaxe', 'unknown subgoal teleport', 'duplicate subgoal mine_iron_text'):
            self.assertIn(text, reasons)

    def test_good_proposal_passes_both_checks(self):
        self.assertEqual(check_facts(self.raw(GOOD), FACTS), [])
        parse_spec(spec_from_proposal(self.raw(GOOD), version='v', author='llm:x', margin=.2, constraints=[]))


def load_script():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'propose_monitor_spec.py'
    spec = importlib.util.spec_from_file_location('propose_monitor_spec_under_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class ScriptTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.human = self.root / 'human.json'
        self.human.write_text(json.dumps({'version': 'h', 'author': 'human', 'margin': .2, 'subgoals': {},
                                          'constraints': SYSTEM}), encoding='utf-8')
        self.facts = self.root / 'facts.json'
        self.facts.write_text(json.dumps({'schema': 'monitor_spec_facts_v1', 'goal': 'iron_ore x1',
                                          'constraints_from': str(self.human),
                                          'subgoals': [dict(key=f.key, kind=f.kind, target=f.target, block=f.block,
                                                            location=f.location, description=f.description)
                                                       for f in FACTS]}), encoding='utf-8')

    def args(self, out, *extra):
        return ['--facts', str(self.facts), '--version', 'llm-test-1', '--out', str(out), *extra]

    def test_iron_demo_facts_cover_the_human_spec(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        facts = json.loads((root / 'configs/monitor-specs/iron-demo-facts-v1.json').read_text(encoding='utf-8'))
        human = json.loads((root / facts['constraints_from']).read_text(encoding='utf-8'))
        self.assertEqual([f['key'] for f in facts['subgoals']], list(human['subgoals']))
        for f in facts['subgoals']:
            sub = human['subgoals'][f['key']]
            self.assertEqual(f['location'] == 'unknown', sub['ending'] == 'until_found', f['key'])
            if sub['ending'] == 'until_found': self.assertEqual(f['block'], sub['found'])

    def test_refuses_without_allow_paid(self):
        out = self.root / 'out'
        self.assertEqual(load_script().main(self.args(out)), 2)
        self.assertFalse(out.exists())

    def test_writes_spec_and_receipt_with_injected_client(self):
        out = self.root / 'out'
        llm, _ = client(reply(WRONG_ROLE), reply(GOOD))
        self.assertEqual(load_script().main(self.args(out, '--allow-paid'), client=llm), 0)
        spec = parse_spec(json.loads((out / 'spec.json').read_text(encoding='utf-8')))
        self.assertEqual((spec.author, len(spec.constraints)), ('llm:gpt-test', 1))
        receipt = json.loads((out / 'receipt.json').read_text(encoding='utf-8'))
        self.assertEqual((receipt['ok'], receipt['calls'], receipt['model']), (True, 2, 'gpt-test'))
        self.assertEqual(len(receipt['facts_sha256']), 64)
        self.assertNotIn('"k"', (out / 'receipt.json').read_text(encoding='utf-8'))
        self.assertIsNone(receipt['record_check'])
        self.assertFalse((out / 'calibrated.json').exists())

    def test_record_check_with_fit_runs_writes_calibrated_copy(self):
        from test_pipeline_monitor_replay import finished, started, stationary, write_run
        rows = stationary(200)   # pressing forward without moving: moving_but_stuck in most rows
        timeline = [started('s', 'mine_iron_text', rows[0]['t'], rows[0]['step'], block='iron_ore'),
                    finished('s', 'mine_iron_text', rows[-1]['t'] + .05, rows[-1]['step'] + 1, block='iron_ore')]
        run = write_run(self.root / 'dev' / 'A1', timeline, rows)
        out = self.root / 'out'
        llm, _ = client(reply(GOOD))
        self.assertEqual(load_script().main(self.args(out, '--allow-paid', '--fit', str(run)), client=llm), 0)
        receipt = json.loads((out / 'receipt.json').read_text(encoding='utf-8'))
        check = receipt['record_check']
        self.assertEqual([(f['key'], f['signal']) for f in check['flags']], [('mine_iron_text', 'moving_but_stuck')])
        self.assertEqual((check['limits'], check['fit']), ({'waste_max': .5, 'anomaly_max': .2}, [str(run)]))
        calibrated = parse_spec(json.loads((out / 'calibrated.json').read_text(encoding='utf-8')))
        self.assertEqual((calibrated.version, calibrated.for_step('mine_iron_text', 'iron_ore').waste), ('llm-test-1-cal', ()))
        self.assertEqual(parse_spec(json.loads((out / 'spec.json').read_text(encoding='utf-8')))
                         .for_step('mine_iron_text', 'iron_ore').waste, ('moving_but_stuck',))

    def test_invalid_after_repair_exits_3_with_receipt_only(self):
        out = self.root / 'out'
        llm, _ = client(reply(WRONG_ROLE), reply(WRONG_ROLE))
        self.assertEqual(load_script().main(self.args(out, '--allow-paid'), client=llm), 3)
        self.assertFalse((out / 'spec.json').exists())
        self.assertFalse(json.loads((out / 'receipt.json').read_text(encoding='utf-8'))['ok'])


if __name__ == '__main__':
    unittest.main()
