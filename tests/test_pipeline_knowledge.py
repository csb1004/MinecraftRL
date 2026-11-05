import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.kb import KnowledgeBase
from minecraftrl.pipeline.knowledge import KNOWLEDGE_PROMPT, acquire, hypothesize, store, validate_candidates, validate_hypothesis
from minecraftrl.pipeline.openai_client import ResponsesClient, url_citations

WIKI = 'https://minecraft.wiki/w/Iron_Ore'


def candidate(subject='iron_ore', relation='harvest_requires', obj=None, conditions=None, urls=(WIKI,)):
    return dict(subject=subject, relation=relation, object_json=json.dumps(obj or {'tool_class': 'pickaxe', 'min_tier': 'stone'}),
                conditions_json=json.dumps(conditions or {}), source_urls=list(urls), excerpt='Requires a stone pickaxe or better.')


def response(claims, urls=(WIKI,)):
    text = json.dumps({'claims': claims})
    return {'status': 'completed', 'output': [{'type': 'web_search_call', 'status': 'completed'}, {'type': 'message', 'content': [
        {'type': 'output_text', 'text': text, 'annotations': [{'type': 'url_citation', 'url': u} for u in urls]}]}],
        'usage': {'input_tokens': 900, 'output_tokens': 300}}


class FakeTransport:
    def __init__(self, reply=None, fail=False): self.reply = reply; self.fail = fail; self.requests = []
    def __call__(self, request, timeout):
        self.requests.append(json.loads(request.data)); self.headers = dict(request.headers)
        if self.fail: raise OSError('network down')
        return self.reply


class ClientTests(unittest.TestCase):
    def test_client_requires_paid_permission(self):
        with self.assertRaises(PermissionError): ResponsesClient('m', 'k', allow_paid=False, max_calls=1)

    def test_budget_consumed_even_on_failure_no_retry_without_budget(self):
        transport = FakeTransport(fail=True); client = ResponsesClient('m', 'k', allow_paid=True, transport=transport, max_calls=1)
        with self.assertRaises(RuntimeError): client.create({'input': 'x'})
        self.assertEqual((client.remaining, len(transport.requests)), (0, 1))
        with self.assertRaises(RuntimeError) as caught: client.create({'input': 'x'})
        self.assertIn('budget', str(caught.exception))

    def test_network_failure_is_retried_once_within_budget(self):
        # Evaluation round 1, E02 (2026-11-05): one transport failure while reading "철 캐줘" ended the run.
        reply = response([candidate()]); attempts = []
        def flaky(request, timeout):
            attempts.append(1)
            if len(attempts) == 1: raise OSError('connection reset')
            return reply
        client = ResponsesClient('m', 'k', allow_paid=True, transport=flaky, max_calls=3, retry_wait_s=0.)
        self.assertEqual(client.create({'input': 'x'}), reply)
        self.assertEqual((len(attempts), client.remaining, [c['ok'] for c in client.calls]), (2, 1, [False, True]))
        self.assertEqual(client.calls[0]['error'], 'OSError')

    def test_rate_limit_waits_longer_before_the_retry(self):
        import urllib.error
        attempts, waits = [], []
        reply = response([candidate()])
        def limited(request, timeout):
            attempts.append(1)
            if len(attempts) == 1: raise urllib.error.HTTPError('u', 429, 'rate limited', {'Retry-After': '7'}, None)
            return reply
        client = ResponsesClient('m', 'k', allow_paid=True, transport=limited, max_calls=3, retry_wait_s=0., sleep=waits.append)
        self.assertEqual(client.create({'input': 'x'}), reply)
        self.assertEqual((waits, client.calls[0]['status']), ([7.], 429))

    def test_client_errors_are_not_retried(self):
        import urllib.error
        attempts = []
        def bad(request, timeout):
            attempts.append(1); raise urllib.error.HTTPError('u', 400, 'bad request', {}, None)
        client = ResponsesClient('m', 'k', allow_paid=True, transport=bad, max_calls=3, retry_wait_s=0.)
        with self.assertRaises(RuntimeError): client.create({'input': 'x'})
        self.assertEqual((len(attempts), client.calls[0]['status']), (1, 400))

    def test_request_is_not_stored_and_key_only_in_header(self):
        transport = FakeTransport(response([])); ResponsesClient('m', 'sk-secret', allow_paid=True, transport=transport, max_calls=1).create({'input': 'x'})
        self.assertIs(transport.requests[0]['store'], False); self.assertNotIn('sk-secret', json.dumps(transport.requests[0]))

    def test_citations_extracted_from_annotations(self):
        self.assertEqual(url_citations(response([], urls=(WIKI, 'https://a.example/b'))), {WIKI, 'https://a.example/b'})

    def test_sources_of_the_search_call_count_as_consulted(self):
        # Real gpt-5.4 call (2026-11-04): with a strict JSON format the message carries no url_citation annotations,
        # so the sources the web_search tool actually returned are the evidence that a URL was consulted.
        reply = response([], urls=())
        reply['output'][0]['action'] = {'type': 'search', 'query': 'iron ore', 'sources': [{'type': 'url', 'url': WIKI}]}
        self.assertEqual(url_citations(reply), {WIKI})


class KnowledgeTests(unittest.TestCase):
    def test_payload_enables_web_search_and_states_version(self):
        transport = FakeTransport(response([candidate()]))
        client = ResponsesClient('m', 'k', allow_paid=True, transport=transport, max_calls=1)
        raw, meta = acquire(client, {'goal': 'iron_ore', 'subject': 'iron_ore', 'relation': 'harvest_requires'}, {'game_version': '1.16.5'})
        payload = transport.requests[0]
        self.assertEqual(payload['tools'], [{'type': 'web_search'}]); self.assertIn('1.16.5', json.dumps(payload['input']))
        self.assertEqual(payload['include'], ['web_search_call.action.sources'])
        self.assertEqual(len(raw), 1); self.assertEqual(meta['citations'], [WIKI]); self.assertEqual(meta['usage']['output_tokens'], 300)

    def test_prompt_requests_harvest_requirement_for_each_chain_block(self):
        for phrase in ('every block', 'by hand', 'recipe', 'harvest_requires'):
            self.assertIn(phrase, KNOWLEDGE_PROMPT)

    def test_uncited_url_rejected(self):
        accepted, rejected = validate_candidates([candidate(urls=('https://elsewhere.example/x',))], {WIKI})
        self.assertEqual((accepted, rejected[0]['reason']), ([], 'uncited_source'))

    def test_unknown_condition_key_and_relation_rejected(self):
        bad = [candidate(conditions={'luck': 'high'}), candidate(relation='teleports_to'), candidate(urls=())]
        self.assertEqual([r['reason'] for r in validate_candidates(bad, {WIKI})[1]], ['unknown_condition_key', 'unknown_relation', 'no_source'])

    def test_object_shape_checked_per_relation(self):
        bad = [candidate(obj={'tool_class': 'pickaxe', 'min_tier': 'adamant'}), candidate(relation='recipe', obj={'count': 1})]
        self.assertEqual([r['reason'] for r in validate_candidates(bad, {WIKI})[1]], ['bad_object', 'bad_object'])

    def test_tool_class_with_hand_tier_is_ambiguous_and_rejected(self):
        # Evaluation round 2, E03: gpt-5.4 wrote stone {pickaxe, hand} meaning "any pickaxe"; the program reads hand as
        # "no tool needed", so the guard let bare hands break stone without a drop.
        bad = candidate(subject='stone', obj={'tool_class': 'pickaxe', 'min_tier': 'hand'})
        good = candidate(subject='log', obj={'tool_class': None, 'min_tier': 'hand'})
        accepted, rejected = validate_candidates([bad, good], {WIKI})
        self.assertEqual(([a['subject'] for a in accepted], [r['reason'] for r in rejected]), (['log'], ['bad_object']))
        self.assertIn('lowest tier', KNOWLEDGE_PROMPT); self.assertIn('tool_class null', KNOWLEDGE_PROMPT)

    def test_escape_by_object_checked(self):
        good = candidate(subject='vertical_pit', relation='escape_by', obj={'method': 'pillar_up', 'blocks_per_level': 1})
        bad = [candidate(subject='vertical_pit', relation='escape_by', obj={'method': 'teleport', 'blocks_per_level': 1}),
               candidate(subject='vertical_pit', relation='escape_by', obj={'method': 'pillar_up', 'blocks_per_level': -1}),
               candidate(subject='vertical_pit', relation='escape_by', obj={'method': 'pillar_up'})]
        accepted, rejected = validate_candidates([good] + bad, {WIKI})
        self.assertEqual(len(accepted), 1); self.assertEqual([r['reason'] for r in rejected], ['bad_object'] * 3)

    def test_prompt_explains_the_escape_relation(self):
        self.assertIn('escape_by', KNOWLEDGE_PROMPT); self.assertIn('vertical_pit', KNOWLEDGE_PROMPT)
        self.assertIn('pillar_up', KNOWLEDGE_PROMPT)

    def test_hypothesis_needs_an_observable_cited_condition(self):
        good = {'condition_json': '{"game_version": {"gte": "1.17"}}', 'source_urls': [WIKI], 'excerpt': 'Since 1.17 iron ore drops raw iron.'}
        self.assertEqual(validate_hypothesis(good, {WIKI})['condition'], {'game_version': {'gte': '1.17'}})
        for bad, reason in ((dict(good, condition_json='{"luck": "high"}'), 'unknown_condition_key'),
                            (dict(good, source_urls=['https://elsewhere.example']), 'uncited_source'),
                            (dict(good, condition_json='null'), 'no_condition'), (dict(good, condition_json='{'), 'bad_json')):
            self.assertEqual(validate_hypothesis(bad, {WIKI})['reason'], reason)

    def test_hypothesis_tool_must_be_a_real_tool(self):
        # Evaluation round 2, E04: gpt-5.4 proposed {"tool": "*", "enchantment": "Silk Touch"} (also backwards: raw iron
        # drops without Silk Touch in 1.17+). A wildcard tool is not an observable condition.
        raw = {'condition_json': '{"tool": "*", "enchantment": "Silk Touch"}', 'source_urls': [WIKI], 'excerpt': 'x'}
        self.assertEqual(validate_hypothesis(raw, {WIKI})['reason'], 'bad_condition')
        ok = {'condition_json': '{"tool": "stone_pickaxe"}', 'source_urls': [WIKI], 'excerpt': 'x'}
        self.assertEqual(validate_hypothesis(ok, {WIKI})['condition'], {'tool': 'stone_pickaxe'})

    def test_hypothesize_states_claim_observation_and_version(self):
        reply = {'status': 'completed', 'usage': {}, 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'annotations': [
            {'type': 'url_citation', 'url': WIKI}], 'text': json.dumps({'condition_json': '{"game_version": {"gte": "1.17"}}',
                                                                         'source_urls': [WIKI], 'excerpt': 'x'})}]}]}
        transport = FakeTransport(reply)
        client = ResponsesClient('gpt-test', 'k', allow_paid=True, transport=transport, max_calls=1)
        claim = {'subject': 'iron_ore', 'relation': 'drops', 'object': 'raw_iron', 'conditions': {'game_version': '1.16.5'}}
        raw, meta = hypothesize(client, claim, {'broke': 'iron_ore', 'gained': {'iron_ore': 1}}, {'game_version': '1.16.5'})
        sent = json.loads(transport.requests[0]['input'][1]['content'])
        self.assertEqual((sent['claim']['object'], sent['observation']['gained'], sent['game_version']), ('raw_iron', {'iron_ore': 1}, '1.16.5'))
        self.assertEqual(transport.requests[0]['tools'], [{'type': 'web_search'}])
        self.assertEqual(validate_hypothesis(raw, meta['citations'])['condition'], {'game_version': {'gte': '1.17'}})

    def test_store_adds_web_and_llm_evidence(self):
        with tempfile.TemporaryDirectory() as d, KnowledgeBase(Path(d) / 'kb.sqlite') as kb:
            accepted, _ = validate_candidates([candidate()], {WIKI})
            ids = store(kb, accepted, model='gpt-test', run_id='r1')
            claim, = kb.query('iron_ore', 'harvest_requires', context={'game_version': '1.16.5'})
            self.assertEqual((ids, claim.status, claim.support), ([claim.id], 'proposed', 2))


if __name__ == '__main__':
    unittest.main()
