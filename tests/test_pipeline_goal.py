import json
import unittest

from minecraftrl.pipeline.goal import GOAL_SCHEMA, parse_goal
from minecraftrl.pipeline.openai_client import ResponsesClient


def reply(item, count, reason=None):
    text = json.dumps(dict(item=item, count=count, unsupported_reason=reason))
    return {'status': 'completed', 'usage': {'input_tokens': 300, 'output_tokens': 20},
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}]}


class Transport:
    def __init__(self, response): self.response = response; self.requests = []
    def __call__(self, request, timeout): self.requests.append(json.loads(request.data)); return self.response


def client(response):
    transport = Transport(response)
    return ResponsesClient('gpt-test', 'k', allow_paid=True, transport=transport, max_calls=1), transport


class GoalTests(unittest.TestCase):
    def test_iron_request_becomes_the_allowed_goal(self):
        llm, transport = client(reply('iron_ore', None))
        goal = parse_goal(llm, '철 캐줘', allowed={'iron_ore': (1, 64)}, default_count=1)
        self.assertEqual((goal['item'], goal['count'], goal['count_source']), ('iron_ore', 1, 'caller_success_criterion'))
        sent = transport.requests[0]
        self.assertNotIn('tools', sent)  # no web search for reading a request
        self.assertIn('철 캐줘', json.dumps(sent['input'], ensure_ascii=False))
        self.assertEqual(sent['text']['format']['schema'], GOAL_SCHEMA)

    def test_stated_count_is_kept_within_range(self):
        llm, _ = client(reply('iron_ore', 3))
        self.assertEqual(parse_goal(llm, '철광석 3개 캐줘', allowed={'iron_ore': (1, 64)}, default_count=1)['count'], 3)
        llm, _ = client(reply('iron_ore', 100))
        with self.assertRaisesRegex(ValueError, 'unsupported_goal'):
            parse_goal(llm, '철광석 100개', allowed={'iron_ore': (1, 64)}, default_count=1)

    def test_unsupported_item_is_rejected_by_the_program(self):
        llm, _ = client(reply('diamond', 1))
        with self.assertRaisesRegex(ValueError, 'unsupported_goal'):
            parse_goal(llm, '다이아 캐줘', allowed={'iron_ore': (1, 64)}, default_count=1)
        llm, _ = client(reply(None, None, 'not a mining request'))
        with self.assertRaisesRegex(ValueError, 'unsupported_goal'):
            parse_goal(llm, '노래 불러줘', allowed={'iron_ore': (1, 64)}, default_count=1)

    def test_prompt_treats_mining_requests_as_obtaining_the_item(self):
        # Round 3 E03 (2026-11-05): gpt-5.4 refused "철 캐줘" once ("an action request, not an item in the inventory").
        from minecraftrl.pipeline.goal import GOAL_PROMPT
        self.assertIn('mine', GOAL_PROMPT); self.assertIn('is a request to obtain', GOAL_PROMPT)

    def test_missing_count_without_a_caller_criterion_needs_clarification(self):
        llm, _ = client(reply('iron_ore', None))
        with self.assertRaisesRegex(ValueError, 'needs_clarification'):
            parse_goal(llm, '철 캐줘', allowed={'iron_ore': (1, 64)})


if __name__ == '__main__':
    unittest.main()
