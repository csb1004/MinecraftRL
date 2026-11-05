"""Turn the user's request ("철 캐줘") into a goal the program accepts (spec 2 scene 1, plan Task 8 parse_goal).

The LLM only proposes the target and count; the program checks them against the registered goals. A count the
request does not state comes from the caller's fixed success criterion (iron ore >= 1), never from the LLM.
"""
import json

from .openai_client import output_json, usage

GOAL_PROMPT = '''You read a player's Minecraft request and name the item the player wants in their inventory.
A request to mine, dig, chop, collect, gather or get something is a request to obtain that item (e.g. "mine iron" ->
the iron item the mining yields). Use plain item ids without the minecraft: prefix. The game version is 1.16.5, where mining iron ore yields the iron_ore
block item. Give count only if the request states a number; otherwise null. If the request is not about obtaining an
item, set item to null and explain briefly in unsupported_reason.'''

GOAL_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['item', 'count', 'unsupported_reason'],
               'properties': {'item': {'type': ['string', 'null']}, 'count': {'type': ['integer', 'null']},
                              'unsupported_reason': {'type': ['string', 'null']}}}


def parse_goal(client, utterance, *, allowed, default_count=None):
    """-> dict(item, count, count_source, proposal, usage). ValueError('unsupported_goal'|'needs_clarification')."""
    response = client.create({
        'input': [{'role': 'system', 'content': GOAL_PROMPT}, {'role': 'user', 'content': utterance}],
        'text': {'format': {'type': 'json_schema', 'name': 'goal', 'strict': True, 'schema': GOAL_SCHEMA}}})
    proposal = output_json(response)
    item, count = proposal.get('item'), proposal.get('count')
    if item not in allowed: raise ValueError('unsupported_goal: %s' % json.dumps(proposal, ensure_ascii=False))
    source = 'request'
    if count is None:
        if default_count is None: raise ValueError('needs_clarification: the request states no count')
        count, source = default_count, 'caller_success_criterion'
    low, high = allowed[item]
    if type(count) is not int or not low <= count <= high: raise ValueError('unsupported_goal: count %r outside %d..%d' % (count, low, high))
    return dict(item=item, count=count, count_source=source, proposal=proposal, usage=usage(response))
