"""Versioned concrete action rendering; RM metadata stays outside MineCLIP text."""
from copy import deepcopy
import hashlib

from .goal_contract import compile_conditions, digest, fields


def compile_action_conditions(context, proposal, rendering):
    """Caller owns purpose references; authors select explicit action/target clauses.

The old compiler remains unchanged for historical P2 reproduction. This renderer
does not claim that including a constraint makes STEVE-1 capable of obeying it.
"""
    result = compile_conditions(context, proposal)
    if result['status'] != 'candidate':
        raise ValueError('unsupported action conditions')
    fields(rendering, 'schema context_sha256 purpose clauses', 'action rendering')
    if rendering['schema'] != 'goal_actions_v2' or rendering['context_sha256'] != digest(context):
        raise ValueError('stale action rendering')

    def pair(clause):
        fields(clause, 'action_ref target_ref', 'action clause')
        if (not isinstance(clause['action_ref'], str) or not isinstance(clause['target_ref'], str)
                or clause['action_ref'] not in context['actions'] or clause['target_ref'] not in context['targets']):
            raise ValueError('unknown action or target')
        return clause['action_ref'], clause['target_ref']

    purpose = rendering['purpose']
    if not isinstance(purpose, list) or not 1 <= len(purpose) <= 4:
        raise ValueError('caller must provide concrete clauses for the active and root purpose')
    required = {pair(clause) for clause in purpose}
    if not isinstance(rendering['clauses'], dict) or set(rendering['clauses']) != set(result['conditions']):
        raise ValueError('every condition needs an explicit action rendering')
    for key, condition in result['conditions'].items():
        clauses = rendering['clauses'][key]
        if not isinstance(clauses, list) or not 1 <= len(clauses) <= 8:
            raise ValueError('bounded nonempty concrete clauses required')
        pairs = [pair(clause) for clause in clauses]
        primary = (condition['action_ref'], condition['target_ref'])
        if primary not in pairs or not required.issubset(pairs):
            raise ValueError('policy text cannot omit its action, target or purpose')
        prompt = ', '.join(context['actions'][a] + ' ' + context['targets'][t] for a, t in pairs)
        if context['constraints']:
            prompt += '. ' + '; '.join(context['constraints'])
        if len(prompt) > 512:
            raise ValueError('action prompt too long; never silently truncate')
        condition.update(prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    result.update(schema='goal_condition_actions_v2', rendering=deepcopy(rendering),
                  metadata_location='context and predicate references; not prose fields in MineCLIP text')
    return result
