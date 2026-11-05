"""Object-independent authoring rules for policy conditions, not an executor.

Human and future LLM proposals share the same compiler. The caller owns the
context; a valid proposal is still a candidate, never a verified policy skill.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


INSTRUCTIONS = """Propose goal conditions using only the supplied response schema and context.
Every condition needs an action, a concrete target, and an expected observable outcome.
For search, name what must be found; for movement, name the destination; for interaction,
name the affected object and intended change. Never replace purposeful search with aimless exploration.
The compiler preserves the active goal, root goal, and constraints in every instruction.
Use only supplied references. Do not invent facts, coordinates, predicates, skills, or success claims.
Multiple RM states may share one condition; an observation-stage change need not change the instruction.
Terminal states receive no condition. Return unsupported if the context cannot express the intent.
Do not add free-form policy prompts. Structural validity does not prove semantic relevance or capability.
"""


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(label + ': missing or unexpected fields')


def text(value, label, limit=200):
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or len(value) > limit or any(ord(c) < 32 for c in value)):
        raise ValueError(label + ': expected bounded nonempty single-line text')


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', value):
        raise ValueError('invalid reference identifier')


def validate_context(context):
    fields(context, 'schema root_goal active_goal constraints actions targets outcomes states terminal_states', 'context')
    if type(context['schema']) is not int or context['schema'] != 1:
        raise ValueError('unsupported context schema')
    for key in ('root_goal', 'active_goal'):
        text(context[key], key)
    constraints = context['constraints']
    if not isinstance(constraints, list) or len(constraints) > 8:
        raise ValueError('invalid constraints')
    for item in constraints:
        text(item, 'constraint')
    for key in ('actions', 'targets', 'outcomes'):
        registry = context[key]
        if not isinstance(registry, dict) or not 1 <= len(registry) <= 64:
            raise ValueError('invalid ' + key + ' registry')
        for ref, item in registry.items():
            identifier(ref)
            if key == 'outcomes':
                fields(item, 'description predicate_ref', 'outcome')
                text(item['description'], 'outcome description')
                identifier(item['predicate_ref'])
            else:
                text(item, key + ' label')
    for key in ('states', 'terminal_states'):
        values = context[key]
        if not isinstance(values, list) or not 1 <= len(values) <= 64:
            raise ValueError('invalid ' + key)
        for value in values:
            identifier(value)
        if len(set(values)) != len(values):
            raise ValueError('duplicate state')
    if set(context['states']) & set(context['terminal_states']):
        raise ValueError('active and terminal states overlap')


def object_schema(properties):
    return dict(type='object', additionalProperties=False, properties=properties, required=list(properties))


def request(context):
    """Build a local request packet; never call an API."""
    validate_context(context)
    ref = lambda values: dict(type='string', enum=list(values))
    condition = object_schema(dict(id=dict(type='string', pattern='^[a-z][a-z0-9_-]{0,63}$'),
        action_ref=ref(context['actions']), target_ref=ref(context['targets']), outcome_ref=ref(context['outcomes'])))
    schema = object_schema(dict(schema=dict(type='integer', enum=[1]),
        context_sha256=ref([digest(context)]), status=ref(['candidate', 'unsupported']),
        reason=dict(type='string'), conditions=dict(type='array', maxItems=64, items=condition),
        state_bindings=dict(type='array', maxItems=64, items=object_schema(dict(
            state=ref(context['states']), condition_id=dict(type='string'))))))
    return dict(instructions=INSTRUCTIONS, context=deepcopy(context), context_sha256=digest(context),
                response_schema=schema)


def compile_conditions(context, proposal, max_prompt_chars=512):
    validate_context(context)
    fields(proposal, 'schema context_sha256 status reason conditions state_bindings', 'proposal')
    if (type(proposal['schema']) is not int or proposal['schema'] != 1
            or proposal['context_sha256'] != digest(context)):
        raise ValueError('proposal does not match the current context revision')
    if proposal['status'] not in ('candidate', 'unsupported'):
        raise ValueError('invalid proposal status')
    if not isinstance(proposal['reason'], str) or len(proposal['reason']) > 500:
        raise ValueError('invalid proposal reason')
    for key in ('conditions', 'state_bindings'):
        if not isinstance(proposal[key], list) or len(proposal[key]) > 64:
            raise ValueError('invalid ' + key)
    result = dict(schema='goal_condition_candidate_v1', context_sha256=digest(context),
        context=deepcopy(context), proposal=deepcopy(proposal), execution_authorized=False,
        tokenizer_checked=False, capability_verified=False, conditions={}, state_bindings={},
        status=proposal['status'])
    if proposal['status'] == 'unsupported':
        text(proposal['reason'], 'unsupported reason', 500)
        if proposal['conditions'] or proposal['state_bindings']:
            raise ValueError('unsupported proposals must not contain executable bindings')
        return result
    if type(max_prompt_chars) is not int or max_prompt_chars < 1:
        raise ValueError('invalid prompt length limit')
    for item in proposal['conditions']:
        fields(item, 'id action_ref target_ref outcome_ref', 'condition')
        identifier(item['id'])
        if item['id'] in result['conditions']:
            raise ValueError('duplicate condition')
        resolved = {}
        for field, registry in (('action_ref', 'actions'), ('target_ref', 'targets'), ('outcome_ref', 'outcomes')):
            identifier(item[field])
            if item[field] not in context[registry]:
                raise ValueError('unknown ' + field)
            resolved[registry] = context[registry][item[field]]
        parts = [resolved['actions'] + ' ' + resolved['targets'] + '.',
                 'Purpose: ' + context['active_goal'] + '.',
                 'Expected result: ' + resolved['outcomes']['description'] + '.']
        if context['root_goal'] != context['active_goal']:
            parts.append('Overall goal: ' + context['root_goal'] + '.')
        if context['constraints']:
            parts.append('Constraints: ' + '; '.join(context['constraints']) + '.')
        prompt = ' '.join(parts)
        if len(prompt) > max_prompt_chars:
            raise ValueError('prompt too long: revise context explicitly; never truncate goal or constraints')
        result['conditions'][item['id']] = dict(item, prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
            predicate_ref=resolved['outcomes']['predicate_ref'])
    for binding in proposal['state_bindings']:
        fields(binding, 'state condition_id', 'binding')
        state, condition_id = binding['state'], binding['condition_id']
        identifier(state)
        identifier(condition_id)
        if state not in context['states'] or state in result['state_bindings']:
            raise ValueError('unknown, terminal or duplicate state binding')
        if condition_id not in result['conditions']:
            raise ValueError('unknown condition binding')
        result['state_bindings'][state] = condition_id
    if set(result['state_bindings']) != set(context['states']):
        raise ValueError('missing active-state binding; no implicit exploration fallback')
    if set(result['state_bindings'].values()) != set(result['conditions']):
        raise ValueError('unused condition')
    return result


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('request', 'compile'))
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--proposal', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.operation == 'compile') != (args.proposal is not None):
        parser.error('compile requires --proposal; request does not accept it')
    context = read_json(args.context)
    result = request(context) if args.operation == 'request' else compile_conditions(context, read_json(args.proposal))
    # Exclusive creation preserves prior requests/proposals; validation has no game side effects.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')
    print('Wrote ' + args.operation + ' artifact; no API, embedding, game or training execution.')


if __name__ == '__main__':
    main()
