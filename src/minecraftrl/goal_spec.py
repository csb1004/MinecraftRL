"""One authoring source for queue rewards, policy conditions and learning evidence."""
import argparse
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path

from .goal_contract import digest, fields, object_schema, read_json, text
from .rl_core import LOGS

SCHEMA = 'goal_spec_v1'
RESULT = 'inventory_increase_from_activation'
# Runtime-owned capabilities; model responses cannot extend this registry.
TARGETS = {'wood': ('wood_memory', tuple(LOGS), 'wood logs'),
           'dirt': ('dirt_fixed', ('dirt',), 'dirt')}
PICKUP_PROMPTS = {'pickup_wood': 'pick up wood', 'pickup_dirt': 'pick up dirt'}


def target(ref):
    if not isinstance(ref, str) or ref not in TARGETS:
        raise ValueError('unsupported target: ' + str(ref))
    return TARGETS[ref]


def quantity(value):
    if type(value) is not int or not 1 <= value <= 64:
        raise ValueError('quantity must be an integer in [1,64]')
    return value


def compile_spec(source):
    from .goal_queue import validate_program
    if not isinstance(source, dict):
        raise ValueError('goal specification must be an object')
    if source.get('schema') == 'goal_spec_v2':
        return compile_execution_spec(source)
    names = 'schema goals' + (' progress_limits' if 'progress_limits' in source else '')
    fields(source, names, 'goal specification')
    if source['schema'] != SCHEMA or not isinstance(source['goals'], list) or not 1 <= len(source['goals']) <= 64:
        raise ValueError('unsupported goal specification')
    goals, contracts = [], {}
    for goal in source['goals']:
        fields(goal, 'id target quantity priority depends_on requires', 'goal')
        skill, items, label = target(goal['target'])
        count = quantity(goal['quantity'])
        if not isinstance(goal['requires'], list) or len(goal['requires']) > 16:
            raise ValueError('invalid prerequisites')
        needs, seen = [], set()
        for need in goal['requires']:
            fields(need, 'target quantity', 'prerequisite')
            _, required_items, _ = target(need['target'])
            if need['target'] in seen:
                raise ValueError('duplicate prerequisite target')
            seen.add(need['target'])
            needs.append(dict(any_of=list(required_items), count=quantity(need['quantity'])))
        goals.append(dict(id=goal['id'], objective=f'Acquire at least {count} new {label} in server inventory.',
            priority=goal['priority'], depends_on=deepcopy(goal['depends_on']), items=list(items),
            count=count, skill=skill, requires=needs))
    program = dict(schema='goal_queue_v1', goals=goals)
    if 'progress_limits' in source:
        program['progress_limits'] = deepcopy(source['progress_limits'])
    program = validate_program(program)
    for authored, goal in zip(source['goals'], program['goals']):
        contract = dict(target_ref=authored['target'], items=goal['items'], quantity=goal['count'],
            result=RESULT, comparison='ge', prerequisites=goal['requires'], depends_on=goal['depends_on'])
        contracts[goal['id']] = dict(contract, sha256=digest(contract))
    return dict(schema='compiled_goal_spec_v1', source=deepcopy(source), source_sha256=digest(source),
        program=program, program_sha256=digest(program), contracts=contracts,
        capability_verified=False, natural_language_interpretation_verified=False)


def resolve_program(value):
    from .goal_queue import validate_program
    if isinstance(value, dict) and value.get('schema') in (SCHEMA, 'goal_spec_v2'):
        compiled = compile_spec(value)
        return deepcopy(compiled['program']), compiled
    return validate_program(value), None


def compile_execution_spec(source):
    """Explicit v2 semantics; v1 keeps its original acquisition-only contract."""
    fields(source, 'schema goals'+(' progress_limits' if 'progress_limits' in source else ''), 'execution specification')
    if not isinstance(source['goals'], list) or not 1 <= len(source['goals']) <= 64:
        raise ValueError('execution goals required')
    legacy=deepcopy(source);legacy['schema']=SCHEMA
    for goal in legacy['goals']:
        fields(goal, 'id target quantity priority depends_on requires effect preserve', 'execution goal')
        if goal['effect'] not in ('acquire','possess'):
            raise ValueError('unsupported execution effect')
        if not isinstance(goal['preserve'],list) or len(goal['preserve']) > len(TARGETS):
            raise ValueError('invalid preserved resources')
        seen=set()
        for need in goal['preserve']:
            fields(need,'target quantity','preserved resource')
            target(need['target']);quantity(need['quantity'])
            if need['target'] in seen:raise ValueError('duplicate preserved resource')
            seen.add(need['target'])
        del goal['effect'];del goal['preserve']
    compiled=compile_spec(legacy)
    for authored, goal in zip(source['goals'],compiled['program']['goals']):
        contract=compiled['contracts'][goal['id']]
        contract.pop('sha256')
        contract.update(effect=authored['effect'],preserve=deepcopy(authored['preserve']))
        if authored['effect']=='possess':
            contract['result']='inventory_at_least'
            goal['objective']=f"Hold at least {goal['count']} {target(authored['target'])[2]} in server inventory."
        contract['sha256']=digest(contract)
    compiled.update(schema='compiled_goal_spec_v2',source=deepcopy(source),source_sha256=digest(source),
                    program_sha256=digest(compiled['program']))
    return compiled


def assert_goal(queue, goal_id):
    """Allow audited reprioritization, never silent reward/target redefinition."""
    compiled = queue.goal_spec
    if compiled is None:
        return None
    expected = next(g for g in compiled['program']['goals'] if g['id'] == goal_id)
    current = queue.goals[goal_id]
    if {k:v for k,v in current.items() if k != 'priority'} != {k:v for k,v in expected.items() if k != 'priority'}:
        raise ValueError('goal specification drift before reward or action')
    return deepcopy(compiled['contracts'][goal_id])


@lru_cache(maxsize=5)
def expected_prompt(key):
    if key in PICKUP_PROMPTS:
        return PICKUP_PROMPTS[key]
    if key == 'dirt':
        return 'collect dirt [official video embedding]'
    from .logo_wood import prompts
    return prompts()[key]


def condition_key(ref, phase, task):
    target(ref)
    if phase not in ('search', 'approach', 'mine', 'collect') or task not in ('collection', 'pickup'):
        raise ValueError('unsupported condition phase or task')
    if task == 'pickup':
        return 'pickup_'+ref
    if ref == 'dirt':
        return 'dirt'
    return 'seek' if phase == 'search' else 'root'


def policy_condition(queue, phase, task='collection'):
    if queue.active is None or queue.completed:
        raise InterruptedError('no active goal condition')
    contract = assert_goal(queue, queue.active)
    if contract is None:
        raise ValueError('canonical policy condition requires a goal specification')
    return condition_record(queue.goal_spec, queue.active, phase, task)


def condition_record(compiled, goal_id, phase, task):
    contract = compiled['contracts'][goal_id]
    key = condition_key(contract['target_ref'], phase, task)
    return dict(deepcopy(contract), source_sha256=compiled['source_sha256'], goal_id=goal_id,
        goal_key=key, prompt=expected_prompt(key), phase=phase, task=task,
        quantity_delivery='structured_policy_context_and_rm',
        embedding_contains_quantity=False, capability_verified=False)


def validate_trace(data):
    """Reject disconnected canonical metadata before any optimizer update."""
    recorded = data.get('goal_spec')
    if recorded is None:
        if any('goal_binding' in row for row in data['rows']):
            raise ValueError('goal trace lost its source specification')
        return
    compiled = compile_spec(recorded['source'])
    if (recorded != compiled or data.get('program') != compiled['program'] or
            data.get('program_sha256') != compiled['program_sha256']):
        raise ValueError('goal trace compilation mismatch')
    if not data.get('variant','').endswith('-on'):
        raise ValueError('goal trace requires enabled structured policy context')
    task = data.get('recovery_task','collection')
    fields = data['context_fields']
    for row in data['rows']:
        if row.get('goal_id') not in compiled['contracts']:
            raise ValueError('goal trace contains an unknown goal')
        expected = condition_record(compiled, row['goal_id'], row['phase'], task)
        key = expected['goal_key']
        if row.get('goal_binding') != expected or row.get('goal_key') != key:
            raise ValueError('goal trace policy binding mismatch')
        bank = data['bank_metadata']
        if (bank['prompts'].get(key) != expected['prompt'] or
                row.get('embedding_sha256') != bank['embeddings'][key]['vector_sha256']):
            raise ValueError('goal trace embedding mismatch')
        encoded = math.log2(1+expected['quantity'])/math.log2(65.)
        actual = row['phase_context'][fields.index('quantity_log2')]
        if not math.isfinite(actual) or abs(actual-encoded) > 1e-7:
            raise ValueError('goal trace quantity disconnected from policy input')


def request(version=1):
    """Prepare a schema for humans or an API caller; never invokes an LLM."""
    if version not in (1,2):raise ValueError('unsupported authoring version')
    ref = dict(type='string', enum=list(TARGETS))
    count = dict(type='integer', minimum=1, maximum=64)
    need = object_schema(dict(target=ref, quantity=count))
    goal = object_schema(dict(id=dict(type='string', pattern='^[a-zA-Z0-9_-]{1,64}$'),
        target=ref, quantity=count, priority=dict(type='integer', minimum=0, maximum=100),
        depends_on=dict(type='array', items=dict(type='string'), maxItems=63),
        requires=dict(type='array', items=need, maxItems=16)))
    if version==2:
        goal['properties'].update(effect=dict(type='string',enum=['acquire','possess']),
            preserve=dict(type='array',items=need,maxItems=len(TARGETS)))
        goal['required']+=['effect','preserve']
    spec_schema = object_schema(dict(schema=dict(type='string', enum=[SCHEMA if version==1 else 'goal_spec_v2']),
        goals=dict(type='array', items=goal, minItems=1, maxItems=64)))
    return dict(instructions=(
        'Express the user goal with only these registered targets. '+(
        'quantity is a minimum NEW inventory gain from activation of that goal, not existing stock, visibility or a button press. '
        if version==1 else
        'effect=acquire means NEW inventory gain from activation; effect=possess means currently held stock. '
        'Already satisfied possession completes without actions or action credit. '
        'preserve lists minimum inventory that must remain held throughout that active goal, including its final tick. ')+
        'List dependencies before dependents; requires means currently held prerequisite items. '
        'Preserve requested quantities and ordering. Never invent skills, prompts, predicates or constraints. '
        'Return status=candidate, reason="", spec=<goal specification> when expressible. '
        'Otherwise return status=unsupported, a reason, and spec=null; do not substitute another goal. '
        'Schema validity does not establish correct interpretation or policy capability.'),
        response_schema=object_schema(dict(status=dict(type='string', enum=['candidate','unsupported']),
            reason=dict(type='string', maxLength=512), spec=dict(anyOf=[spec_schema, dict(type='null')]))),
        target_items={key:list(value[1]) for key,value in TARGETS.items()},
        API_calls=0, execution_authorized=False)


def compile_response(response):
    fields(response, 'status reason spec', 'goal response')
    if response['status'] == 'unsupported':
        text(response['reason'], 'unsupported reason', 512)
        if response['spec'] is not None:
            raise ValueError('unsupported response must not contain an executable specification')
        return dict(status='unsupported', reason=response['reason'], execution_authorized=False)
    if response['status'] != 'candidate' or response['reason'] != '':
        raise ValueError('invalid candidate response')
    return compile_spec(response['spec'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('request','compile','compile-response'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version',type=int,choices=(1,2),default=1,help='authoring request schema version')
    args = parser.parse_args(argv)
    if (args.operation != 'request') != (args.input is not None):
        parser.error('compilation requires --input; request does not accept it')
    if args.version!=1 and args.operation!='request':parser.error('--version is for request; compilation reads the source schema')
    result = (request(args.version) if args.operation == 'request' else
              (compile_spec if args.operation == 'compile' else compile_response)(read_json(args.input)))
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')
    print('Goal contract written; no API, model, game or training execution.')


if __name__ == '__main__':
    main()
