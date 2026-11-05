"""Caller-owned wood observer contract and independent admission cases.

No JAR/sensor changes. Registry entries describe what the existing server fields
actually establish. In particular a remembered target is not current visibility.
"""
from copy import deepcopy

from .goal_contract import digest
from .goal_actions import compile_action_conditions
from .logo_validators import ValidatorSet
from .rl_core import LOGS


def context():
    def fact(path, description, items=()):
        return dict(path=[path], items=list(items), description=description)
    root = dict(op='ge', fact='resource_count', mode='delta', value=1)
    return dict(schema='logo_context_v1', task='Acquire at least one new wood log in server inventory.',
        facts=dict(
            resource_count=fact('server_inventory', 'Server inventory sum of the six registered log items.', LOGS),
            target_kind=fact('wood_v3_kind', '0 none; 1 remembered existing log; 2 tracked block absent for up to 40 ticks. Not current visibility.'),
            reachable=fact('wood_v3_reachable', 'Server reach and line of sight to the tracked log, or accepted mining.'),
            screen=fact('screen', '0 game; 1 inventory; 2 crafting; 3 other.'),
            alive=fact('server_alive', 'Server player alive flag.'),
            log_breaks=fact('server_log_harvests', 'Cumulative confirmed log harvests by this player. Not attack-button count.')),
        validators=['target_known', 'target_reachable', 'log_broken', 'item_acquired'],
        root='item_acquired', root_requirement=root)


def acceptance_cases():
    """Trusted contract tests, not examples supplied by the proposing model."""
    baseline = dict(server_available=1, error=0, server_tick=100, server_inventory={name:0 for name in LOGS},
                    wood_v3_kind=0, wood_v3_reachable=0, screen=0, server_alive=1, server_log_harvests=7)
    cases = []

    def add(name, changes, expected, start=None):
        before = deepcopy(baseline if start is None else start)
        current = dict(deepcopy(before), server_tick=101, **changes)
        cases.append(dict(name=name, baseline=before, current=current, expected=expected))

    no = dict(target_known=False, target_reachable=False, log_broken=False, item_acquired=False)
    add('no evidence; historical breaks do not count', {}, no)
    add('remembered log beyond reach', dict(wood_v3_kind=1), dict(no, target_known=True))
    add('reachable existing log', dict(wood_v3_kind=1, wood_v3_reachable=1), dict(no, target_known=True, target_reachable=True))
    add('GUI blocks interaction', dict(wood_v3_kind=1, wood_v3_reachable=1, screen=1), dict(no, target_known=True))
    add('dead player cannot interact', dict(wood_v3_kind=1, wood_v3_reachable=1, server_alive=0), dict(no, target_known=True))
    add('attack is not harvest', dict(wood_server_mining=1), no)
    add('disappearing target is not own harvest or pickup', dict(wood_v3_kind=2), no)
    add('confirmed own harvest without inventory', dict(server_log_harvests=8, wood_v3_kind=2), dict(no, log_broken=True))
    add('client-only inventory is not pickup', dict(inventory=dict(oak_log=20)), no)
    for item in LOGS:
        inventory = dict(baseline['server_inventory'], **{item:1})
        add('new '+item+' even without own mining', dict(server_inventory=inventory), dict(no, item_acquired=True))
    held = deepcopy(baseline)
    held['server_inventory']['oak_log'] = 3
    add('old inventory is not acquisition', {}, no, held)
    add('server unavailable', dict(server_available=0), {name:None for name in no})
    add('missing item count is unknown', dict(server_inventory={}), dict(item_acquired=None))
    add('NaN inventory is unknown', dict(server_inventory=dict(baseline['server_inventory'], oak_log='NaN')), dict(item_acquired=None))
    return cases


def admit(proposal):
    validators = ValidatorSet(context(), proposal)
    review = validators.verify(acceptance_cases())
    if not review['passed']:
        raise ValueError('validator proposal failed caller-owned acceptance cases: '+str(review['failures']))
    return validators, review


def action_candidate():
    contract = dict(schema=1, root_goal='collect at least one wood log', active_goal='collect at least one wood log',
        constraints=[], actions=dict(explore_find='go explore to find', chop='chop down', gather='gather', pick='pick up'),
        targets=dict(source='a tree', tree='the tree', resource='wood'),
        outcomes=dict(located=dict(description='a log target is registered by the server tracker', predicate_ref='target_known'),
                      acquired=dict(description='server inventory gains at least one wood log', predicate_ref='item_acquired')),
        states=['search', 'approach', 'mine', 'collect'], terminal_states=['complete'])
    proposal = dict(schema=1, context_sha256=digest(contract), status='candidate', reason='',
        conditions=[dict(id='seek', action_ref='explore_find', target_ref='source', outcome_ref='located'),
                    dict(id='gather', action_ref='chop', target_ref='tree', outcome_ref='acquired')],
        state_bindings=[dict(state=stage, condition_id='seek' if stage == 'search' else 'gather') for stage in contract['states']])
    clause = lambda action, target: dict(action_ref=action, target_ref=target)
    rendering = dict(schema='goal_actions_v2', context_sha256=digest(contract), purpose=[clause('gather','resource')],
        clauses=dict(seek=[clause('explore_find','source'), clause('chop','tree'), clause('gather','resource')],
                     gather=[clause('chop','tree'), clause('gather','resource'), clause('pick','resource')]))
    return compile_action_conditions(contract, proposal, rendering)


def prompts():
    conditions = action_candidate()['conditions']
    return dict(root=conditions['gather']['prompt'], seek=conditions['seek']['prompt'])


def stage(values, server):
    if values['item_acquired']:
        return 'complete'
    if any(value is None for value in values.values()):
        raise ValueError('unknown stage evidence')
    if values['target_reachable']:
        return 'mine'
    if values['target_known']:
        return 'approach'
    # The missing-target hint only chooses a pickup attempt after a confirmed
    # harvest. It never establishes acquisition or supplies a success reward.
    if values['log_broken'] and server['wood_v3_kind'] == 2:
        return 'collect'
    return 'search'
