"""Root-aligned reward and independently replayable E2 evidence."""
from copy import deepcopy
import hashlib
import math
import struct

from .goal_queue import GoalQueue
from .goal_spec import TARGETS, validate_trace
from .execution_contract import execution_request
from .rl_core import gae
from .runtime import inventory_counts

GAMMA=.999
ALPHA=.2
EXTRA_FIELDS=('effect_acquire','effect_possess','program_progress',
    *(name for ref in sorted(TARGETS) for name in ('preserve_'+ref+'_min','inventory_'+ref)),
    *(f'goal_vector_{i:03d}' for i in range(512)))
REWARD_PROFILE=dict(schema='program_return_v1',gamma=GAMMA,alpha=ALPHA,
    objective='whole_program_completion',shaping='discounted_program_potential',
    boundary='episode_terminal_only',terminal_potential=0.)


def check_committed_evidence(row, previous_server, event):
    if (row['step']!=event['step'] or row['env_action']!=event['action'] or row['reward']!=event['reward']):
        raise ValueError('raw action/reward differs from committed row')
    for state,server in ((row['execution_before'],previous_server),
                         (row['execution_after'],event['info']['workshop'])):
        if (state['tick']!=server['server_tick'] or state['inventory']!=inventory_counts({'inventory':server['server_inventory']})):
            raise ValueError('raw inventory/tick differs from execution evidence')


def program_potential(queue):
    return sum(1. if r['status']=='completed' else min(1.,r['progress']/queue.goals[k]['count'])
               for k,r in queue.records.items())/len(queue.goals)


def state_record(queue):
    return dict(tick=queue.last_tick,inventory=deepcopy(queue.inventory),active_goal=queue.active,
                records=deepcopy(queue.records),completed=queue.completed)


def context_tail(queue, goal_vector):
    if len(goal_vector)!=512 or not all(math.isfinite(v) for v in goal_vector):
        raise ValueError('finite existing 512-dimensional goal vector required')
    c=queue.goal_spec['contracts'][queue.active]
    preserve={need['target']:need['quantity'] for need in c['preserve']}
    values=[float(c['effect']=='acquire'),float(c['effect']=='possess'),program_potential(queue)]
    for ref in sorted(TARGETS):
        count=sum(queue.inventory.get(n,0) for n in TARGETS[ref][1])
        values += [math.log2(1+preserve.get(ref,0))/math.log2(65.),
                   math.log2(1+min(64,count))/math.log2(65.)]
    return values+list(goal_vector)


def learning_reward(before_potential, after_potential, completed, terminal):
    return float(completed)+ALPHA*(GAMMA*(0. if terminal else after_potential)-before_potential)


def program_returns(run, *, gamma=GAMMA):
    return gae([r['learning_reward'] for r in run['rows']], [r['value'] for r in run['rows']],
               run['bootstrap'],terminal=run['terminal'],gamma=gamma,lam=1.)


def validate_execution_trace(data):
    validate_trace(data)
    if data.get('reward_profile')!=REWARD_PROFILE or data['goal_spec']['source']['schema']!='goal_spec_v2':
        raise ValueError('incompatible execution reward/meaning')
    q=GoalQueue(data['goal_spec']['source'],{skill:items for skill,items,_ in TARGETS.values()})
    def server(state):return dict(server_available=1,error=0,server_tick=state['tick'],server_inventory=state['inventory'])
    initial=data['rows'][0]['execution_before'];q.observe(server(initial));q.select()
    for row in data['rows']:
        if (not all(math.isfinite(row[k]) for k in ('learning_reward','program_reward','potential_before','potential_after'))
                or type(row['terminal']) is not bool or len(row['phase_context'])!=len(data['context_fields'])):
            raise ValueError('invalid execution learning evidence')
        before,after=row['execution_before'],row['execution_after']
        if state_record(q)!=before or after['tick']!=before['tick']+1:
            raise ValueError('execution state discontinuity')
        if row['execution_request']!=execution_request(q,row['phase'],data['recovery_task']):
            raise ValueError('execution request drift')
        vector=row['phase_context'][-512:]
        if hashlib.sha256(struct.pack('<512f',*vector)).hexdigest()!=row['embedding_sha256']:
            raise ValueError('explicit goal vector disconnected')
        tail=context_tail(q,vector)
        if any(not math.isfinite(a) or abs(a-b)>1e-6 for a,b in zip(row['phase_context'][-len(EXTRA_FIELDS):],tail)):
            raise ValueError('execution context disconnected')
        phi=program_potential(q)
        reward=q.observe(server(after))
        if q.active is not None:
            reported=after['records'][q.active]
            if reported['status']=='blocked':
                if reported['reason'] not in ('no_confirmed_progress','result_not_acquired'):
                    raise ValueError('unsupported controller failure')
                q.block(reported['reason'],{})
        for event in row.get('boundary_updates',[]):
            if event['status']=='applied':
                q.reprioritize(event['goal_id'],event['priority'],event['expected_revision'])
        q.select()
        if state_record(q)!=after or reward!=row['reward']:
            raise ValueError('execution outcome differs from inventory')
        expected=learning_reward(phi,program_potential(q),q.completed,row['terminal'])
        if (row['program_reward']!=float(q.completed) or abs(row['learning_reward']-expected)>1e-7
                or row['potential_before']!=phi or row['potential_after']!=program_potential(q)):
            raise ValueError('root return/shaping mismatch')
        if q.completed and not row['terminal']:raise ValueError('completed program must terminate')
