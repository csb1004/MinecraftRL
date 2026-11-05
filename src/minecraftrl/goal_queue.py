"""Caller-owned, non-preemptive root goals using the shared priority scheduler.

Execution capability is supplied by the runtime, never by a goal proposal.
An unavailable goal remains visible and blocked instead of becoming exploration.
"""
from copy import deepcopy
import re

from .goal_contract import digest
from .goal_spec import assert_goal, resolve_program, target
from .interaction_progress import count
from .logo_validators import observation_valid
from .priority_schedule import PriorityLedger
from .runtime import inventory_counts


def validate_program(value):
    if (not isinstance(value, dict) or set(value) not in ({'schema','goals'}, {'schema','goals','progress_limits'})
            or value['schema'] != 'goal_queue_v1'):
        raise ValueError('invalid goal program')
    if 'progress_limits' in value:
        limits=value['progress_limits']
        if not isinstance(limits,dict) or set(limits) != {'stall_ticks','recovery_ticks'}:
            raise ValueError('invalid progress limits')
        if any(type(n) is not int or not 1 <= n <= 3600 for n in limits.values()):
            raise ValueError('progress budgets must be 1 to 3600 server ticks')
    if not isinstance(value['goals'], list) or not 1 <= len(value['goals']) <= 64:
        raise ValueError('program needs 1 to 64 goals')
    ledger = PriorityLedger()
    for goal in value['goals']:
        if not isinstance(goal, dict) or set(goal) != {'id','objective','priority','depends_on','items','count','skill','requires'}:
            raise ValueError('invalid goal fields')
        for name in ('id','skill'):
            if not isinstance(goal[name], str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',goal[name]):
                raise ValueError('invalid '+name)
        if not isinstance(goal['objective'], str) or not 1 <= len(goal['objective'].strip()) <= 512:
            raise ValueError('goal objective required')
        def items(names):
            if (not isinstance(names, list) or not 1 <= len(names) <= 64 or
                    any(not isinstance(n,str) or not re.fullmatch(r'[a-z0-9_]{1,64}',n) for n in names)
                    or len(set(names)) != len(names)):
                raise ValueError('invalid inventory item group')
        items(goal['items'])
        if type(goal['count']) is not int or not 1 <= goal['count'] <= 64:
            raise ValueError('goal count must be in [1,64]')
        if not isinstance(goal['requires'],list) or len(goal['requires']) > 16:
            raise ValueError('invalid inventory prerequisites')
        for need in goal['requires']:
            if not isinstance(need,dict) or set(need) != {'any_of','count'}:
                raise ValueError('invalid prerequisite')
            items(need['any_of'])
            if type(need['count']) is not int or not 1 <= need['count'] <= 64:
                raise ValueError('invalid prerequisite count')
        ledger.register(goal['id'], priority=goal['priority'], depends_on=goal['depends_on'])
    return deepcopy(value)


class GoalQueue:
    def __init__(self, program, capabilities):
        self.program, self.goal_spec = resolve_program(program)
        self.program_sha256 = digest(self.program)
        self.capabilities = dict(capabilities)
        self.ledger = PriorityLedger()
        self.goals = {g['id']:deepcopy(g) for g in self.program['goals']}
        self.records = {}
        self.events = []
        self.last_tick = None
        self.inventory = None
        self.active = None
        for key, goal in self.goals.items():
            self.ledger.register(key, priority=goal['priority'], depends_on=goal['depends_on'])
            self.records[key] = dict(status='pending', baseline=None, progress=0, revision=1,
                                     activated_tick=None, completed_tick=None, reason=None)

    def _event(self, event, key, **extra):
        self.events.append(dict(event=event, goal_id=key, tick=self.last_tick,
                                revision=self.records[key]['revision'], **deepcopy(extra)))

    def _set_status(self, key, status, reason=None):
        self.records[key].update(status=status, reason=reason)
        self.ledger.set_status(key, 'pending' if status=='active' else status)
        if key == self.active and status != 'active':
            self.active = None

    def _unavailable(self, goal):
        allowed_items = self.capabilities.get(goal['skill'])
        if allowed_items is None:
            return 'unsupported_skill'
        if set(goal['items']) != set(allowed_items):
            return 'skill_target_mismatch'
        for need in goal['requires']:
            if sum(self.inventory.get(n,0) for n in need['any_of']) < need['count']:
                return 'missing_prerequisite_items'
        contract=assert_goal(self,goal['id']) or {}
        if self._preservation_missing(contract,self.inventory):
            return 'missing_preserved_items'
        return None

    @staticmethod
    def _preservation_missing(contract, inventory):
        return any(sum(inventory.get(n,0) for n in target(need['target'])[1]) < need['quantity']
                   for need in contract.get('preserve',[]))

    def observe(self, server):
        if not observation_valid(server):
            raise ValueError('authoritative queue observation unavailable')
        tick = count(server['server_tick'], 'tick')
        if self.last_tick is not None and tick <= self.last_tick:
            raise ValueError('queue observation must advance')
        inventory = inventory_counts({'inventory':server.get('server_inventory')})
        if self.active is not None:
            assert_goal(self, self.active)
        if self.last_tick is not None:
            seconds=(tick-self.last_tick)/20.
            self.ledger.advance(seconds, serving=self.active, service_s=seconds if self.active else 0.)
        self.last_tick, self.inventory = tick, inventory
        reward = 0.
        if self.active is not None:
            key=self.active; goal=self.goals[key]; record=self.records[key]
            contract=assert_goal(self,key) or {}
            if self._preservation_missing(contract,inventory):
                self.block('preservation_violated',dict(inventory=inventory,preserve=contract['preserve']))
                return 0.
            record['progress']=max(0,sum(inventory.get(n,0) for n in goal['items'])-record['baseline'])
            if record['progress'] >= goal['count']:
                record['completed_tick']=tick
                self._set_status(key,'completed')
                self._event('goal_completed',key,progress=record['progress'],baseline=record['baseline'])
                reward=1.
        return reward

    def select(self):
        if self.inventory is None:
            raise RuntimeError('observe server before selecting a goal')
        # Only availability blocks may automatically clear on new real evidence.
        for key,goal in self.goals.items():
            assert_goal(self, key)
            record=self.records[key]
            if record['status'] in ('completed','cancelled','paused'):
                continue
            if record['status']=='blocked' and record['reason'] not in ('unsupported_skill','skill_target_mismatch','missing_prerequisite_items','missing_preserved_items'):
                continue
            reason=self._unavailable(goal)
            if reason:
                if record['status']!='blocked' or record['reason']!=reason:
                    self._set_status(key,'blocked',reason)
                    self._event('goal_blocked',key,reason=reason)
            elif record['status']=='blocked':
                self._set_status(key,'pending')
                self._event('goal_available',key)
        selected=self.ledger.select(until_complete=True)
        key=selected['goal_id']
        if key is None:
            return None
        if self.active != key:
            record=self.records[key];goal=self.goals[key]
            if record['baseline'] is None:
                contract=assert_goal(self,key) or {}
                record['baseline']=(0 if contract.get('effect')=='possess' else
                                    sum(self.inventory.get(n,0) for n in goal['items']))
                record['activated_tick']=self.last_tick
            self.active=key
            self._set_status(key,'active')
            self._event('goal_activated',key,selection=selected,baseline=record['baseline'])
            contract=assert_goal(self,key) or {}
            if contract.get('effect')=='possess':
                record['progress']=sum(self.inventory.get(n,0) for n in goal['items'])
                if record['progress']>=goal['count']:
                    record['completed_tick']=self.last_tick
                    self._set_status(key,'completed')
                    self._event('goal_satisfied_on_activation',key,progress=record['progress'],policy_action_credit=False)
                    return self.select()
        return deepcopy(self.goals[key])

    def block(self, reason, evidence):
        if self.active is None or not isinstance(reason,str) or not reason:
            raise ValueError('active goal and failure reason required')
        key=self.active
        self._set_status(key,'blocked',reason)
        self._event('goal_blocked',key,reason=reason,evidence=evidence)

    def reprioritize(self, key, priority, expected_revision):
        record=self.records[key]
        if expected_revision != record['revision'] or record['status'] in ('completed','cancelled'):
            raise ValueError('stale or terminal goal revision')
        before=self.ledger.records[key]['priority']
        self.ledger.reprioritize(key,priority)
        self.goals[key]['priority']=priority
        record['revision']+=1
        self._event('goal_reprioritized',key,previous_priority=before,priority=priority)

    def cancel(self, key, expected_revision):
        if expected_revision != self.records[key]['revision'] or self.records[key]['status'] in ('completed','cancelled'):
            raise ValueError('stale or terminal goal revision')
        self.records[key]['revision']+=1
        self._set_status(key,'cancelled','explicit_cancel')
        self._event('goal_cancelled',key)

    @property
    def completed(self):
        return all(r['status']=='completed' for r in self.records.values())

    def snapshot(self):
        result = dict(schema='goal_queue_snapshot_v1',program=self.program,
            program_sha256=self.program_sha256,active_goal=self.active,records=self.records,
            priorities=self.ledger.snapshot(),tick=self.last_tick,events=self.events,
            completed=self.completed,automatic_resume_authorized=False)
        if self.goal_spec is not None:
            result['goal_spec'] = self.goal_spec
        return deepcopy(result)
