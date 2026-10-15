"""Persistent collection goals sharing the existing task-session execution boundary."""
from copy import deepcopy

from .priority_schedule import PriorityLedger
from .tasks import TaskState, validate_command


class ScheduledTasks:
    def __init__(self):
        self.states = {}
        self.versions = {}
        self.failures = {}
        self.stagnation = {}
        self.ledger = PriorityLedger()
        self.current_id = None
        self.last_observation = -1

    @property
    def task(self):
        return self.states[self.current_id].task if self.current_id is not None else None

    def snapshot(self): return deepcopy(self.task)

    @staticmethod
    def _command(command):
        result = {k:v for k,v in command.items() if k not in ('priority','depends_on','deadline_s')}
        if result['op']=='enqueue': result['op']='request'
        return result

    def check(self, command):
        validate_command(command)
        key, op = command['task_id'], command['op']
        if op in ('enqueue','request','request_plan'):
            if key in self.states: raise ValueError('goal identity already exists; modify or resume it')
            if len(self.states)>=64: raise ValueError('goal queue capacity reached')
            if any(d not in self.states for d in command.get('depends_on',[])):
                raise ValueError('unknown goal dependency')
            return
        if key not in self.states: raise ValueError('unknown task')
        state = self.states[key]
        if op=='reprioritize':
            if command['expected_revision']!=state.task['revision']: raise ValueError('stale revision')
            if state.task['status'] in ('completed','cancelled'): raise ValueError('terminal goal priority')
        else: state.check(command)
        if op in ('modify','modify_plan','reprioritize') and len(self.versions[key])>=32:
            raise ValueError('goal revision history capacity reached')
        if op=='resume' and len(self.failures[key])>=32: raise ValueError('goal failure history capacity reached')

    def apply(self, command, inventory):
        self.check(command)
        key, op = command['task_id'], command['op']
        if op in ('enqueue','request','request_plan'):
            state = TaskState(); state.apply(self._command(command),inventory)
            self.ledger.register(key,priority=command.get('priority',50),
                depends_on=command.get('depends_on',[]),deadline_s=command.get('deadline_s'))
            self.states[key]=state; self.versions[key]=[]; self.failures[key]=[]
            self.stagnation[key]=dict(revision=state.task['revision'],best=0,service_at_progress=0.)
        else:
            state = self.states[key]
            if op in ('modify','modify_plan','reprioritize'): self.versions[key].append(state.snapshot())
            if op=='reprioritize':
                self.ledger.reprioritize(key,command['priority'])
                state.task['revision']+=1
                self.stagnation[key]['revision']=state.task['revision']
            else:
                state.apply(command,inventory)
                if op in ('modify','modify_plan','resume'):
                    self.stagnation[key]=dict(revision=state.task['revision'],best=state.task['progress'],
                        service_at_progress=self.ledger.records[key]['direct_service_s'])
            self._sync(key)
        return self.states[key].snapshot()

    def _sync(self, key):
        task = self.states[key].task
        status = ('blocked' if task.get('failure') and task['status']=='paused' else
                  'pending' if task['status']=='running' else task['status'])
        self.ledger.set_status(key,status)

    def advance(self, seconds, *, service_s=0.):
        for key in self.states: self._sync(key)
        self.ledger.advance(seconds,serving=self.current_id if service_s else None,service_s=service_s)

    def select(self, inventory, *, safe_boundary=True):
        for key in self.states: self._sync(key)
        result = self.ledger.select(safe_boundary=safe_boundary)
        self.current_id = result['goal_id']
        return result

    def observe(self, inventory, sequence):
        if sequence<=self.last_observation: raise ValueError('observation sequence must increase')
        self.last_observation=sequence
        if self.current_id is None: return None
        key = self.current_id
        event = self.states[key].observe(inventory,sequence)
        task = self.states[key].task; timing = self.stagnation[key]
        if task['revision']!=timing['revision'] or task['progress']>timing['best']:
            self.stagnation[key]=dict(revision=task['revision'],best=task['progress'],
                service_at_progress=self.ledger.records[key]['direct_service_s'])
        self._sync(key)
        return event

    def stalled_s(self):
        if self.task is None or self.task['status']!='running': return 0.
        key=self.current_id
        return self.ledger.records[key]['direct_service_s']-self.stagnation[key]['service_at_progress']

    def pause_for_no_progress(self,elapsed_s,threshold_s):
        result=self.states[self.current_id].pause_for_no_progress(elapsed_s,threshold_s)
        self.failures[self.current_id].append(dict(deepcopy(result['failure']),revision=result['revision']))
        self._sync(self.current_id)
        return result

    def all_terminal(self):
        return bool(self.states) and all(s.task['status'] in ('completed','cancelled') for s in self.states.values())

    def checkpoint(self,world_id):
        if not isinstance(world_id,str) or not world_id: raise ValueError('world identity required')
        return dict(schema=4,world_id=world_id,last_observation=self.last_observation,
            current_id=self.current_id,goals={k:s.checkpoint(world_id) for k,s in self.states.items()},
            priorities=self.ledger.snapshot(),versions=deepcopy(self.versions),failures=deepcopy(self.failures),
            stagnation=deepcopy(self.stagnation))

    @classmethod
    def from_legacy(cls,state):
        result=cls(); result.last_observation=state.last_observation
        if state.task:
            key=state.task['task_id']; result.states[key]=state; result.versions[key]=[];result.failures[key]=[]
            if state.task.get('failure'):
                result.failures[key].append(dict(deepcopy(state.task['failure']),revision=state.task['revision']))
            result.ledger.register(key)
            result.stagnation[key]=dict(revision=state.task['revision'],best=state.task['progress'],service_at_progress=0.)
            result._sync(key)
        return result

    @classmethod
    def restore(cls,value,world_id):
        if (set(value)!={'schema','world_id','last_observation','current_id','goals','priorities','versions','failures','stagnation'}
                or value['schema']!=4 or value['world_id']!=world_id or
                type(value['last_observation']) is not int or value['last_observation'] < -1):
            raise ValueError('invalid scheduled-task checkpoint')
        result=cls(); result.ledger=PriorityLedger.restore(value['priorities'])
        goals=value['goals']
        if not isinstance(goals,dict) or set(goals)!=set(result.ledger.records): raise ValueError('goal identities differ')
        if value['current_id'] is not None and value['current_id'] not in goals: raise ValueError('unknown current goal')
        if any(not isinstance(value[k],dict) or set(value[k])!=set(goals) for k in ('versions','stagnation','failures')):
            raise ValueError('missing goal histories')
        for key, saved in goals.items():
            if not isinstance(saved,dict) or saved.get('schema') not in (1,2,3): raise ValueError('nested task queue')
            state=TaskState.restore(saved,world_id)
            if state.task is None or state.task['task_id']!=key or state.last_observation>value['last_observation']:
                raise ValueError('invalid goal binding')
            row=result.ledger.records[key]
            expected=('blocked' if saved['task'].get('failure') and saved['task']['status']=='paused' else
                      'pending' if saved['task']['status']=='running' else saved['task']['status'])
            if row['status']!=expected: raise ValueError('inconsistent goal status')
            history=value['versions'][key]
            if not isinstance(history,list) or len(history)>32: raise ValueError('invalid version history')
            for old in history:
                schema=3 if 'failure' in old else 2 if 'plan' in old else 1
                TaskState.restore(dict(schema=schema,world_id=world_id,task=old,last_observation=value['last_observation']),world_id)
                if old['task_id']!=key or old['revision']>=state.task['revision']: raise ValueError('invalid goal version')
            timing=value['stagnation'][key]
            from .priority_schedule import nonnegative
            failures=value['failures'][key]
            if not isinstance(failures,list) or len(failures)>32: raise ValueError('invalid failure history')
            for failure in failures:
                if (not isinstance(failure,dict) or set(failure)!=
                        {'reason','elapsed_s','threshold_s','observation_sequence','revision'} or
                        failure['reason']!='no_progress' or type(failure['revision']) is not int or
                        not 1<=failure['revision']<=state.task['revision'] or
                        type(failure['observation_sequence']) is not int or
                        not -1<=failure['observation_sequence']<=value['last_observation'] or
                        not nonnegative(failure['elapsed_s']) or not nonnegative(failure['threshold_s']) or
                        not 0<failure['threshold_s']<=180 or failure['elapsed_s']<failure['threshold_s']):
                    raise ValueError('invalid failure evidence')
            if (not isinstance(timing,dict) or set(timing)!={'revision','best','service_at_progress'} or
                    type(timing['revision']) is not int or not 1<=timing['revision']<=state.task['revision'] or
                    type(timing['best']) is not int or timing['best']<0 or
                    not nonnegative(timing['service_at_progress']) or timing['service_at_progress']>row['direct_service_s']):
                raise ValueError('invalid stagnation history')
            result.states[key]=state
            # Restart is a user-visible pause, including formerly blocked goals.
            if row['status']=='pending': result.ledger.set_status(key,'paused')
        result.versions=deepcopy(value['versions']); result.stagnation=deepcopy(value['stagnation'])
        result.failures=deepcopy(value['failures'])
        result.last_observation=value['last_observation']
        return result
