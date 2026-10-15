"""Structured task commitment, inventory events and revision-safe dispatch."""
import copy
import math
from collections import deque
import re
import threading

GOALS = {
    'wood': ('collect wood', ('oak_log','birch_log','spruce_log','jungle_log','acacia_log','dark_oak_log')),
    'dirt': ('collect dirt', ('dirt',)),
}


def validate_steps(steps):
    if not isinstance(steps,list) or not 1<=len(steps)<=8: raise ValueError('plan requires 1 to 8 stages')
    for stage in steps:
        if not isinstance(stage,dict) or set(stage)!={'goal','count'}: raise ValueError('invalid stage fields')
        if not isinstance(stage['goal'],str) or stage['goal'] not in GOALS: raise ValueError('unsupported stage goal')
        if type(stage['count']) is not int or not 1<=stage['count']<=64: raise ValueError('invalid stage count')


def validate_command(command):
    if not isinstance(command, dict): raise ValueError('command must be an object')
    for key in ('command_id','task_id'):
        if not isinstance(command.get(key),str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',command[key]):
            raise ValueError('invalid '+key)
    op=command.get('op')
    if op not in ('request','modify','request_plan','modify_plan','pause','resume','cancel','enqueue','reprioritize'): raise ValueError('unsupported command')
    allowed={'command_id','task_id','op'}
    if op not in ('request','request_plan','enqueue'):
        allowed.add('expected_revision')
        if type(command.get('expected_revision')) is not int or command['expected_revision']<1:
            raise ValueError('positive expected_revision required')
    if op in ('request','modify','enqueue'):
        allowed.update(('goal','count'))
        if not isinstance(command.get('goal'),str) or command['goal'] not in GOALS:
            raise ValueError('unsupported collection goal')
        if type(command.get('count')) is not int or not 1<=command['count']<=64:
            raise ValueError('count must be an integer in [1,64]')
    if op in ('request_plan','modify_plan'):
        allowed.add('steps'); validate_steps(command.get('steps'))
    if op in ('enqueue','reprioritize'):
        from .priority_schedule import priority_value, nonnegative
        allowed.add('priority'); priority_value(command.get('priority',50))
        if op=='reprioritize' and 'priority' not in command: raise ValueError('priority required')
        if op=='enqueue':
            allowed.update(('depends_on','deadline_s'))
            deps=command.get('depends_on',[])
            if (not isinstance(deps,list) or len(deps)>64 or any(not isinstance(d,str) or
                    not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',d) for d in deps) or len(set(deps))!=len(deps)):
                raise ValueError('invalid task dependencies')
            deadline=command.get('deadline_s')
            if deadline is not None and (not nonnegative(deadline) or not deadline):
                raise ValueError('deadline_s must be a positive active-session lifetime')
    if set(command)-allowed: raise ValueError('unknown command fields')


class TaskState:
    """One committed task; only validated inventory observations complete it."""
    def __init__(self):
        self.task=None
        self.last_observation=-1

    def snapshot(self): return copy.deepcopy(self.task)

    def checkpoint(self,world_id):
        if not isinstance(world_id,str) or not world_id: raise ValueError('world identity required')
        schema=3 if self.task and 'failure' in self.task else (2 if self.task and 'plan' in self.task else 1)
        return {'schema':schema,'world_id':world_id,'task':self.snapshot(),'last_observation':self.last_observation}

    @classmethod
    def restore(cls,value,world_id):
        if isinstance(value,dict) and type(value.get('schema')) is int and value['schema']==4:
            from .scheduled_tasks import ScheduledTasks
            return ScheduledTasks.restore(value,world_id)
        if not isinstance(value,dict) or set(value)!={'schema','world_id','task','last_observation'} or type(value['schema']) is not int or value['schema'] not in (1,2,3) or value['world_id']!=world_id:
            raise ValueError('checkpoint schema or world mismatch')
        sequence=value['last_observation']
        if type(sequence) is not int or sequence < -1: raise ValueError('invalid observation sequence')
        task=copy.deepcopy(value['task']); state=cls()
        if task is not None:
            required={'task_id','revision','status','goal','policy_goal','target_items','target_count','baseline','progress'}
            has_plan=value['schema']==2 or (value['schema']==3 and isinstance(task,dict) and 'plan' in task)
            if has_plan: required|={'plan','stage_index','completed_stages'}
            if value['schema']==3: required.add('failure')
            if not isinstance(task,dict) or set(task)!=required: raise ValueError('invalid task checkpoint')
            validate_command(dict(command_id='restore',task_id=task['task_id'],op='modify',
                                  expected_revision=task['revision'],goal=task['goal'],count=task['target_count']))
            for key in ('baseline','progress'):
                if type(task[key]) is not int or task[key]<0: raise ValueError('invalid inventory progress')
            goal,items=GOALS[task['goal']]
            if task['policy_goal']!=goal or task['target_items']!=list(items): raise ValueError('goal metadata mismatch')
            if task['status'] not in ('running','paused','cancelled','completed'): raise ValueError('invalid saved status')
            if task['status']=='completed' and task['progress']<task['target_count']: raise ValueError('invalid saved completion')
            if value['schema']==3:
                failure=task['failure']
                if task['status'] not in ('paused','cancelled') or not isinstance(failure,dict) or set(failure)!={'reason','elapsed_s','threshold_s','observation_sequence'}:
                    raise ValueError('invalid failure checkpoint')
                if failure['reason']!='no_progress': raise ValueError('invalid failure reason')
                for key in ('elapsed_s','threshold_s'):
                    if type(failure[key]) not in (int,float) or not math.isfinite(failure[key]) or failure[key]<=0:
                        raise ValueError('invalid failure timing')
                if failure['threshold_s']>180 or failure['elapsed_s']<failure['threshold_s']:
                    raise ValueError('unproven no-progress interval')
                if type(failure['observation_sequence']) is not int or not -1<=failure['observation_sequence']<=sequence:
                    raise ValueError('invalid failure observation')
            if has_plan:
                validate_steps(task['plan']); index=task['stage_index']; completed=task['completed_stages']
                if type(index) is not int or not 0<=index<len(task['plan']): raise ValueError('invalid stage index')
                if task['plan'][index]!={'goal':task['goal'],'count':task['target_count']}: raise ValueError('active stage mismatch')
                if not isinstance(completed,list) or len(completed)!=index+int(task['status']=='completed'):
                    raise ValueError('invalid completed stage history')
                if task['status']=='completed' and index!=len(task['plan'])-1: raise ValueError('premature plan completion')
                for i,receipt in enumerate(completed):
                    if not isinstance(receipt,dict) or set(receipt)!={'index','goal','count','baseline','progress'}:
                        raise ValueError('invalid stage receipt')
                    if receipt['index']!=i or {k:receipt[k] for k in ('goal','count')}!=task['plan'][i]:
                        raise ValueError('stage history mismatch')
                    if any(type(receipt[k]) is not int or receipt[k]<0 for k in ('index','baseline','progress')) or receipt['progress']<receipt['count']:
                        raise ValueError('unproven stage completion')
            if task['status']=='running': task['status']='paused'
        elif value['schema']!=1: raise ValueError('missing task')
        state.task=task; state.last_observation=sequence
        return state

    def check(self, command):
        validate_command(command)
        op=command['op']
        if op in ('enqueue','reprioritize'): raise ValueError('command requires scheduled task session')
        if op in ('request','request_plan'):
            if self.task is not None and self.task['status'] not in ('completed','cancelled'):
                raise ValueError('session already has a task')
            return
        if self.task is None or command['task_id']!=self.task['task_id']:
            raise ValueError('unknown task')
        if command['expected_revision']!=self.task['revision']: raise ValueError('stale revision')
        valid={'modify':('running','paused'), 'modify_plan':('running','paused'), 'pause':('running',),
               'resume':('paused',), 'cancel':('running','paused')}
        if self.task['status'] not in valid[op]: raise ValueError('invalid task state for '+op)
        if op=='modify_plan' and self.task.get('stage_index',0)+len(command['steps'])>8:
            raise ValueError('plan exceeds 8 total stages')

    def apply(self, command, inventory):
        self.check(command)
        op=command['op']
        if op in ('request','modify','request_plan','modify_plan'):
            previous=self.snapshot()
            revision=1 if self.task is None else self.task['revision']+1
            status='paused' if op in ('modify','modify_plan') and self.task and self.task['status']=='paused' else 'running'
            stage=command['steps'][0] if op in ('request_plan','modify_plan') else command
            text,items=GOALS[stage['goal']]
            self.task={'task_id':command['task_id'], 'revision':revision, 'status':status,
                       'goal':stage['goal'], 'policy_goal':text, 'target_items':list(items),
                       'target_count':stage['count'], 'baseline':sum(inventory.get(k,0) for k in items),
                       'progress':0}
            if op=='request_plan':
                self.task.update(plan=copy.deepcopy(command['steps']),stage_index=0,completed_stages=[])
            elif op=='modify_plan' or (op=='modify' and previous and 'plan' in previous):
                index=previous.get('stage_index',0); receipts=previous.get('completed_stages',[])
                if op=='modify_plan': plan=previous.get('plan',[])[:index]+command['steps']
                else:
                    plan=previous['plan']; plan[index]={'goal':stage['goal'],'count':stage['count']}
                self.task.update(plan=copy.deepcopy(plan),stage_index=index,completed_stages=copy.deepcopy(receipts))
        else:
            self.task['status']={'pause':'paused','resume':'running','cancel':'cancelled'}[op]
            if op=='resume': self.task.pop('failure',None)
        return self.snapshot()

    def pause_for_no_progress(self,elapsed_s,threshold_s):
        if self.task is None or self.task['status']!='running': raise ValueError('running task required')
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in (elapsed_s,threshold_s)) or not 0<threshold_s<=180 or elapsed_s<threshold_s:
            raise ValueError('invalid no-progress interval')
        self.task.update(status='paused',revision=self.task['revision']+1,
                         failure={'reason':'no_progress','elapsed_s':elapsed_s,'threshold_s':threshold_s,
                                  'observation_sequence':self.last_observation})
        return self.snapshot()

    def observe(self, inventory, sequence):
        if sequence<=self.last_observation: raise ValueError('observation sequence must increase')
        self.last_observation=sequence
        if self.task is None or self.task['status'] not in ('running','paused'): return None
        self.task['progress']=max(0,sum(inventory.get(k,0) for k in self.task['target_items'])-self.task['baseline'])
        if self.task['status']=='running' and self.task['progress']>=self.task['target_count']:
            if 'plan' in self.task:
                task=self.task; index=task['stage_index']
                receipt={'index':index,'goal':task['goal'],'count':task['target_count'],
                         'baseline':task['baseline'],'progress':task['progress']}
                task['completed_stages'].append(receipt)
                if index+1<len(task['plan']):
                    stage=task['plan'][index+1]; text,items=GOALS[stage['goal']]
                    task.update(stage_index=index+1,revision=task['revision']+1,goal=stage['goal'],
                                target_count=stage['count'],policy_goal=text,target_items=list(items),
                                baseline=sum(inventory.get(k,0) for k in items),progress=0)
                    return {'event':'plan_stage_completed','completed_stage':copy.deepcopy(receipt),
                            'observation_sequence':sequence,'task':self.snapshot()}
            self.task['status']='completed'
            return {'event':'inventory_target_reached','observation_sequence':sequence,
                    'task':self.snapshot(),
                    'task_id':self.task['task_id'],'revision':self.task['revision'],
                    'progress':self.task['progress'],'target_count':self.task['target_count']}
        return None


class CommandBus:
    """Enqueue invalidates old inference; dispatch and enqueue are serialized.

    In-flight environment calls may finish. Acknowledgment happens on the worker
    after releasing inputs, not when a file is merely written or queued.
    """
    def __init__(self):
        self.lock=threading.RLock(); self.queue=deque(); self.generation=0; self.seen={}

    def submit(self, command):
        validate_command(command)
        with self.lock:
            key=command['command_id']
            if key in self.seen:
                if self.seen[key]!=command: raise ValueError('command_id reused with different content')
                return False
            value=copy.deepcopy(command)
            self.seen[key]=value; self.queue.append(value); self.generation+=1
            return True

    def token(self):
        with self.lock: return self.generation

    def pending(self):
        with self.lock: return bool(self.queue)

    def drain(self):
        with self.lock:
            values=list(self.queue); self.queue.clear(); return values

    def dispatch(self, token, fn):
        with self.lock:
            if token!=self.generation or self.queue: return False,None
            return True,fn()
