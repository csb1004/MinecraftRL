"""Priority goal program and persistent inner RM for the official STEVE-1 actor.

Current executable collection conditions: wood (fixed observer) and dirt (official
visual goal). Other declared goals remain blocked; no scripted motor fallback.
"""
import hashlib
import io
from copy import deepcopy
from pathlib import Path

from .goal_contract import read_json
from .goal_spec import TARGETS, condition_key, expected_prompt, policy_condition
from .goal_queue import GoalQueue
from .interaction_progress import InteractionProgress
from .logo_wood import admit
from .memory import atomic_json
from .policies import Steve1Policy
from .rl_collect import RLEnvironment
from .rl_core import LOGS
from .rm_conditioning import ConditioningPolicy, StageSelector
from .rm_logo_conditioning import load_bank as load_wood_bank
from .runtime import write_json

CAPABILITIES = {skill:items for skill,items,_ in TARGETS.values()}


def load_bank(path):
    import numpy as np
    path=Path(path)
    embeddings, metadata=load_wood_bank(path/'wood')
    name, expected=Steve1Policy.OFFICIAL_VISUAL['collect dirt']
    raw=(path/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('official dirt visual goal hash mismatch')
    vector=np.load(io.BytesIO(raw),allow_pickle=False)
    if vector.shape!=(1,512) or vector.dtype!=np.float32 or not np.isfinite(vector).all():
        raise ValueError('invalid dirt goal vector')
    embeddings['dirt']=vector
    metadata['prompts']=dict(metadata['prompts'],dirt='collect dirt [official video embedding]')
    metadata['embeddings']['dirt']=dict(file_sha256=expected,
        vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),source='official_video_embedding')
    return embeddings, metadata


class QueueReward:
    def __init__(self, queue, server):
        self.queue, self.server=queue,server

    @property
    def completed(self):return self.queue.completed

    def observe(self, inventory):
        current=self.server()
        if inventory != current['server_inventory']:
            raise ValueError('queue reward requires authoritative inventory')
        return self.queue.observe(current)


class ProgramEnvironment(RLEnvironment):
    def __init__(self, env, policy, **kwargs):
        super().__init__(env,policy,**kwargs)
        policy.program_env=self

    def initial_reward(self, server):
        self.queue=self.policy.queue
        self.queue.observe(server)
        return QueueReward(self.queue,lambda:self.gui)

    def reset(self):
        obs=super().reset()
        self.update_goal(initial=True)
        return dict(obs,inventory=dict(self.gui['server_inventory']))

    def update_goal(self, initial=False):
        key=self.queue.active
        if not initial and key is not None and self.policy.progress is not None:
            self.policy.update_progress(self.gui, complete=False)
        if not initial and hasattr(self.policy,'apply_boundary'):
            self.policy.apply_boundary()
        if self.queue.completed:
            self.policy.observe_stage('complete')
        else:
            goal=self.queue.select()
            if self.queue.completed:
                self.policy.observe_stage('complete')
                self.terminal=True
            if goal is not None and goal['id']!=self.policy.goal_id:
                self.policy.bind_goal(goal,self.gui)
            elif goal is None and self.policy.selector.current is None:
                # Unbound queue is stopped by prepare_control before inference.
                self.policy.observe_stage('search')
        self.policy.persist_progress()

    def step(self, action):
        previous_goal=self.queue.active
        obs,reward,done,info=super().step(action)
        # Commit already happened under the old goal/condition; next-goal binding
        # only affects the next observation/action, without resetting this world.
        if self.queue.records[previous_goal]['status']=='completed':
            self.policy.rows[-1]['goal_completed']=True
            if self.policy.progress is not None:
                self.policy.update_progress(self.gui,complete=True)
        if not done:
            self.update_goal()
        else:
            self.policy.persist_progress()
        self.policy.rows[-1]['queue_after']=dict(active_goal=self.queue.active,
            completed=self.queue.completed,goal_status=self.queue.records[previous_goal]['status'])
        return dict(obs,inventory=dict(self.gui['server_inventory'])),reward,done,info


class ProgramPolicy(ConditioningPolicy):
    bank_loader=staticmethod(load_bank)
    environment_type=ProgramEnvironment

    def __init__(self,*args,program,**kwargs):
        self.program_path=Path(program)
        self.queue=GoalQueue(read_json(program),CAPABILITIES)
        if (self.queue.goal_spec is not None and self.queue.goal_spec['source']['schema']=='goal_spec_v2'
                and not getattr(self,'execution_contracts',False)):
            raise ValueError('goal_spec_v2 requires the E2 execution path')
        if self.queue.goal_spec is not None and (
                'quantity_log2' not in getattr(self,'context_fields',()) or
                not getattr(self,'variant','').endswith('-on')):
            raise ValueError('goal_spec_v1 requires enabled S2/R1 structured policy context')
        self.goal_id=None
        self.progress=None
        self.phase_snapshot=None
        super().__init__(*args,**kwargs)
        if self.queue.goal_spec is not None:
            # Bank loaders verify vectors/tokens; also pin the target-to-key meaning.
            keys = {condition_key(c['target_ref'], phase, getattr(self,'task','collection'))
                    for c in self.queue.goal_spec['contracts'].values() for phase in ('search','approach')}
            if any(k not in self.embeddings or self.prompts.get(k)!=expected_prompt(k) for k in keys):
                raise ValueError('goal specification and fixed embedding bank disagree')
        self.validators,self.validator_review=admit(self.bank_metadata['proposal'])

    def bind_goal(self, goal, server):
        self.goal_id=goal['id']
        self.selector=StageSelector()
        self.progress=None
        self.phase_snapshot=None
        self.goal_baseline=deepcopy(server)
        if goal['skill']=='wood_memory':
            values=self.validators.observe(server,server)
            if any(v is None for v in values.values()):
                raise ValueError('missing wood phase evidence')
            self.progress=InteractionProgress(tick=server['server_tick'],
                interactions=server['server_log_harvests'],inventory_count=sum(server['server_inventory'].get(n,0) for n in LOGS),
                known=values['target_known'],reachable=values['target_reachable'],
                **self.queue.program.get('progress_limits',{}))
            self.phase_snapshot=self.progress.snapshot()
            self.observe_stage(self.progress.stage)
        else:
            self.observe_stage('collect')

    def update_progress(self, server, complete):
        values=self.validators.observe(server,self.goal_baseline)
        row=self.progress.observe(tick=server['server_tick'],interactions=server['server_log_harvests'],
            inventory_count=sum(server['server_inventory'].get(n,0) for n in LOGS),
            known=values['target_known'],reachable=values['target_reachable'],complete=complete)
        self.phase_snapshot=row
        if row['blocked']:
            self.queue.block(row['blocked'],row)
        elif not complete:
            self.observe_stage(row['stage'])

    def persist_progress(self, force=False):
        if getattr(self,'output',None):
            meaningful=(self.queue.active,tuple((k,r['status'],r['progress'],r['revision'])
                for k,r in self.queue.records.items()),len(self.queue.events),
                None if self.phase_snapshot is None else (self.phase_snapshot['stage'],
                    self.phase_snapshot['pending_result_since'],self.phase_snapshot['blocked']))
            if not force and meaningful==getattr(self,'_saved_progress',None):return
            atomic_json(self.output/'goal-queue.json',self.queue.snapshot())
            write_json(self.output/'skill-progress.json',self.phase_snapshot)
            self._saved_progress=meaningful
            if self.queue.active is None and not self.queue.completed:
                write_json(self.output/'learning-request.json',dict(
                    reason='no_executable_goal',queue=self.queue.snapshot(),last_skill_progress=self.phase_snapshot,
                    executed=False,API_calls=0,requires_new_training_protocol=True))

    def reset(self, goal=None):
        super().reset('collect wood')  # Shared initialization only; conditions come from the queue.
        self.condition_metadata.update(program_sha256=self.queue.program_sha256,
            program=self.queue.program,requested_program_goal=goal,
            stage_source='persistent_interaction_progress_over_fixed_observer',
            goal_handoff='completion boundary; recurrent state preserved',
            subgoal_conditions='wood search/root; dirt official visual',
            phase_context_training='not trained by frozen program evaluation')
        if self.queue.goal_spec is not None:
            self.condition_metadata['goal_spec'] = deepcopy(self.queue.goal_spec)
        write_json(self.output/'rm-condition.json',self.condition_metadata)

    def goal_key(self):
        if self.queue.active is None or self.queue.completed:
            raise InterruptedError('no executable goal or program completed')
        if self.queue.goal_spec is not None:
            return self.goal_binding()['goal_key']
        if self.queue.goals[self.queue.active]['skill']=='dirt_fixed':return 'dirt'
        return 'seek' if self.selector.current=='search' else 'root'

    def goal_binding(self):
        condition = policy_condition(self.queue, self.selector.current, getattr(self,'task','collection'))
        key = condition['goal_key']
        if key not in self.embeddings or self.prompts.get(key) != condition['prompt']:
            raise ValueError('goal specification and policy condition disagree')
        return condition

    def prepare_control(self, *, request_stop, emit, **kwargs):
        if self.queue.active is None and not self.queue.completed:
            emit('goal_queue_blocked',queue=self.queue.snapshot(),phase=self.phase_snapshot)
            request_stop('planning_paused')

    def row_context(self):
        goal=self.queue.goals[self.goal_id]
        result = dict(goal_id=self.goal_id,goal_revision=self.queue.records[self.goal_id]['revision'],
                    objective=goal['objective'],skill=goal['skill'],program_sha256=self.queue.program_sha256,
                    progress_before=self.phase_snapshot)
        if self.queue.goal_spec is not None:
            result['goal_binding'] = self.goal_binding()
        return result

    def close(self):
        self.persist_progress(force=True)
        super().close()
