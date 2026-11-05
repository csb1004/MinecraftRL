"""E2: explicit goal meaning, whole-program RL and committed result evidence."""
from copy import deepcopy
import json

from .conditioned_head import ConditionedHead
from .recovery_learning import RecoveryPolicy, RecoveryEnvironment, CONTEXT_FIELDS as RECOVERY_FIELDS
from .execution_contract import BoundaryUpdates, execution_request, classify_failure
from .execution_returns import (EXTRA_FIELDS, REWARD_PROFILE, context_tail, state_record,
    program_potential, learning_reward, program_returns, validate_execution_trace)
from .rl_policy import state_fingerprint
from .runtime import write_json

SCOPE='execution_contract_e2'
CONTEXT_FIELDS=(*RECOVERY_FIELDS,*EXTRA_FIELDS)
VARIANTS=('nonlinear-on',)
ARCHIVE='execution-trajectory.pt'


class ExecutionHead(ConditionedHead):
    def __init__(self,base,width,variant):
        if variant not in VARIANTS:raise ValueError('E2 requires explicit context')
        super().__init__(base,width,variant,context_dim=len(CONTEXT_FIELDS))
        self.base.requires_grad_(False)


class ExecutionEnvironment(RecoveryEnvironment):
    def step(self,action):
        obs,reward,done,info=super().step(action)
        p=self.policy;row=p.learning_rows[-1]
        after=state_record(p.queue)
        before_phi=row['potential_before'];after_phi=program_potential(p.queue)
        row.update(terminal=self.terminal,execution_after=after,program_reward=float(p.queue.completed),potential_after=after_phi,
            learning_reward=learning_reward(before_phi,after_phi,p.queue.completed,self.terminal),
            boundary_updates=deepcopy(p.last_boundary_updates))
        p.last_boundary_updates=[]
        p.rows[-1].update({k:deepcopy(row[k]) for k in ('execution_request','execution_before','execution_after',
            'program_reward','potential_before','potential_after','learning_reward','boundary_updates','terminal')})
        event=dict(step=row['step'],request=row['execution_request'],outcome=after,
                   program_reward=row['program_reward'],learning_reward=row['learning_reward'],terminal=row['terminal'])
        with (p.output/'execution-events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(event,sort_keys=True)+'\n')
        return obs,reward,bool(done or self.terminal),info


class ExecutionPolicy(RecoveryPolicy):
    scope=SCOPE
    context_fields=CONTEXT_FIELDS
    archive_prefix='execution'
    environment_type=ExecutionEnvironment
    max_root_goals=None
    execution_contracts=True

    def __init__(self,*args,**kwargs):
        self.boundaries=BoundaryUpdates();self.last_boundary_updates=[]
        super().__init__(*args,**kwargs)
        if self.queue.goal_spec is None or self.queue.goal_spec['source']['schema']!='goal_spec_v2':
            raise ValueError('E2 requires goal_spec_v2')

    def make_actor(self,base,width):
        actor=ExecutionHead(base,width,self.variant)
        self.official_motor_sha256=state_fingerprint(actor.base.state_dict())
        return actor

    def apply_boundary(self):
        events=self.boundaries.apply(self.queue,inference_pending=self.pending is not None)
        self.last_boundary_updates.extend(events)

    def semantic_context(self,evidence):
        values,description=super().semantic_context(evidence)
        self.current_request=execution_request(self.queue,self.selector.current,self.task)
        self.current_before=state_record(self.queue)
        self.current_potential=program_potential(self.queue)
        values+=context_tail(self.queue,self.goal_embed[0].tolist())
        description.update(schema=SCOPE,execution_request=self.current_request)
        return values,description

    def act(self,screen):
        action=super().act(screen)
        self.learning_pending.update(execution_request=deepcopy(self.current_request),
            execution_before=deepcopy(self.current_before),potential_before=self.current_potential)
        return action

    def reset(self,goal=None):
        super().reset(goal)
        self.condition_metadata.update(learning_scope=SCOPE,reward_profile=REWARD_PROFILE,
            goal=goal or 'Complete goal queue: '+', '.join(self.queue.goals),
            phase_context_training='E2 contract/goal context; independent residual/value training',
            explicit_goal_vector=True,execution_contract='acquire_or_possess_with_active_inventory_invariants',
            information_condition='privileged_inventory_and_masked_fixed_phase',
            boundary_updates='priority_only_after_committed_action; semantic_changes_require_new_run')
        write_json(self.output/'rm-condition.json',self.condition_metadata)

    def close(self):
        super().close()
        if hasattr(self,'saved_learning'):
            self.saved_learning.update(reward_profile=REWARD_PROFILE,boundary_events=self.boundaries.events)

    def archive_learning(self,report):
        super().archive_learning(report)
        reason=(self.failure or {}).get('reason') or self.end_reason or report['status']
        blocked=[r['reason'] for r in self.queue.records.values() if r['status']=='blocked']
        if 'preservation_violated' in blocked:reason='preservation_violated'
        report.update(execution_scope=SCOPE,program_reward=sum(r.get('program_reward',0.) for r in self.learning_rows),
            execution_training_eligible=report.get('s2_training_eligible',False))
        if not self.queue.completed:
            evidence=classify_failure(dict(reason=reason,execution_valid=report.get('execution_valid'),
                goal_id=self.goal_id,last_progress=self.phase_snapshot,
                queue=self.queue.snapshot(),phase_observation='fixed; unavailable facts remain unknown'))
            write_json(self.output/'failure-evidence.json',evidence)
            report['failure_category']=evidence['category']


# Explicit trainer profile; E1 retains its original profile/defaults.
actor_type=ExecutionHead
returns=program_returns
validate=validate_execution_trace
