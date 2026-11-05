"""Common pickup intent, masked evidence and frozen-motor residual learning."""
import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

from .conditioned_head import ConditionedHead
from .semantic_goal import CONTEXT_FIELDS as S2_FIELDS, context
from .s2_learning import SemanticPolicy, SemanticEnvironment
from .rm_program import load_bank as program_bank
from .rm_conditioning import StageSelector
from .runtime import write_json
from .rl_policy import state_fingerprint
from .goal_spec import PICKUP_PROMPTS

SCOPE = 'common_recovery_r1'
CONTEXT_FIELDS = (*S2_FIELDS, 'phase_valid', 'pending_valid')
PROMPTS = PICKUP_PROMPTS
VARIANTS = ('nonlinear-off', 'nonlinear-on')


def validate_args(args):
    active=getattr(args,'recovery_variant',None)
    prepare=getattr(args,'recovery_prepare',False)
    world=getattr(args,'recovery_world',None)
    checkpoint=getattr(args,'recovery_checkpoint',None)
    extra=(getattr(args,'recovery_sampling_seed',None) is not None or
           getattr(args,'recovery_task','collection')!='collection' or
           getattr(args,'recovery_split','development')!='development' or
           getattr(args,'recovery_action_budget',1200)!=1200)
    if not (active or prepare or world or checkpoint or extra):return
    if (getattr(args,'rm_conditioning',None)!='goal-queue' or
            any(getattr(args,k,None) for k in ('s2_variant','s2_checkpoint','s1_record','s1_checkpoint','rl_collect'))):
        raise ValueError('recovery requires isolated goal-queue mode')
    if prepare:
        if active or world or checkpoint or extra:raise ValueError('neutral preparation cannot load a policy or world')
        return
    if active not in VARIANTS or world is None:raise ValueError('explicit recovery variant and saved start required')
    root=Path(args.root).resolve()
    for path in (world,checkpoint):
        if path is not None and (not Path(path).resolve().is_relative_to(root/'results') or not Path(path).is_file()):
            raise ValueError('recovery input must be an existing project-local result')
    if not 1<=getattr(args,'recovery_action_budget',1200)<=1200:raise ValueError('invalid recovery action budget')


def recovery_context(queue, phase, progress, evidence, last_action=None, repeat=0,
                     last_progress_tick=None, *, task):
    if task not in ('pickup','collection'):raise ValueError('unknown recovery task')
    values,desc=context(queue,phase,progress,evidence,last_action,repeat,last_progress_tick)
    wood=queue.goals[queue.active]['skill']=='wood_memory'
    pending_valid=wood and progress is not None
    if task=='pickup':
        values[:4]=[0.,0.,0.,1.]
        phase_valid=True
        desc.update(stage='recover_result',stage_source='requested_skill')
    elif not wood:
        values[:4]=[0.]*4
        phase_valid=False
        desc.update(stage='unknown',stage_source='unavailable_fine_phase')
    else:
        phase_valid=True
        desc['stage_source']='fixed_observer_interaction_history'
    if task=='pickup' or desc['stage']=='recover_result':
        # Existing known/reachable facts refer to a block, never a dropped item.
        values[9:13]=[0.]*4
        desc['evidence']=dict(target_known=None,target_reachable=None)
    if not pending_valid: values[8]=values[14]=0.
    values += [float(phase_valid),float(pending_valid)]
    desc.update(schema=SCOPE,phase_valid=phase_valid,pending_valid=pending_valid,
                task=task,item_location='unobserved',actor_fixture_metadata=False)
    return values,desc


class RecoveryHead(ConditionedHead):
    def __init__(self,base,width,variant):
        if variant not in VARIANTS:raise ValueError('unsupported recovery head')
        super().__init__(base,width,variant,context_dim=len(CONTEXT_FIELDS))
        self.base.requires_grad_(False)


def load_bank(path):
    import numpy as np
    embeddings,metadata=program_bank(path)
    folder=Path(path)/'pickup'
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest.get('prompts')!=PROMPTS or manifest.get('schema')!=SCOPE:
        raise ValueError('recovery goal bank identity mismatch')
    if manifest.get('official_weights')!=metadata['official_weights']:
        raise ValueError('recovery embeddings official source mismatch')
    for key in PROMPTS:
        raw=(folder/(key+'.npy')).read_bytes();entry=manifest['embeddings'][key]
        if hashlib.sha256(raw).hexdigest()!=entry['file_sha256']:raise ValueError('pickup bank file changed')
        vector=np.load(io.BytesIO(raw),allow_pickle=False)
        if vector.shape!=(1,512) or vector.dtype!=np.float32 or not np.isfinite(vector).all():
            raise ValueError('invalid pickup goal vector')
        if hashlib.sha256(vector.tobytes()).hexdigest()!=entry['vector_sha256']:raise ValueError('pickup vector mismatch')
        embeddings[key]=vector
        metadata['prompts'][key]=PROMPTS[key];metadata['embeddings'][key]=entry
    return embeddings,metadata


class RecoveryEnvironment(SemanticEnvironment):
    def reset(self):
        obs=super().reset()
        policy=self.policy
        if policy.task=='pickup' and any(self.gui['server_inventory'].get(n,0) for g in self.queue.goals.values() for n in g['items']):
            raise ValueError('pickup target already acquired during startup; no performance sample')
        world=getattr(self.env,'recovery_world',None)
        if not world:raise ValueError('recovery requires verified saved start')
        if world.get('experimental_split')!=policy.split:raise ValueError('world and policy split mismatch')
        location=obs.get('location_stats',{})
        if 'position' in world:
            expected=world['position']
            actual=[float(location[k]) for k in ('xpos','ypos','zpos')]
            if max(abs(a-b) for a,b in zip(actual,expected))>.02:
                raise ValueError('fixture player position differs after restoration')
        write_json(policy.output/'recovery-start.json',dict(world_sha256=world['sha256'],
            server_inventory=self.gui['server_inventory'],location=location,split=policy.split,
            fixture_setup_only=True,actor_receives_world_metadata=False,task=policy.task))
        return obs


class RecoveryPolicy(SemanticPolicy):
    scope=SCOPE
    context_fields=CONTEXT_FIELDS
    archive_prefix='recovery'
    bank_loader=staticmethod(load_bank)
    environment_type=RecoveryEnvironment
    max_root_goals=1

    def __init__(self,*args,task='collection',**kwargs):
        if task not in ('pickup','collection'):raise ValueError('invalid recovery task')
        self.task=task
        super().__init__(*args,**kwargs)
        if self.max_root_goals is not None and len(self.queue.goals)!=self.max_root_goals:
            raise ValueError('R1 validates one acquisition goal per episode; multi-goal transfer needs a separate protocol')
        if state_fingerprint(self.agent.policy.pi_head.base.state_dict())!=self.official_motor_sha256:
            raise ValueError('recovery checkpoint changed the frozen official motor head')

    def make_actor(self,base,width):
        actor=RecoveryHead(base,width,self.variant)
        self.official_motor_sha256=state_fingerprint(actor.base.state_dict())
        return actor

    def bind_goal(self,goal,server):
        if self.task=='collection':return super().bind_goal(goal,server)
        self.goal_id=goal['id'];self.selector=StageSelector()
        self.progress=self.phase_snapshot=None
        self.goal_baseline=deepcopy(server)
        self.observe_stage('collect')

    def semantic_context(self,evidence):
        return recovery_context(self.queue,self.selector.current,self.phase_snapshot,evidence,
            self.monitor.last_action,self.monitor.repeat,self.monitor.last_progress_tick,task=self.task)

    def goal_key(self):
        if self.queue.goal_spec is not None:
            return super().goal_key()
        if self.task=='collection':return super().goal_key()
        if self.queue.active is None or self.queue.completed:raise InterruptedError('pickup goal ended')
        return 'pickup_dirt' if self.queue.goals[self.queue.active]['skill']=='dirt_fixed' else 'pickup_wood'

    def reset(self,goal=None):
        super().reset(goal)
        self.condition_metadata.update(recovery_task=self.task,trainable_scope='residual_adapter_and_value_only',
            weights='official_frozen_backbone_and_motor_with_recovery_adapter',
            observation_contract='fixed_observer; no dropped entity coordinates or visibility labels')
        write_json(self.output/'rm-condition.json',self.condition_metadata)

    def archive_learning(self,report):
        if hasattr(self,'saved_learning'):
            self.saved_learning.update(recovery_task=self.task,world_sha256=self.program_env.env.recovery_world['sha256'],
                official_motor_sha256=self.official_motor_sha256)
        super().archive_learning(report)
        report.update(recovery_scope=self.scope,recovery_task=self.task,recovery_variant=self.variant,
                      recovery_checkpoint_sha256=self.checkpoint_sha256)
