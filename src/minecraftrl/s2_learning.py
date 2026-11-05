"""Opt-in common goal-conditioned learning; original S1 behavior stays intact."""
import hashlib
import io
from pathlib import Path
from .conditioned_head import ConditionedHead, ConditionedValue, VARIANTS
from .semantic_goal import CONTEXT_DIM, CONTEXT_FIELDS, ResultMonitor, context
from .s1_learning import LearningPolicy
from .rm_program import ProgramPolicy, ProgramEnvironment
from .rl_policy import cpu_tree, state_fingerprint
from .runtime import write_json

SCOPE = 's2_state_change_v1'


class SemanticEnvironment(ProgramEnvironment):
    def reset(self):
        obs = super().reset()
        self.policy.monitor.observe(self.queue)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        policy = self.policy
        evidence = policy.monitor.observe(self.queue, action)
        reason = None
        if not self.queue.completed:
            if self.queue.active is None:
                reason = self.queue.records[policy.goal_id]['reason'] or 'no_executable_goal'
            elif self.queue.goals[self.queue.active]['skill']=='dirt_fixed' and evidence['no_inventory_progress_ticks']>=600:
                self.queue.block('no_confirmed_progress', evidence)
                reason = 'no_confirmed_progress'
        if reason is not None:
            # A declared finite task-budget failure, not an external cancellation.
            self.terminal = True
            policy.rows[-1]['terminal'] = True
            policy.learning_rows[-1]['terminal'] = True
            policy.failure = dict(reason=reason, tick=self.gui['server_tick'], monitor=evidence,
                last_progress=policy.phase_snapshot, queue=self.queue.snapshot(),
                prior_attempts=policy.monitor_events[-12:], executed=False, API_calls=0,
                allowed_resolution='validated_rm_revision_or_separate_training_request',
                unavailable_observations='unknown')
            policy.persist_progress(force=True)
            done = True
        elif not done and not self.queue.completed and len(policy.rows)>=policy.action_budget:
            # Truncation: preserve final recurrent state and bootstrap in close().
            reason = 'action_budget'
            done = True
        if reason:
            policy.end_reason = reason
            info = dict(info, s2_end_reason=reason)
            policy.rows[-1]['queue_after'] = dict(active_goal=self.queue.active,
                completed=self.queue.completed,goal_status=self.queue.records[policy.rows[-1]['goal_id']]['status'])
        if reason or reward or len(policy.rows)%120==0:
            policy.monitor_events.append(dict(step=len(policy.rows), reward=reward, reason=reason, **evidence))
        return obs, reward, done, info


class SemanticPolicy(LearningPolicy):
    environment_type = SemanticEnvironment
    scope = SCOPE
    context_fields = CONTEXT_FIELDS
    archive_prefix = 's2'

    def make_actor(self, base, width):
        return ConditionedHead(base, width, self.variant, context_dim=len(self.context_fields))

    def semantic_context(self, evidence):
        return context(self.queue,self.selector.current,self.phase_snapshot,evidence,
            self.monitor.last_action,self.monitor.repeat,self.monitor.last_progress_tick)

    def __init__(self,*args,variant='linear-on',checkpoint=None,split='development',action_budget=1200,**kwargs):
        if variant not in VARIANTS or not 1 <= action_budget <= 1200:
            raise ValueError('invalid bounded S2 condition')
        self.variant, self.split, self.action_budget = variant, split, action_budget
        ProgramPolicy.__init__(self,*args,**kwargs)
        self.monitor, self.monitor_events = ResultMonitor(), []
        self.failure = self.end_reason = None
        self.learning_rows, self.learning_pending = [], None
        self.base_sha256 = self.bank_metadata['official_weights']['steve1.weights']
        width = self.agent.policy.net.output_latent_size()
        self.agent.policy.pi_head = self.make_actor(self.agent.policy.pi_head,width).to(self.device).requires_grad_(False)
        self.value = ConditionedValue(width,variant,context_dim=len(self.context_fields)).to(self.device).requires_grad_(False)
        self.feature_width = width+len(self.context_fields)
        self.checkpoint_sha256 = None
        if checkpoint is not None:
            raw = Path(checkpoint).read_bytes()
            saved = self.torch.load(io.BytesIO(raw),map_location='cpu')
            if (saved.get('scope')!=self.scope or saved.get('variant')!=variant or
                    saved.get('base_model_sha256')!=self.base_sha256 or saved.get('context_fields')!=list(self.context_fields)
                    or saved.get('guidance')!=6.):
                raise ValueError('incompatible S2 checkpoint')
            self.agent.policy.pi_head.load_state_dict(saved['heads']['actor'],strict=True)
            self.value.load_state_dict(saved['heads']['value'],strict=True)
            self.checkpoint_sha256 = hashlib.sha256(raw).hexdigest()
        self.initial_heads = cpu_tree(dict(actor=self.agent.policy.pi_head.state_dict(),value=self.value.state_dict()))
        self.initial_heads_sha256 = state_fingerprint(self.initial_heads)

    def reset(self,goal=None):
        ProgramPolicy.reset(self,goal)
        self.condition_metadata.update(learning_scope=self.scope,variant=self.variant,context_fields=list(self.context_fields),
            checkpoint_sha256=self.checkpoint_sha256,split=self.split,action_budget=self.action_budget,
            phase_context_training='S2 state-change context; explicit off ablation',
            weights='official_frozen_backbone_with_s2_heads',optimizer_updates_during_game=0)
        write_json(self.output/'rm-condition.json',self.condition_metadata)

    def features(self,screen):
        torch = self.torch
        inputs = self.agent._env_obs_to_agent({'pov':screen},self.goal_embed)
        inputs = {k:torch.cat([v,v],0).unsqueeze(1) for k,v in inputs.items()}
        inputs['mineclip_embed'][1] = 0
        first = torch.zeros((2,1),dtype=torch.bool,device=self.device)
        with torch.no_grad(),torch.cuda.amp.autocast():
            (pi,_),state = self.agent.policy.net(inputs,self.agent.hidden_state,context={'first':first})
        evidence = dict(target_known=None,target_reachable=None)
        if self.queue.goals[self.queue.active]['skill']=='wood_memory':
            observed = self.validators.observe(self.program_env.gui,self.goal_baseline)
            evidence = {k:observed[k] for k in evidence}
        values, self.current_contract = self.semantic_context(evidence)
        extra = torch.tensor(values,device=self.device,dtype=torch.float32).view(1,1,len(self.context_fields)).expand(2,1,len(self.context_fields))
        return torch.cat([pi.float(),extra],-1),state,values

    def row_context(self):
        return dict(ProgramPolicy.row_context(self),semantic_goal=getattr(self,'current_contract',None),variant=self.variant)

    def close(self):
        terminal = bool(getattr(self,'program_env',None) and self.program_env.terminal)
        bootstrap = 0.
        bootstrap_features = None
        if self.learning_rows and not terminal and self.queue.active is not None:
            self.goal_embed = self.embeddings[self.goal_key()].copy()
            features,_,_ = self.features(self.program_env.last['pov'])
            bootstrap_features = features.detach().cpu()
            with self.torch.no_grad(): bootstrap = self.value(features[0:1]).item()
        self.saved_learning = dict(scope=self.scope,split=self.split,variant=self.variant,rows=self.learning_rows,
            bootstrap=bootstrap,bootstrap_features=bootstrap_features,terminal=terminal,end_reason=self.end_reason,
            initial_heads=self.initial_heads,initial_heads_sha256=self.initial_heads_sha256,
            base_model_sha256=self.base_sha256,feature_width=self.feature_width,context_dim=len(self.context_fields),
            context_fields=list(self.context_fields),guidance=6.,program=self.queue.program,
            program_sha256=self.queue.program_sha256,bank_metadata=self.bank_metadata,
            checkpoint_sha256=self.checkpoint_sha256,recurrent_commit='dispatch_only',sampling_seed=self.seed)
        if self.queue.goal_spec is not None:
            self.saved_learning['goal_spec'] = self.queue.snapshot()['goal_spec']
        write_json(self.output/'result-monitor.json',dict(events=self.monitor_events,failure=self.failure,end_reason=self.end_reason))
        self.value = None
        self.learning_pending = None
        ProgramPolicy.close(self)
        if self.failure: write_json(self.output/'learning-request.json',self.failure)

    def archive_learning(self,report):
        if not hasattr(self,'saved_learning'): return
        data = self.saved_learning
        eligible = (report['execution_valid'] and report['status'] in ('target_reached','time_limit','environment_done')
                    and bool(data['rows']) and self.split=='train')
        data.update(eligible=eligible,status=report['status'])
        self.torch.save(data,self.output/(self.archive_prefix+'-trajectory.pt'))
        report.update(s2_training_eligible=eligible,s2_scope=self.scope,s2_variant=self.variant,
            s2_split=self.split,s2_checkpoint_sha256=self.checkpoint_sha256,s2_end_reason=self.end_reason,
            s2_trajectory_steps=len(data['rows']),s2_bootstrap=data['bootstrap'],weights_updated_during_game=False)
