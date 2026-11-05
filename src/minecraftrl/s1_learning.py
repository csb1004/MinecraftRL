"""S1 head-only PPO data path, separate from retired experiment entrypoints.

The image/goal-conditioned recurrent network stays frozen. Six reusable progress
features reach both action and value heads; zero actor columns preserve the base
head at initialization. Only dispatched actions become learning transitions.
"""
import hashlib
import math
from pathlib import Path

from .rl_policy import cpu_tree, extend_context_head, guided, state_fingerprint
from .rm_program import ProgramPolicy
from .runtime import jsonable, write_json

SCOPE='s1_phase_head_v1'
PHASES=('search','approach','mine','collect')
EXTRA_DIM=6


def phase_features(queue, phase, pending_result=False):
    if queue.active is None:
        raise ValueError('active goal required for learning context')
    goal=queue.goals[queue.active];record=queue.records[queue.active]
    if phase not in PHASES:raise ValueError('invalid active learning phase')
    if type(pending_result) is not bool:raise ValueError('invalid pending result evidence')
    return [float(phase==name) for name in PHASES]+[
        min(1.,record['progress']/goal['count']),float(pending_result)]


def validate_data(data, summary):
    if (data.get('scope')!=SCOPE or data.get('split')!='train' or data.get('eligible') is not True
            or summary.get('execution_valid') is not True or summary.get('worker_exit_code')!=0
            or data.get('status') not in ('target_reached','time_limit','environment_done')
            or summary.get('status')!=data['status'] or not data.get('rows')):
        raise ValueError('only verified new training rollouts may update the actor')
    rows=data['rows']
    if len(rows)!=summary.get('steps') or data.get('recurrent_commit')!='dispatch_only':
        raise ValueError('unverified learning transitions')
    for i,row in enumerate(rows):
        if row['step']!=i+1 or len(row['phase_context'])!=EXTRA_DIM or row['phase'] not in PHASES:
            raise ValueError('invalid learning row identity')
        if any(not math.isfinite(row[k]) for k in ('old_logp','value','reward')):
            raise ValueError('nonfinite learning row')
        if row['terminal'] and i!=len(rows)-1:raise ValueError('internal terminal transition')
    if bool(rows[-1]['terminal'])!=data['terminal'] or not math.isfinite(data['bootstrap']):
        raise ValueError('invalid final bootstrap boundary')
    if data['terminal'] and data['bootstrap']!=0.:raise ValueError('terminal bootstrap must be zero')
    if state_fingerprint(data['initial_heads'])!=data['initial_heads_sha256']:
        raise ValueError('head identity mismatch')


class LearningPolicy(ProgramPolicy):
    def __init__(self,*args,checkpoint=None,split='development',**kwargs):
        super().__init__(*args,**kwargs)
        self.split=split
        self.learning_rows=[]
        self.learning_pending=None
        self.base_sha256=self.bank_metadata['official_weights']['steve1.weights']
        torch=self.torch
        width=self.agent.policy.net.output_latent_size()
        extend_context_head(self.agent.policy.pi_head,width,EXTRA_DIM)
        self.value=torch.nn.Linear(width+EXTRA_DIM,1).to(self.device)
        torch.nn.init.zeros_(self.value.weight);torch.nn.init.zeros_(self.value.bias)
        self.value.requires_grad_(False)
        self.checkpoint_sha256=None
        if checkpoint is not None:
            raw=Path(checkpoint).read_bytes()
            self.checkpoint_sha256=hashlib.sha256(raw).hexdigest()
            import io
            saved=torch.load(io.BytesIO(raw),map_location='cpu')
            if (saved.get('scope')!=SCOPE or saved.get('base_model_sha256')!=self.base_sha256
                    or saved.get('context_dim')!=EXTRA_DIM or saved.get('guidance')!=6.):
                raise ValueError('incompatible S1 checkpoint')
            self.agent.policy.pi_head.load_state_dict(saved['heads']['actor'],strict=True)
            self.value.load_state_dict(saved['heads']['value'],strict=True)
        self.initial_heads=cpu_tree(dict(actor=self.agent.policy.pi_head.state_dict(),value=self.value.state_dict()))
        self.initial_heads_sha256=state_fingerprint(self.initial_heads)
        self.feature_width=width+EXTRA_DIM

    def reset(self,goal=None):
        super().reset(goal)
        self.condition_metadata.update(learning_scope=SCOPE,context_dim=EXTRA_DIM,
            context_fields=[*PHASES,'goal_fraction','pending_result'],
            checkpoint_sha256=self.checkpoint_sha256,initial_heads_sha256=self.initial_heads_sha256,
            split=self.split,weights='official_frozen_backbone_with_phase_heads',
            optimizer_updates_during_game=0)
        write_json(self.output/'rm-condition.json',self.condition_metadata)

    def features(self,screen):
        torch=self.torch;agent=self.agent
        inputs=agent._env_obs_to_agent({'pov':screen},self.goal_embed)
        inputs={k:torch.cat([v,v],0).unsqueeze(1) for k,v in inputs.items()}
        inputs['mineclip_embed'][1]=0
        # Match the official inference convention; recurrent state itself is only
        # reset at episode start and is committed only after an actual dispatch.
        first=torch.zeros((2,1),dtype=torch.bool,device=self.device)
        with torch.no_grad(),torch.cuda.amp.autocast():
            (pi,_),state=agent.policy.net(inputs,agent.hidden_state,context={'first':first})
        pending=bool(self.phase_snapshot and self.phase_snapshot['pending_result_since'] is not None)
        context=phase_features(self.queue,self.selector.current,pending_result=pending)
        extra=torch.tensor(context,device=self.device,dtype=torch.float32).view(1,1,EXTRA_DIM).expand(2,1,EXTRA_DIM)
        return torch.cat([pi.float(),extra],-1),state,context

    def act(self,screen):
        if self.is_stopped() or self.queue.completed:
            raise InterruptedError('STOP/completion before S1 inference')
        if self.pending is not None:raise RuntimeError('uncommitted S1 action')
        key=self.goal_key();self.goal_embed=self.embeddings[key].copy()
        features,state,context=self.features(screen)
        with self.torch.no_grad():
            pd=guided(self.agent.policy.pi_head,features)
            if any(not self.torch.isfinite(v).all() for v in pd.values()):
                raise ValueError('nonfinite S1 distribution')
            action=self.agent.policy.pi_head.sample(pd,deterministic=False)
            logp=self.agent.policy.pi_head.logprob(action,pd).item()
            value=self.value(features[0:1]).item()
        if not math.isfinite(logp) or not math.isfinite(value):raise ValueError('nonfinite S1 prediction')
        env_action=self.agent._agent_action_to_env({k:v[:,0] for k,v in action.items()})
        self.pending_state=state
        self.pending=dict(stage=self.selector.current,raw_stage=self.raw_stage,goal_key=key,
            prompt=self.prompts[key],embedding_sha256=self.bank_metadata['embeddings'][key]['vector_sha256'],
            env_action=jsonable(env_action),old_logp=logp,value=value,phase_context=context,**self.row_context())
        self.learning_pending=dict(features=features.detach().cpu(),action=cpu_tree(action),
            old_logp=logp,value=value,phase_context=context,phase=self.selector.current,
            goal_id=self.goal_id,goal_key=key,embedding_sha256=self.pending['embedding_sha256'])
        if 'goal_binding' in self.pending:
            self.learning_pending['goal_binding'] = self.pending['goal_binding']
        return env_action

    def commit(self,action,reward,terminal):
        super().commit(action,reward,terminal)
        self.learning_rows.append(dict(self.learning_pending,reward=reward,terminal=terminal,step=len(self.rows)))
        self.learning_pending=None

    def close(self):
        terminal=bool(getattr(self,'program_env',None) and self.program_env.terminal)
        bootstrap=0.
        if self.learning_rows and not terminal and self.queue.active is not None:
            self.goal_embed=self.embeddings[self.goal_key()].copy()
            features,_,_=self.features(self.program_env.last['pov'])
            with self.torch.no_grad():bootstrap=self.value(features[0:1]).item()
        self.saved_learning=dict(scope=SCOPE,split=self.split,rows=self.learning_rows,bootstrap=bootstrap,
            terminal=terminal,initial_heads=self.initial_heads,initial_heads_sha256=self.initial_heads_sha256,
            base_model_sha256=self.base_sha256,feature_width=self.feature_width,context_dim=EXTRA_DIM,
            guidance=6.,program=self.queue.program,program_sha256=self.queue.program_sha256,
            bank_metadata=self.bank_metadata,checkpoint_sha256=self.checkpoint_sha256,
            recurrent_commit='dispatch_only',sampling_seed=self.seed)
        self.value=None;self.learning_pending=None
        super().close()

    def archive_learning(self,report):
        if not hasattr(self,'saved_learning'):
            report['s1_training_eligible']=False
            return
        data=self.saved_learning
        eligible=(report['execution_valid'] and report['status'] in ('target_reached','time_limit','environment_done')
                  and bool(data['rows']) and self.split=='train')
        data.update(eligible=eligible,status=report['status'])
        self.torch.save(data,self.output/'s1-trajectory.pt')
        report.update(s1_training_eligible=eligible,s1_scope=SCOPE,s1_split=self.split,
            s1_checkpoint_sha256=self.checkpoint_sha256,s1_initial_heads_sha256=self.initial_heads_sha256,
            s1_trajectory_steps=len(data['rows']),s1_bootstrap=data['bootstrap'],
            weights_updated_during_game=False)
