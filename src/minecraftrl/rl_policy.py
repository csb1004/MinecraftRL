"""STEVE-1 head RL and offline tail evaluation with normalized guided probabilities."""
import copy
import gc
import hashlib
from .policies import Steve1Policy
from .runtime import jsonable,write_json

def cpu_tree(value):
    if hasattr(value,'detach'):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(cpu_tree(v) for v in value)
    return value

def state_fingerprint(value):
    digest=hashlib.sha256()
    def visit(item):
        if hasattr(item,'detach'):
            array=item.detach().cpu().contiguous().numpy()
            digest.update(str((array.dtype,array.shape)).encode());digest.update(array.tobytes())
        elif isinstance(item,dict):
            for key in sorted(item):digest.update(str(key).encode());visit(item[key])
        elif isinstance(item,(tuple,list)):
            digest.update(str(len(item)).encode())
            for child in item:visit(child)
        else:digest.update(repr(item).encode())
    visit(value);return digest.hexdigest()

def guided(head,features):
    import torch
    logits=head(features)
    return {k:torch.log_softmax(7*v[0:1]-6*v[1:2],dim=-1) for k,v in logits.items()}

def guided_minibatch(head,features):
    """Keep each forward GEMM shape identical to collection; retain gradients."""
    import torch
    parts=[guided(head,features[:,i:i+1].contiguous()) for i in range(features.shape[1])]
    if not parts:raise ValueError('empty policy minibatch')
    return {k:torch.cat([part[k] for part in parts],dim=1) for k in parts[0]}


def extend_context_head(head,base_dim,extra_dim):
    """Zero-init extra columns: reuse parent heads without inventing stage skill."""
    import torch
    found=0
    for module in head.modules():
        if isinstance(module,torch.nn.Linear):
            if module.in_features!=base_dim:raise ValueError('unexpected head architecture')
            module.weight=torch.nn.Parameter(torch.cat([module.weight.detach(),
                module.weight.new_zeros((module.out_features,extra_dim))],dim=1))
            module.in_features+=extra_dim;found+=1
    if not found:raise ValueError('head has no linear input')

class RLPolicy(Steve1Policy):
    def __init__(self,env,root,output,checkpoint=None,embed_path=None,seed=0,prewarm=True,reward_mode='sparse'):
        super().__init__(env,root);self.output=output;self.embed_path=embed_path;self.rows=[];self.pending=None;self.seed=seed
        from .rl_core import SHAPED_REWARDS
        if reward_mode!='sparse' and reward_mode not in SHAPED_REWARDS:raise ValueError('unknown reward mode')
        self.reward_mode=reward_mode;self.reward_context=None
        torch=self.torch;p=self.agent.policy
        p.requires_grad_(False)
        self.value=torch.nn.Linear(p.net.output_latent_size(),1).to(self.device)
        torch.nn.init.zeros_(self.value.weight);torch.nn.init.zeros_(self.value.bias)
        saved=None
        if checkpoint:
            # Read heads/optional tail on CPU; never stage optimizer states on GPU.
            saved=torch.load(checkpoint,map_location='cpu')
        from .policy_tail import load_tail,tail_state,temporal_state,TEMPORAL_SCOPE
        self.has_trained_tail=False
        self.has_trained_temporal=bool(saved and saved.get('scope')==TEMPORAL_SCOPE)
        if saved and saved.get('scope') in ('head_and_tail',TEMPORAL_SCOPE):
            from .world_store import digest
            if saved.get('base_model_sha256')!=digest(root/'weights/steve1/steve1.weights'):
                raise ValueError('tail checkpoint base model mismatch')
        if saved:self.has_trained_tail=load_tail(p.net,saved)
        saved_mode=(saved or {}).get('condition',{}).get('reward_mode','sparse')
        compatible=(saved_mode==reward_mode or (saved_mode=='wood-potential-v1' and reward_mode=='wood-potential-v2')
                    or (saved_mode in ('wood-potential-v1','wood-potential-v2') and reward_mode=='wood-potential-v3')
                    or (saved_mode in ('wood-potential-v1','wood-potential-v2','wood-potential-v3') and reward_mode=='wood-potential-v4'))
        if saved_mode!='sparse' and not compatible:raise ValueError('checkpoint requires matching RM context')
        self.source_reward_mode=saved_mode
        if saved is not None and saved_mode=='sparse':
            p.pi_head.load_state_dict(saved['actor']);self.value.load_state_dict(saved['value'])
        if reward_mode!='sparse':
            extend_context_head(p.pi_head,p.net.output_latent_size(),6)
            extend_context_head(self.value,p.net.output_latent_size(),6)
        if saved is not None and saved_mode!='sparse':
            p.pi_head.load_state_dict(saved['actor']);self.value.load_state_dict(saved['value'])
        self.checkpoint=str(checkpoint) if checkpoint else None
        self.initial_heads=dict(actor=cpu_tree(p.pi_head.state_dict()),value=cpu_tree(self.value.state_dict()))
        if self.has_trained_tail:self.initial_heads['backbone_tail']=tail_state(p.net)
        if self.has_trained_temporal:self.initial_heads['backbone_temporal']=temporal_state(p.net)
        self.backend_preparation=dict(completed=False,enabled=prewarm)
        if prewarm:
            from .cuda_startup import prime_convolution
            write_json(output/'backend-preparation-started.json',dict(phase='policy_initialization'))
            self.backend_preparation.update(prime_convolution(p.net.img_process.cnn.stacks[0].firstconv.layer))
        write_json(output/'backend-preparation.json',self.backend_preparation)
    def reset(self,goal=None):
        import numpy as np
        if self.embed_path:
            self.agent.reset(cond_scale=6.);self.goal_embed=np.load(self.embed_path,allow_pickle=False)
        else:super().reset(goal)
        np.save(self.output/'goal-embed.npy',self.goal_embed)
        self.mineclip=self.prior=None;gc.collect();self.torch.cuda.empty_cache()
        self.first=True;self.initial_state=cpu_tree(self.agent.hidden_state)
        self.torch.manual_seed(self.seed);self.torch.cuda.manual_seed_all(self.seed)
        self.condition_metadata=dict(controller='steve1_head_rl',prompt=goal,cond_scale=6.,
            embedding_sha256=hashlib.sha256(self.goal_embed.tobytes()).hexdigest(),checkpoint=self.checkpoint,
            initial_recurrent_sha256=state_fingerprint(self.initial_state),sampling_seed=self.seed,
            recurrent_commit_mode='dispatch_commit',
            trainable_scope='action_head_and_new_wood_value_head; frozen vision/recurrent backbone')
        self.condition_metadata['backend_preparation']=getattr(self,'backend_preparation',None)
        self.condition_metadata['reward_mode']=self.reward_mode
        if self.has_trained_tail:
            self.condition_metadata['backbone_tail_sha256']=state_fingerprint(self.initial_heads['backbone_tail'])
            self.condition_metadata['trainable_scope']='offline-finetuned action head and pretrained final layers; evaluation only; vision/recurrent frozen'
        if self.has_trained_temporal:
            self.condition_metadata['backbone_temporal_sha256']=state_fingerprint(self.initial_heads['backbone_temporal'])
            self.condition_metadata['trainable_scope']='offline BC last recurrent block, final layers and action head; evaluation only; prefix frozen'
        if self.reward_mode!='sparse':
            from .rl_core import SHAPED_REWARDS
            self.condition_metadata['reward_config']=dict(SHAPED_REWARDS[self.reward_mode].metadata)
            self.condition_metadata['source_checkpoint_reward_mode']=self.source_reward_mode
        write_json(self.output/'rl-condition.json',self.condition_metadata)
    def features(self,screen):
        torch=self.torch;agent=self.agent
        obs=agent._env_obs_to_agent({'pov':screen},self.goal_embed)
        obs={k:torch.cat([v,v],0).unsqueeze(1) for k,v in obs.items()};obs['mineclip_embed'][1]=0
        first=torch.full((2,1),self.first,dtype=torch.bool,device=self.device)
        with torch.no_grad(),torch.cuda.amp.autocast():
            (pi,v),state=agent.policy.net(obs,agent.hidden_state,context={'first':first})
        pi=pi.float()
        if self.reward_mode!='sparse':
            if self.reward_context is None:raise ValueError('RM context missing before inference')
            context=torch.tensor(self.reward_context,device=self.device,dtype=pi.dtype).view(1,1,6).expand(2,1,6)
            pi=torch.cat([pi,context],dim=-1)
        return pi,state
    def set_reward_context(self,context):
        import math
        if len(context)!=6 or not all(math.isfinite(x) for x in context):raise ValueError('invalid RM context')
        self.reward_context=list(context)
    def act(self,screen):
        torch=self.torch
        if self.first:
            import faulthandler,time
            with (self.output/'first-inference-stack.txt').open('w') as trace:
                faulthandler.dump_traceback_later(20,file=trace)
                started=time.monotonic()
                try:features,state=self.features(screen)
                finally:faulthandler.cancel_dump_traceback_later()
            write_json(self.output/'first-feature-timing.json',dict(seconds=time.monotonic()-started))
        else:features,state=self.features(screen)
        with torch.no_grad():
            pd=guided(self.agent.policy.pi_head,features)
            if any(not torch.isfinite(v).all() for v in pd.values()):raise ValueError('nonfinite action distribution')
            action=self.agent.policy.pi_head.sample(pd,deterministic=False)
            logp=self.agent.policy.pi_head.logprob(action,pd).item()
            value=self.value(features[0:1]).item()
            import math
            if not math.isfinite(logp) or not math.isfinite(value):raise ValueError('nonfinite policy output')
        env_action=self.agent._agent_action_to_env({k:v[:,0] for k,v in action.items()})
        self.pending=dict(features=features.cpu(),action=cpu_tree(action),old_logp=logp,value=value,
                          first=self.first,env_action=jsonable(env_action))
        if self.reward_mode!='sparse':self.pending['reward_context']=list(self.reward_context)
        # Inference may be discarded at the deadline before env.step. Commit
        # recurrent state only with the action actually sent to the game.
        self.pending_state=state
        return env_action
    def commit(self,action,reward,terminal):
        if self.pending is None or jsonable(action)!=self.pending['env_action']:raise ValueError('RL action/dispatch mismatch')
        self.rows.append(dict(self.pending,reward=reward,terminal=terminal));self.pending=None
        self.agent.hidden_state=self.pending_state;self.pending_state=None;self.first=False
    def bootstrap(self,screen):
        with self.torch.no_grad():
            features,_=self.features(screen);return self.value(features[0:1]).item()
    def close(self):
        # run_rl writes the trajectory before the generic runner closes this object.
        pass
    def dispose(self):
        self.pending_state=None;self.value=None;super().close()
