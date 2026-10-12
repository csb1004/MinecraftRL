"""Disposable one-update feasibility probe; never emits a usable policy checkpoint."""
import copy,json,sys,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import RLPolicy,guided
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];out=root/'results/tail-finetuning-probe-20261008';out.mkdir(exist_ok=False)
stop=root/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
report=dict(completed=False,error=None,probe_optimizer_updates=0,game_actions=0,deployable_checkpoint=False)
try:
    torch.set_num_threads(1);torch.manual_seed(9401);torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0,str(root/'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.actions import ActionTransformer
    from steve1.VPT.lib.action_head import make_action_head
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS,set_default_torch_device
    from steve1.MineRLConditionalAgent import MineRLConditionalAgent
    from steve1.embed_conditioned_policy import MinecraftAgentPolicy
    set_default_torch_device('cuda');mapper=CameraHierarchicalMapping(n_camera_bins=11)
    kwargs,head_kwargs=model_kwargs(root/'weights/steve1/2x.model')
    p=MinecraftAgentPolicy(DictType(**mapper.get_action_space_update()),kwargs,head_kwargs)
    weights=torch.load(root/'weights/steve1/steve1.weights',map_location='cpu');p.load_state_dict(weights);del weights
    p=p.cuda().eval().requires_grad_(False)
    report['full_policy_parameters']=sum(v.numel() for v in p.parameters())
    report['full_fp32_adam_parameter_floor_mib']=report['full_policy_parameters']*16/2**20
    report['full_adam_floor_scope']='Arithmetic weights+gradients+two FP32 moments only; not measured full-backbone training or AMP/offload'
    agent=MineRLConditionalAgent.__new__(MineRLConditionalAgent);agent.policy=p;agent.device=torch.device('cuda');agent.hidden_state=p.initial_state(2)
    f=RLPolicy.__new__(RLPolicy);f.agent=agent;f.torch=torch;f.device='cuda';f.first=True;f.reward_mode='wood-potential-v4'
    f.goal_embed=np.load(root/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy',allow_pickle=False)
    run=root/'results/rl-train-104743-20261008T024538270180Z/run'
    row=json.loads((run/'teacher-rows.json').read_text())[0];f.set_reward_context(row['reward_context'])
    cache={}
    handle=p.net.lastlayer.register_forward_pre_hook(lambda module,args:cache.update(x=args[0].detach().clone()))
    from minecraftrl.cuda_startup import prime_convolution
    prime_convolution(p.net.img_process.cnn.stacks[0].firstconv.layer)
    with Image.open(run/row['frame']) as im:baseline,_=f.features(np.asarray(im.convert('RGB')))
    handle.remove()
    tail=torch.nn.Sequential(p.net.lastlayer,p.net.final_ln).requires_grad_(True)
    actor=make_action_head(DictType(**mapper.get_action_space_update()),baseline.shape[-1],**head_kwargs).cuda()
    parent=root/'results/recovery-gap-20261007/train-gap-epochs2/checkpoint.pt'
    actor.load_state_dict(torch.load(parent,map_location='cpu')['actor'])
    before={k:v.detach().clone() for k,v in tail.state_dict().items()}
    def features():
        with torch.cuda.amp.autocast():latent=tail(cache['x'])
        return torch.cat([latent.float(),baseline[...,-6:]],dim=-1)
    reconstructed=features();error=(reconstructed-baseline).abs().max().item()
    assert error<1e-5,'pre-tail replay differs from runtime'
    action={k:np.asarray(v) for k,v in row['action'].items()}
    factored=ActionTransformer(**ACTION_TRANSFORMER_KWARGS).env2policy(action)
    mapped=mapper.from_factored({k:v[None] for k,v in factored.items()})
    target={k:torch.from_numpy(v).unsqueeze(1).cuda() for k,v in mapped.items()}
    params=list(tail.parameters())+list(actor.parameters());optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0)
    if stop.exists():raise InterruptedError('STOP before disposable probe update')
    loss=-actor.logprob(target,guided(actor,reconstructed)).mean();loss.backward()
    grad=sum(v.grad.abs().sum().item() for v in tail.parameters() if v.grad is not None)
    assert grad>0 and all(torch.isfinite(v.grad).all() for v in params if v.grad is not None)
    assert all(v.grad is None for v in p.net.recurrent_layer.parameters())
    assert all(v.grad is None for v in p.net.img_process.parameters())
    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    if stop.exists():raise InterruptedError('STOP before disposable optimizer step')
    optimizer.step();report['probe_optimizer_updates']=1
    delta=max((v-before[k]).abs().max().item() for k,v in tail.state_dict().items());assert delta>0
    report.update(completed=True,replay_max_error=error,tail_gradient_l1=grad,tail_max_weight_change=delta,
                  tail_parameters=sum(v.numel() for v in tail.parameters()),head_parameters=sum(v.numel() for v in actor.parameters()),
                  vision_recurrent_frozen=True,parent_sha256=digest(parent),training_source=str(run),
                  scope='One real training frame, detached recurrent features, original pretrained final layer and action head; no gameplay validation')
except BaseException as exc:
    report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc())
finally:
    report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
    write_json(out/'probe.json',report);publish_completion(stop,'RL training ended; disposable tail feasibility probe finished.')
print(json.dumps(report));raise SystemExit(0 if report['completed'] else 2)
