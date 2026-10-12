"""Disposable temporal-gradient/VRAM probe on eight training frames, after gameplay."""
import gc,json,sys,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import RLPolicy,guided,cpu_tree
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];out=root/'results/temporal-finetuning-probe-20261008'
batch=json.loads((root/'results/base-policy-comparison-20261008/experiment.json').read_text())
assert batch['completed'] and batch['active'] is None
out.mkdir(exist_ok=False);stop=root/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
report=dict(completed=False,error=None,probe_optimizer_updates=0,game_actions=0,deployable_checkpoint=False,sequence_length=8)
try:
    torch.set_num_threads(1);torch.manual_seed(9521);torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0,str(root/'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.actions import ActionTransformer
    from steve1.VPT.lib.action_head import make_action_head
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS,set_default_torch_device
    from steve1.MineRLConditionalAgent import MineRLConditionalAgent
    from steve1.embed_conditioned_policy import MinecraftAgentPolicy
    set_default_torch_device('cuda');mapper=CameraHierarchicalMapping(n_camera_bins=11)
    space=DictType(**mapper.get_action_space_update());kwargs,head_kwargs=model_kwargs(root/'weights/steve1/2x.model')
    p=MinecraftAgentPolicy(space,kwargs,head_kwargs)
    weights=torch.load(root/'weights/steve1/steve1.weights',map_location='cpu');p.load_state_dict(weights);del weights
    p=p.cuda().eval().requires_grad_(False)
    block=p.net.recurrent_layer.blocks[-1];tail=torch.nn.Sequential(p.net.lastlayer,p.net.final_ln)
    initial=cpu_tree(p.initial_state(2)[-1]);cache={};inputs=[];baselines=[];contexts=[]
    handle=block.register_forward_pre_hook(lambda module,args:cache.update(x=args[0].detach().cpu().clone()))
    agent=MineRLConditionalAgent.__new__(MineRLConditionalAgent);agent.policy=p;agent.device=torch.device('cuda');agent.hidden_state=p.initial_state(2)
    f=RLPolicy.__new__(RLPolicy);f.agent=agent;f.torch=torch;f.device='cuda';f.reward_mode='wood-potential-v4'
    f.goal_embed=np.load(root/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy',allow_pickle=False)
    run=root/'results/rl-train-104743-20261008T035427136726Z/run'
    rows=json.loads((run/'teacher-rows.json').read_text())[:8]
    plan=json.loads((root/'configs/diverse-teacher-bc-20261008.json').read_text())
    source=next(s for s in plan['sessions'] if s['run']==run.relative_to(root).as_posix())
    assert source['split']=='train' and digest(run/'teacher-rows.json')==source['rows_sha256']
    from minecraftrl.cuda_startup import prime_convolution
    prime_convolution(p.net.img_process.cnn.stacks[0].firstconv.layer)
    for i,row in enumerate(rows):
        if stop.exists():raise InterruptedError('STOP in temporal input extraction')
        assert digest(run/row['frame'])==source['frame_sha256'][i]
        f.first=i==0;f.set_reward_context(row['reward_context'])
        with Image.open(run/row['frame']) as im:features,state=f.features(np.asarray(im.convert('RGB')))
        inputs.append(cache['x']);baselines.append(features.cpu());contexts.append(features[...,-6:].cpu())
        agent.hidden_state=state
    handle.remove();agent.hidden_state=None;agent.policy=None
    p.cpu();del p,f,agent,features,state;cache.clear();gc.collect();torch.cuda.empty_cache()
    block=block.cuda().eval().requires_grad_(True);tail=tail.cuda().eval().requires_grad_(True)
    actor=make_action_head(space,baselines[0].shape[-1],**head_kwargs).cuda()
    parent=root/'results/recovery-gap-20261007/train-gap-epochs2/checkpoint.pt'
    actor.load_state_dict(torch.load(parent,map_location='cpu')['actor'])
    def device_tree(value):
        if isinstance(value,torch.Tensor):return value.cuda()
        if isinstance(value,(tuple,list)):return type(value)(device_tree(v) for v in value)
        if isinstance(value,dict):return {k:device_tree(v) for k,v in value.items()}
        return value
    def forward(sequence):
        memory=device_tree(initial);outputs=[]
        # Keep runtime GEMM shapes and propagate gradients across carried memory.
        for i in range(8):
            with torch.cuda.amp.autocast():
                latent,memory=block(sequence[:,i:i+1],torch.full((2,1),i==0,dtype=torch.bool,device='cuda'),memory)
                latent=tail(torch.relu(latent))
            outputs.append(torch.cat([latent.float(),contexts[i].cuda()],-1))
        return outputs
    x=torch.cat(inputs,1).cuda().detach().requires_grad_(True)
    with torch.no_grad():
        replay=forward(x)
        report['replay_max_error']=max(float((v-baselines[i].cuda()).abs().max()) for i,v in enumerate(replay))
        assert report['replay_max_error']<1e-5
        altered=x.detach().clone();altered[:,-1]+=1
        changed=forward(altered)
        report['future_to_past_max_error']=max(float((replay[i]-changed[i]).abs().max()) for i in range(7))
        assert report['future_to_past_max_error']==0
        del replay,changed,altered
    action={k:np.asarray(v) for k,v in rows[-1]['action'].items()}
    factored=ActionTransformer(**ACTION_TRANSFORMER_KWARGS).env2policy(action)
    mapped=mapper.from_factored({k:v[None] for k,v in factored.items()})
    target={k:torch.from_numpy(v).unsqueeze(1).cuda() for k,v in mapped.items()}
    params=list(block.parameters())+list(tail.parameters())+list(actor.parameters())
    before=cpu_tree(block.state_dict());optimizer=torch.optim.AdamW(params,lr=1e-6,weight_decay=0)
    if stop.exists():raise InterruptedError('STOP before temporal backward')
    outputs=forward(x);loss=-actor.logprob(target,guided(actor,outputs[-1])).mean()
    assert torch.isfinite(loss);loss.backward()
    report['earlier_frame_gradient_l1']=float(x.grad[:,:-1].abs().sum())
    report['temporal_gradient_l1']=sum(float(v.grad.abs().sum()) for v in block.parameters() if v.grad is not None)
    assert report['earlier_frame_gradient_l1']>0 and report['temporal_gradient_l1']>0
    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    if stop.exists():raise InterruptedError('STOP before disposable temporal optimizer step')
    optimizer.step();report['probe_optimizer_updates']=1
    report['temporal_max_weight_change']=max(float((v.detach().cpu()-before[k]).abs().max()) for k,v in block.state_dict().items())
    assert report['temporal_max_weight_change']>0
    report.update(completed=True,temporal_parameters=sum(v.numel() for v in block.parameters()),
                  total_trainable_parameters=sum(v.numel() for v in params),source=str(run),parent_sha256=digest(parent),
                  scope='Last recurrent block plus final layers/head; frozen prefix cached; loss on final frame only; no deployable checkpoint')
except BaseException as exc:
    report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc())
finally:
    report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
    write_json(out/'probe.json',report);publish_completion(stop,'RL training ended; disposable temporal gradient probe finished.')
print(json.dumps(report));raise SystemExit(0 if report['completed'] else 2)
