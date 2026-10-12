"""One-use original STEVE-1/runtime parity probe on recorded development frames."""
import copy,json,sys,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import RLPolicy,guided,cpu_tree,state_fingerprint,extend_context_head
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json,jsonable
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];out=root/'results/base-policy-parity-20261008';out.mkdir(exist_ok=False)
stop=root/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
report=dict(completed=False,error=None,game_actions=0,optimizer_updates=0,sources=[],frames=[])
try:
    torch.set_num_threads(1);torch.manual_seed(9511)
    sys.path.insert(0,str(root/'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.actions import ActionTransformer
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS,set_default_torch_device
    from steve1.MineRLConditionalAgent import MineRLConditionalAgent
    from steve1.embed_conditioned_policy import MinecraftAgentPolicy
    set_default_torch_device('cuda');mapper=CameraHierarchicalMapping(n_camera_bins=11)
    policy=MinecraftAgentPolicy(DictType(**mapper.get_action_space_update()),*model_kwargs(root/'weights/steve1/2x.model'))
    weights=torch.load(root/'weights/steve1/steve1.weights',map_location='cpu');policy.load_state_dict(weights);del weights
    policy=policy.cuda().eval().requires_grad_(False);original_head=policy.pi_head
    local_head=copy.deepcopy(original_head);extend_context_head(local_head,policy.net.output_latent_size(),6)
    original_state=cpu_tree(local_head.state_dict())
    # Artifact contains the exact pretrained head plus zero RM columns, with no optimizer updates.
    value=torch.nn.Linear(policy.net.output_latent_size()+6,1)
    torch.nn.init.zeros_(value.weight);torch.nn.init.zeros_(value.bias)
    checkpoint=dict(schema=1,scope='head_only',actor=original_state,value=cpu_tree(value.state_dict()),
                    condition=dict(reward_mode='wood-potential-v4'),updates=0,
                    provenance=dict(kind='unaltered_pretrained_policy_with_zero_RM_columns',
                                    base_model_sha256=digest(root/'weights/steve1/steve1.weights')))
    torch.save(checkpoint,out/'original-head.pt')
    agent=MineRLConditionalAgent.__new__(MineRLConditionalAgent);agent.policy=policy;agent.device=torch.device('cuda')
    agent.action_mapper=mapper;agent.action_transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
    f=RLPolicy.__new__(RLPolicy);f.agent=agent;f.torch=torch;f.device='cuda';f.reward_mode='wood-potential-v4'
    embedding=root/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
    f.goal_embed=np.load(embedding,allow_pickle=False);f.set_reward_context([0,1,0,0,0,0])
    from minecraftrl.cuda_startup import prime_convolution
    report['backend_preparation']=prime_convolution(policy.net.img_process.cnn.stacks[0].firstconv.layer)
    cache={}
    policy.net.register_forward_hook(lambda module,args,output:cache.update(features=output[0][0].detach().clone()))
    for label,run in [('wall','rl-eval-2000003-20261008T043921103939Z'),('natural','rl-eval-2000029-20261008T045803907945Z')]:
        frames=root/'results'/run/'run/frames';official_state=policy.initial_state(2);local_state=policy.initial_state(2)
        report['sources'].append(dict(label=label,path=str(frames),frames=64))
        for i in range(64):
            if stop.exists():raise InterruptedError('STOP during parity probe')
            path=frames/f'{i:06d}.png'
            with Image.open(path) as image:screen=np.asarray(image.convert('RGB'))
            obs=agent._env_obs_to_agent({'pov':screen},f.goal_embed)
            policy.pi_head=original_head;torch.manual_seed(9511+i)
            # Same preprocessing/act/decoding as upstream get_action, exposing its distribution.
            with torch.no_grad(),torch.cuda.amp.autocast():
                action,official_state,info=policy.act(obs,torch.tensor([False],device='cuda'),official_state,
                                                     stochastic=True,return_pd=True,cond_scale=6.)
            official_features=cache['features'].float();official_env=agent._agent_action_to_env(action)
            reference={k:torch.log_softmax(v.unsqueeze(1),-1) for k,v in info['pd'].items()}
            agent.hidden_state=local_state;f.first=i==0;features,local_state=f.features(screen)
            policy.pi_head=local_head
            with torch.no_grad():
                pd=guided(local_head,features);torch.manual_seed(9511+i)
                sampled=local_head.sample(pd,deterministic=False)
                local_env=agent._agent_action_to_env({k:v[:,0] for k,v in sampled.items()})
                # Isolate FP32 head arithmetic from local feature/reset/context handling.
                no_context=guided(original_head,official_features)
                amp_head=None
                with torch.cuda.amp.autocast():amp_head=guided(original_head,official_features)
            def tv(a,b):return {k:float((a[k].exp()-b[k].exp()).abs().sum()/2) for k in a}
            row=dict(source=label,index=i,frame_sha256=digest(path),
                     feature_max_error=float((features[...,:-6]-official_features).abs().max()),
                     recurrent_identical=state_fingerprint(local_state)==state_fingerprint(official_state),
                     official_local_tv=tv(reference,pd),official_amp_tv=tv(reference,amp_head),
                     local_fp32_unextended_tv=tv(pd,no_context),same_sampled_env_action=jsonable(official_env)==jsonable(local_env))
            report['frames'].append(row)
        print('parity '+label,flush=True);write_json(out/'probe.json',report)
    report.update(completed=True,checkpoint_sha256=digest(out/'original-head.pt'),embedding_sha256=digest(embedding),
                  original_model_sha256=digest(root/'weights/steve1/steve1.weights'),
                  max_feature_error=max(r['feature_max_error'] for r in report['frames']),
                  recurrent_mismatches=sum(not r['recurrent_identical'] for r in report['frames']),
                  sampled_action_mismatches=sum(not r['same_sampled_env_action'] for r in report['frames']))
    for key in ('official_local_tv','official_amp_tv','local_fp32_unextended_tv'):
        report[key+'_max']={k:max(r[key][k] for r in report['frames']) for k in ('buttons','camera')}
except BaseException as exc:
    report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc())
finally:
    write_json(out/'probe.json',report);publish_completion(stop,'RL training ended; original-policy parity probe finished.')
print(json.dumps({k:v for k,v in report.items() if k!='frames'}));raise SystemExit(0 if report['completed'] else 2)
