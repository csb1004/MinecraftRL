"""Same image and recurrent state, different registered goals. No game or sampling."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from minecraftrl.policies import model_kwargs
from minecraftrl.policy_tail import load_tail
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.rl_policy import RLPolicy,cpu_tree,guided,state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.temporal_bc import device_tree
from minecraftrl.world_store import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];run=args.run.resolve();out=args.output.resolve()
    if not all(p.is_relative_to(root/'results') for p in (run,out)):raise ValueError('project results required')
    out.mkdir(exist_ok=False);stop=root/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
    started=time.monotonic();read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    report=dict(completed=False,error=None,game_actions=0,api_calls=0,optimizer_updates=0,branches=[])
    def check():
        if stop.exists():raise InterruptedError('STOP during read-only condition probe')
    try:
        torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
        rows=read(run/'goal-recovery-rows.json');data=torch.load(run/'trajectory.pt',map_location='cpu')
        first=next(i for i,r in enumerate(rows) if r['subgoal_context']['phase']=='subgoal')
        checkpoint=root/'results/onset-coverage-training-20261009/checkpoint.pt'
        sources=[checkpoint,run/'trajectory.pt',run/'goal-recovery-rows.json',Path(__file__)]
        report['source_sha256']={p.relative_to(root).as_posix():digest(p) for p in sources}
        sys.path.insert(0,str(root/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_head import make_action_head
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.torch_util import set_default_torch_device
        from steve1.MineRLConditionalAgent import MineRLConditionalAgent
        from steve1.embed_conditioned_policy import MinecraftAgentPolicy
        set_default_torch_device('cpu');mapper=CameraHierarchicalMapping(n_camera_bins=11)
        space=DictType(**mapper.get_action_space_update());kwargs,head_kwargs=model_kwargs(root/'weights/steve1/2x.model')
        policy=MinecraftAgentPolicy(space,kwargs,head_kwargs)
        base=torch.load(root/'weights/steve1/steve1.weights',map_location='cpu');policy.load_state_dict(base,strict=True);del base
        saved=torch.load(checkpoint,map_location='cpu');load_tail(policy.net,saved)
        policy=policy.cuda().eval().requires_grad_(False)
        actor=make_action_head(space,data['rows'][0]['features'].shape[-1],**head_kwargs).cuda().eval()
        actor.load_state_dict(saved['actor']);actor.requires_grad_(False);del saved
        from minecraftrl.cuda_startup import prime_convolution
        prime_convolution(policy.net.img_process.cnn.stacks[0].firstconv.layer)
        agent=MineRLConditionalAgent.__new__(MineRLConditionalAgent);agent.policy=policy;agent.device=torch.device('cuda')
        f=RLPolicy.__new__(RLPolicy);f.agent,f.torch,f.device,f.reward_mode=agent,torch,'cuda','wood-potential-v4'
        embeds={'collect_wood':np.load(run/'goal-embed.npy',allow_pickle=False)}
        embeds.update({name:np.load(run/(name+'-embed.npy'),allow_pickle=False) for name in read(run/'neural-skills.json')})
        agent.hidden_state=device_tree(cpu_tree(data['initial_recurrent_state']),'cuda')
        def screen(i):
            with Image.open(run/rows[i]['frame']) as im:return np.asarray(im.convert('RGB'))
        f.goal_embed=embeds['collect_wood']
        for i in range(first):
            check();f.first=i==0;f.set_reward_context(rows[i]['reward_context'])
            _,state=f.features(screen(i));agent.hidden_state=state
        memory=cpu_tree(agent.hidden_state);memory_sha=state_fingerprint(memory)
        pixels=screen(first);distributions={};actual_skill=rows[first]['subgoal_context']['skill_id']
        f.first=False;f.set_reward_context(rows[first]['reward_context'])
        for name,embed in embeds.items():
            check();agent.hidden_state=device_tree(cpu_tree(memory),'cuda');assert state_fingerprint(agent.hidden_state)==memory_sha
            f.goal_embed=embed;features,_=f.features(pixels)
            with torch.no_grad():pd=cpu_tree(guided(actor,features))
            distributions[name]=pd
            row=dict(skill_id=name,embedding_sha256=hashlib.sha256(embed.tobytes()).hexdigest(),
                memory_before_sha256=memory_sha,feature_sha256=state_fingerprint(features),
                probabilities_sha256=state_fingerprint(pd))
            if name==actual_skill:
                error=(features.cpu()-data['rows'][first]['features']).abs().max().item()
                row['recorded_feature_max_error']=error
                assert error<1e-3,error
            report['branches'].append(row)
        original=distributions['collect_wood']
        for row in report['branches']:
            pd=distributions[row['skill_id']]
            row['kl_from_original']=sum((original[k].exp()*(original[k]-pd[k])).sum().item() for k in original)
            row['max_probability_delta']=max((original[k].exp()-pd[k].exp()).abs().max().item() for k in original)
        assert state_fingerprint(memory)==memory_sha
        for name,sha in report['source_sha256'].items():assert digest(root/name)==sha
        report.update(completed=True,first_subgoal_action=first+1,prefix_frames=first,
            frame_sha256=digest(run/rows[first]['frame']),actual_skill=actual_skill,
            conditions_change_distribution=all(r['max_probability_delta']>0 for r in report['branches'] if r['skill_id']!='collect_wood'),
            scope='One actual observation and recurrent memory; measures input influence, not task competence')
        torch.save(dict(memory_before=memory,distributions=distributions),out/'paired-conditions.pt')
    except BaseException as exc:
        report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        report.update(elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        write_json(out/'report.json',report)
        publish_completion(stop,'RL collection ended; subgoal condition probe finished.')
    print(json.dumps(report));return 0 if report['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
