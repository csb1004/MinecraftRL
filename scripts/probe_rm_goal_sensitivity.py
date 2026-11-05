"""Frozen, offline same-frame goal sensitivity; never launches Minecraft.

This checks numerical influence on action distributions, not task competence.
Run separately from gameplay to avoid concurrent GPU model loads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from minecraftrl.artifact_hash import sha256
from minecraftrl.policies import Steve1Policy
from minecraftrl.rm_conditioning import load_bank
from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import jsonable, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    if (root/'results/STOP').exists():
        raise RuntimeError('STOP latched; do not start another diagnostic')
    batch=root/'results/rm-conditioning-20261031'
    out=batch/'goal-sensitivity'
    out.mkdir(exist_ok=False)
    bank,metadata=load_bank(batch/'goal-bank')
    for name in ('2x.model','steve1.weights'):
        if sha256(root/'weights/steve1'/name)!=metadata['official_weights'][name]:
            raise ValueError('official policy hash changed')
    sys.path.insert(0,str(root/'external/steve1'))
    import gym
    import numpy as np
    import torch
    from PIL import Image
    from steve1.VPT.agent import ENV_KWARGS,TARGET_ACTION_SPACE
    # Only the published action-space contract is supplied. No environment,
    # Java process, reset, step, inventory reward or task success is simulated.
    spaces=SimpleNamespace(task=SimpleNamespace(**ENV_KWARGS),
                           action_space=gym.spaces.Dict(TARGET_ACTION_SPACE))
    policy=Steve1Policy(spaces,root,prompt_profile='official-visual',guidance=6.)
    policy.agent.policy.requires_grad_(False)
    from minecraftrl.cuda_startup import prime_convolution
    write_json(out/'backend-preparation.json',
               prime_convolution(policy.agent.policy.net.img_process.cnn.stacks[0].firstconv.layer))
    plan=json.loads((batch/'pilot-startup-r1-plan.json').read_text(encoding='utf-8-sig'))
    rows=[]
    try:
        for run in plan['runs']:
            if run['mode']!='fixed':
                continue
            path=batch/run['name']/'frames/000000.png'
            frame=np.array(Image.open(path).convert('RGB'))
            distributions={}
            actions={}
            for key in [*bank,'root_repeat']:
                if (root/'results/STOP').exists():
                    raise InterruptedError('STOP during offline diagnostic')
                goal='root' if key=='root_repeat' else key
                policy.agent.reset(cond_scale=6.)
                before_state=state_fingerprint(policy.agent.hidden_state)
                inputs=policy.agent._env_obs_to_agent({'pov':frame},bank[goal])
                with torch.no_grad(),torch.cuda.amp.autocast():
                    action,state,result=policy.agent.policy.act(
                        inputs,policy.agent._dummy_first,policy.agent.hidden_state,
                        stochastic=False,return_pd=True,cond_scale=6.)
                if state_fingerprint(policy.agent.hidden_state)!=before_state:
                    raise RuntimeError('Official forward mutated its input recurrent state')
                distributions[key]={head:torch.log_softmax(logits.float(),dim=-1).cpu()
                                    for head,logits in result['pd'].items()}
                actions[key]=jsonable(policy.agent._agent_action_to_env(action))
                del inputs,action,state,result
            comparisons={}
            for key in distributions:
                if key=='root':
                    continue
                differences={}
                for head,logp in distributions['root'].items():
                    logq=distributions[key][head]
                    p,q=logp.exp(),logq.exp()
                    differences[head]=dict(kl_root_to_goal=float((p*(logp-logq)).sum()),
                                           total_variation=float((p-q).abs().sum()/2))
                comparisons[key]=dict(heads=differences,
                                      deterministic_action_changed=actions[key]!=actions['root'])
            rows.append(dict(run=run['name'],frame_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             initial_recurrent_state=True,recurrent_input_preserved=True,
                             comparisons=comparisons,actions=actions))
        write_json(out/'report.json',dict(schema=1,rows=rows,optimizer_updates=0,game_steps=0,
                   guidance=6.,prompts=metadata['prompts'],official_weights=metadata['official_weights'],
                   interpretation='Offline numerical influence only; no competence, reward or learning claim.'))
        print('Recorded same-frame goal distributions for '+str(len(rows))+' held-fixed initial images; game steps 0.')
    finally:
        policy.close()


if __name__=='__main__':
    main()
