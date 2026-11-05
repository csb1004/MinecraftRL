"""Bounded offline PPO on newly collected S1 trajectories; no game or API.

Every rollout must have a clean supervisor verdict and identical starting heads.
The frozen backbone is never loaded by this trainer. Output is a candidate for
separate evaluation, not an automatically promoted policy.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from minecraftrl.policies import model_kwargs
from minecraftrl.rl_core import gae
from minecraftrl.rl_opt import ppo_loss
from minecraftrl.rl_policy import cpu_tree, guided_minibatch, state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.s1_learning import validate_data, SCOPE, EXTRA_DIM


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-file',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();out=args.output.resolve()
    if not out.is_relative_to(root/'results') or out.exists():raise ValueError('new project results output required')
    if args.stop_file.exists():raise InterruptedError('STOP before training')
    import torch
    sys.path.insert(0,str(root/'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.action_head import make_action_head
    config=dict(gamma=.999,gae_lambda=1.,actor_lr=1e-5,value_lr=1e-3,epochs=1,
                minibatch=32,ppo_clip=.1,reference_kl_coefficient=.1,target_kl=.02,
                max_grad_norm=1.,value_coefficient=.5,seed=103131,device='cuda')
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.reset_peak_memory_stats()
    runs=[];sources=[]
    for path in args.runs:
        path=path.resolve()
        if not path.is_relative_to(root/'results'):raise ValueError('project-local training runs required')
        data=torch.load(path/'s1-trajectory.pt',map_location='cpu')
        summary=json.loads((path/'summary.json').read_text())
        validate_data(data,summary)
        runs.append(data)
        sources.append(dict(run=str(path.relative_to(root)),trajectory_sha256=hashlib.sha256((path/'s1-trajectory.pt').read_bytes()).hexdigest(),
                            summary_sha256=hashlib.sha256((path/'summary.json').read_bytes()).hexdigest()))
    first=runs[0]
    if len({r['sampling_seed'] for r in runs})<2:raise ValueError('multiple training seeds required')
    for r in runs:
        for key in ('scope','initial_heads_sha256','feature_width','base_model_sha256','context_dim','guidance'):
            if r[key]!=first[key]:raise ValueError('incompatible training batch '+key)
    device=config['device']
    _,head_kwargs=model_kwargs(root/'weights/steve1/2x.model')
    space=DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())
    actor=make_action_head(space,first['feature_width'],**head_kwargs).to(device)
    actor.load_state_dict(first['initial_heads']['actor'],strict=True)
    value=torch.nn.Linear(first['feature_width'],1).to(device)
    value.load_state_dict(first['initial_heads']['value'],strict=True)
    reference=copy.deepcopy(actor).eval().requires_grad_(False)
    rows=[row for run in runs for row in run['rows']]
    for row in rows:
        f=row['features']
        if f.shape!=(2,1,first['feature_width']) or f.dtype!=torch.float32 or not torch.isfinite(f).all():
            raise ValueError('invalid frozen features')
        expected=torch.tensor(row['phase_context']).view(1,1,EXTRA_DIM).expand(2,1,EXTRA_DIM)
        if not torch.equal(f[:,:,-EXTRA_DIM:],expected):raise ValueError('phase context not actually connected')
    features=torch.cat([row['features'] for row in rows],1).to(device)
    actions={k:torch.cat([row['action'][k] for row in rows],1).to(device) for k in rows[0]['action']}
    old=torch.tensor([row['old_logp'] for row in rows],device=device)
    adv=[];returns=[]
    for run in runs:
        a,r=gae([x['reward'] for x in run['rows']],[x['value'] for x in run['rows']],run['bootstrap'],
                terminal=run['terminal'],gamma=config['gamma'],lam=config['gae_lambda'])
        adv.extend(a);returns.extend(r)
    advantage=torch.tensor(adv,device=device);target=torch.tensor(returns,device=device)
    if not torch.isfinite(advantage).all() or advantage.std(unbiased=False)<1e-8:
        raise ValueError('no usable advantage variation; do not claim a policy update')
    advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    replay_errors=[]
    with torch.no_grad():
        for start in range(0,len(rows),config['minibatch']):
            stop=min(start+config['minibatch'],len(rows));f=features[:,start:stop].contiguous()
            logp=actor.logprob({k:v[:,start:stop] for k,v in actions.items()},guided_minibatch(actor,f)).flatten()
            replay_errors.append(float((logp-old[start:stop]).abs().max()))
    replay_error=max(replay_errors)
    if replay_error>1e-4:raise ValueError('collection probability not reproduced: '+str(replay_error))
    out.mkdir(parents=True,exist_ok=False)
    optimizer=torch.optim.Adam([dict(params=actor.parameters(),lr=config['actor_lr']),
                                dict(params=value.parameters(),lr=config['value_lr'])])
    history=[];started=time.monotonic();early_stop=False
    for epoch in range(config['epochs']):
        order=torch.randperm(len(rows),device=device)
        for start in range(0,len(rows),config['minibatch']):
            if args.stop_file.exists():raise InterruptedError('STOP during offline PPO')
            indices=order[start:start+config['minibatch']]
            f=features[:,indices].contiguous();act={k:v[:,indices] for k,v in actions.items()}
            pd=guided_minibatch(actor,f)
            logp=actor.logprob(act,pd).flatten()
            logratio=logp-old[indices]
            approximate_kl=float(((logratio.exp()-1)-logratio).mean().detach())
            if approximate_kl>config['target_kl']:
                early_stop=True;break
            with torch.no_grad():refpd=guided_minibatch(reference,f)
            kl=sum((refpd[k].exp()*(refpd[k]-pd[k])).sum(-1).mean() for k in pd)
            actor_loss=ppo_loss(logp,old[indices],advantage[indices],config['ppo_clip'])
            value_loss=(value(f[0:1]).flatten()-target[indices]).square().mean()
            loss=actor_loss+config['reference_kl_coefficient']*kl+config['value_coefficient']*value_loss
            if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
            optimizer.zero_grad();loss.backward()
            context_grad=sum(float(m.weight.grad[:,-EXTRA_DIM:].square().sum())
                for m in actor.modules() if isinstance(m,torch.nn.Linear) and m.weight.grad is not None)**.5
            parameters=[*actor.parameters(),*value.parameters()]
            grad_norm=torch.nn.utils.clip_grad_norm_(parameters,config['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(epoch=epoch,minibatch=start,actor_loss=float(actor_loss.detach()),
                value_loss=float(value_loss.detach()),reference_kl=float(kl.detach()),
                approximate_kl=approximate_kl,gradient_norm=float(grad_norm),actor_context_gradient_norm=context_grad))
        if early_stop:break
    heads=cpu_tree(dict(actor=actor.state_dict(),value=value.state_dict()))
    if not history or state_fingerprint(heads['actor'])==state_fingerprint(first['initial_heads']['actor']):
        raise ValueError('no actor update occurred')
    if any(not torch.isfinite(v).all() for h in heads.values() for v in h.values()):
        raise ValueError('nonfinite trained weights')
    if args.stop_file.exists():raise InterruptedError('STOP before saving candidate')
    actor_context_delta=sum(float((v[:,-EXTRA_DIM:]-first['initial_heads']['actor'][k][:,-EXTRA_DIM:]).square().sum())
        for k,v in heads['actor'].items() if v.ndim==2 and v.shape[1]==first['feature_width'])**.5
    value_context_delta=float((heads['value']['weight'][:,-EXTRA_DIM:]-first['initial_heads']['value']['weight'][:,-EXTRA_DIM:]).norm())
    checkpoint=dict(scope=SCOPE,base_model_sha256=first['base_model_sha256'],context_dim=EXTRA_DIM,
        guidance=6.,heads=heads,config=config,sources=sources,parent_heads_sha256=first['initial_heads_sha256'],
        optimizer_updates=len(history),promoted=False)
    torch.save(checkpoint,out/'checkpoint.pt')
    write_json(out/'training.json',dict(config=config,sources=sources,rows=len(rows),
        seeds=[r['sampling_seed'] for r in runs],rewards=[sum(x['reward'] for x in r['rows']) for r in runs],
        replay_logp_max_error=replay_error,history=history,early_stop=early_stop,
        actor_changed=True,actor_context_delta_norm=actor_context_delta,value_context_delta_norm=value_context_delta,
        initial_heads_sha256=first['initial_heads_sha256'],heads_sha256=state_fingerprint(heads),
        elapsed_s=time.monotonic()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
        checkpoint_sha256=hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest(),
        evaluated=False,promoted=False,scope=SCOPE))
    print(json.dumps(dict(rows=len(rows),updates=len(history),replay_error=replay_error,promoted=False)),flush=True)


if __name__=='__main__':main()
