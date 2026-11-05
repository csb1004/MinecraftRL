"""Matched-budget S2 2x2 PPO from verified, identical initial behavior."""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
import torch
from minecraftrl.conditioned_head import ConditionedHead, ConditionedValue, VARIANTS
from minecraftrl.semantic_goal import CONTEXT_DIM, CONTEXT_FIELDS
from minecraftrl.s2_learning import SCOPE
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_core import gae
from minecraftrl.rl_opt import ppo_loss
from minecraftrl.rl_policy import cpu_tree, guided_minibatch, state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.goal_spec import validate_trace


def validate(data, summary):
    if (data.get('scope')!=SCOPE or data.get('split')!='train' or data.get('eligible') is not True
            or data.get('variant')!='linear-on' or data.get('checkpoint_sha256') is not None
            or summary.get('execution_valid') is not True or summary.get('worker_exit_code')!=0
            or data.get('status')!=summary.get('status')
            or data.get('status') not in ('target_reached','time_limit','environment_done')):
        raise ValueError('only clean new initial-policy training runs may enter shared PPO')
    rows=data['rows']
    if not rows or len(rows)!=summary.get('steps') or data.get('recurrent_commit')!='dispatch_only':
        raise ValueError('invalid committed transitions')
    if data.get('context_fields')!=list(CONTEXT_FIELDS) or data['context_dim']!=CONTEXT_DIM:
        raise ValueError('incompatible semantic context')
    if state_fingerprint(data['initial_heads'])!=data['initial_heads_sha256']:
        raise ValueError('head fingerprint mismatch')
    for i,row in enumerate(rows):
        if row['step']!=i+1 or row['terminal'] and i!=len(rows)-1:
            raise ValueError('invalid terminal/step identity')
        if row['value']!=0 or not all(math.isfinite(row[k]) for k in ('reward','value','old_logp')):
            raise ValueError('initial zero critic/finite probabilities required')
        features=row['features']
        expected=torch.tensor(row['phase_context']).view(1,1,CONTEXT_DIM).expand(2,1,CONTEXT_DIM)
        if (features.shape!=(2,1,data['feature_width']) or features.dtype!=torch.float32
                or not torch.isfinite(features).all() or not torch.equal(features[:,:,-CONTEXT_DIM:],expected)):
            raise ValueError('context not actually connected to features')
    if bool(rows[-1]['terminal'])!=data['terminal'] or data['bootstrap']!=0.:
        raise ValueError('initial bootstrap/terminal mismatch')
    if not data['terminal'] and data.get('bootstrap_features') is None:
        raise ValueError('truncation final state missing')
    validate_trace(data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-file',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();out=args.output.resolve()
    if not out.is_relative_to(root/'results') or out.exists(): raise ValueError('fresh project-local output required')
    if args.stop_file.exists(): raise InterruptedError('STOP before trainer')
    torch.set_num_threads(2)
    sys.path.insert(0,str(root/'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.action_head import make_action_head
    config=dict(gamma=.999,gae_lambda=1.,actor_lr=.0001,value_lr=.001,epochs=2,minibatch=32,
                ppo_clip=.1,reference_kl_coefficient=.1,target_kl=.02,max_grad_norm=1.,
                seed=110101,device='cuda',gradient_clipping='separate_actor_and_value')
    runs=[];sources=[]
    for path in args.runs:
        path=path.resolve()
        if not path.is_relative_to(root/'results'): raise ValueError('external training data rejected')
        data=torch.load(path/'s2-trajectory.pt',map_location='cpu')
        summary=json.loads((path/'summary.json').read_text())
        validate(data,summary);runs.append(data)
        sources.append(dict(run=str(path.relative_to(root)),
            trajectory_sha256=hashlib.sha256((path/'s2-trajectory.pt').read_bytes()).hexdigest(),
            summary_sha256=hashlib.sha256((path/'summary.json').read_bytes()).hexdigest()))
    if len({r['sampling_seed'] for r in runs})!=len(runs) or len(runs)<2:
        raise ValueError('distinct new training seeds required')
    first=runs[0]
    for r in runs:
        for key in ('initial_heads_sha256','feature_width','base_model_sha256','guidance','context_fields'):
            if r[key]!=first[key]: raise ValueError('incompatible behavior batch '+key)
    rows=[row for run in runs for row in run['rows']]
    features=torch.cat([r['features'] for r in rows],1).cuda()
    actions={k:torch.cat([r['action'][k] for r in rows],1).cuda() for k in rows[0]['action']}
    old=torch.tensor([r['old_logp'] for r in rows],device='cuda')
    raw_adv=[];returns=[]
    for r in runs:
        a,t=gae([x['reward'] for x in r['rows']],[x['value'] for x in r['rows']],r['bootstrap'],
                terminal=r['terminal'],gamma=config['gamma'],lam=config['gae_lambda'])
        raw_adv+=a;returns+=t
    advantage=torch.tensor(raw_adv,device='cuda');target=torch.tensor(returns,device='cuda')
    if advantage.std(unbiased=False)<1e-8: raise ValueError('no usable advantage variation')
    advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    _,head_kwargs=model_kwargs(root/'weights/steve1/2x.model')
    space=DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())
    width=first['feature_width']-CONTEXT_DIM
    initial_base={k[len('base.'):]:v for k,v in first['initial_heads']['actor'].items() if k.startswith('base.')}
    out.mkdir(parents=True)
    write_json(out/'data-audit.json',dict(sources=sources,rows=len(rows),
        phases=dict(Counter(r['phase'] for r in rows)),
        terminal_failures=sum(r['terminal'] and not any(x['reward'] for x in r['rows']) for r in runs),
        quantity_values=sorted(set(r['phase_context'][7] for r in rows)),
        progress_values=sorted(set(r['phase_context'][5] for r in rows)),
        rewards=[sum(x['reward'] for x in r['rows']) for r in runs],
        shared_data_justification='identical behavior only after per-variant probability reproduction',
        additional_game_runs=0))
    summaries={}
    for variant in VARIANTS:
        if args.stop_file.exists(): raise InterruptedError('STOP before next candidate')
        actor=ConditionedHead(make_action_head(space,width,**head_kwargs),width,variant).cuda()
        actor.base.load_state_dict(initial_base,strict=True)
        value=ConditionedValue(width,variant).cuda()
        reference=copy.deepcopy(actor).eval().requires_grad_(False)
        initial=cpu_tree(actor.state_dict())
        max_error=0.
        with torch.no_grad():
            for start in range(0,len(rows),32):
                f=features[:,start:start+32].contiguous()
                pd=guided_minibatch(actor,f)
                logp=actor.logprob({k:v[:,start:start+32] for k,v in actions.items()},pd).flatten()
                max_error=max(max_error,float((logp-old[start:start+32]).abs().max()))
        if max_error>1e-4: raise ValueError('shared behavior probability gate failed: '+variant+' '+str(max_error))
        optimizer=torch.optim.Adam([dict(params=actor.parameters(),lr=config['actor_lr']),
                                    dict(params=value.parameters(),lr=config['value_lr'])])
        torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
        torch.cuda.reset_peak_memory_stats();started=time.monotonic();history=[];early_stop=False
        for epoch in range(config['epochs']):
            order=torch.randperm(len(rows),device='cuda')
            for start in range(0,len(rows),32):
                if args.stop_file.exists(): raise InterruptedError('STOP during update')
                indices=order[start:start+32];f=features[:,indices].contiguous()
                act={k:v[:,indices] for k,v in actions.items()}
                pd=guided_minibatch(actor,f);logp=actor.logprob(act,pd).flatten()
                logratio=logp-old[indices]
                approx=float(((logratio.exp()-1)-logratio).mean().detach())
                if approx>config['target_kl']:
                    early_stop=True;break
                with torch.no_grad(): refpd=guided_minibatch(reference,f)
                kl=sum((refpd[k].exp()*(refpd[k]-pd[k])).sum(-1).mean() for k in pd)
                actor_loss=ppo_loss(logp,old[indices],advantage[indices],config['ppo_clip'])+config['reference_kl_coefficient']*kl
                value_loss=(value(f[0:1]).flatten()-target[indices]).square().mean()
                if not torch.isfinite(actor_loss+value_loss): raise ValueError('nonfinite update')
                optimizer.zero_grad();(actor_loss+value_loss).backward()
                actor_norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
                value_norm=torch.nn.utils.clip_grad_norm_(value.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                history.append(dict(epoch=epoch,start=start,actor_loss=float(actor_loss.detach()),
                    value_loss=float(value_loss.detach()),approximate_kl=approx,reference_kl=float(kl.detach()),
                    actor_gradient_norm=float(actor_norm),value_gradient_norm=float(value_norm)))
            if early_stop: break
        heads=cpu_tree(dict(actor=actor.state_dict(),value=value.state_dict()))
        if not history or state_fingerprint(heads['actor'])==state_fingerprint(initial): raise ValueError('no actor update')
        if not all(torch.isfinite(v).all() for h in heads.values() for v in h.values()): raise ValueError('nonfinite candidate')
        # Hold actual visual/history features fixed; zero only RM semantic context.
        tv=[]
        with torch.no_grad():
            for i in range(0,len(rows),max(1,len(rows)//32)):
                f=features[:,i:i+1].clone();p1=guided_minibatch(actor,f)
                f[:,:,-CONTEXT_DIM:]=0;p0=guided_minibatch(actor,f)
                tv.append(max(float((p1[k].exp()-p0[k].exp()).abs().sum()/2) for k in p1))
        folder=out/variant;folder.mkdir()
        checkpoint=dict(scope=SCOPE,variant=variant,context_fields=list(CONTEXT_FIELDS),guidance=6.,
            base_model_sha256=first['base_model_sha256'],heads=heads,config=config,sources=sources,
            optimizer_updates=len(history),promoted=False)
        if args.stop_file.exists(): raise InterruptedError('STOP before checkpoint save')
        torch.save(checkpoint,folder/'checkpoint.pt')
        summaries[variant]=dict(variant=variant,config=config,sources=sources,rows=len(rows),
            replay_logp_max_error=max_error,updates=len(history),history=history,early_stop=early_stop,
            elapsed_s=time.monotonic()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
            context_zero_counterfactual_max_tv=max(tv),context_zero_counterfactual_mean_tv=sum(tv)/len(tv),
            actor_delta_l2=sum(float((v-initial[k]).square().sum()) for k,v in heads['actor'].items())**.5,
            checkpoint_sha256=hashlib.sha256((folder/'checkpoint.pt').read_bytes()).hexdigest(),promoted=False)
        write_json(folder/'training.json',summaries[variant])
        print(json.dumps({k:v for k,v in summaries[variant].items() if k not in ('history','sources','config')}),flush=True)
        del actor,value,reference,optimizer
        torch.cuda.empty_cache()
    write_json(out/'summary.json',{k:{key:v for key,v in val.items() if key!='history'} for k,val in summaries.items()})


if __name__=='__main__': main()
