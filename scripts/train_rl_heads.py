"""Run in the isolated WSL policy environment; no game or cloud connection."""
import argparse,copy,hashlib,json,sys,time,traceback
from pathlib import Path
import torch
from minecraftrl.rl_core import episode_returns,validate_training_data,validate_collector_result
from minecraftrl.rl_opt import ppo_loss
from minecraftrl.rl_policy import guided_minibatch as guided,cpu_tree
from minecraftrl.policies import model_kwargs
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from minecraftrl.rl_control import claim_automatic_stop,publish_completion

def main():
    p=argparse.ArgumentParser();p.add_argument('trajectory',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--additional',type=Path,nargs='*',default=[]);p.add_argument('--resume',type=Path)
    p.add_argument('--device',choices=['cuda'],default='cuda');p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--target-bundle',type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parents[1];output=args.output.resolve()
    if not output.is_relative_to(root/'results') or args.epochs not in (1,2):raise SystemExit('bounded project-local training required')
    output.mkdir(exist_ok=False,parents=True);stop=root/'results/STOP'
    if not stop.exists():raise SystemExit('STOP must be latched before training')
    claim_automatic_stop(stop,output/'STOP-before.txt');report=dict(completed=False,updates=0,scope='head_only',error=None)
    try:
        sys.path.insert(0,str(root/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.action_head import make_action_head
        paths=[args.trajectory,*args.additional]
        if len({p.resolve() for p in paths})!=len(paths):raise ValueError('duplicate trajectory')
        for path in paths:validate_collector_result(json.loads((path.parent/'summary.json').read_text()))
        episodes=[torch.load(path,map_location='cpu') for path in paths]
        for episode in episodes:validate_training_data(episode)
        data=episodes[0]
        reward_config=data.get('reward_config',dict(name='sparse',gamma=.99))
        if reward_config['name']!='sparse' and (args.target_bundle or args.resume):
            raise ValueError('shaping pilot requires a fresh optimizer and its matching returns')
        for episode in episodes[1:]:
            if episode.get('reward_config',dict(name='sparse',gamma=.99))!=reward_config:raise ValueError('mixed reward contracts')
            for group in ('actor','value'):
                left=data['initial_heads'][group];right=episode['initial_heads'][group]
                if left.keys()!=right.keys() or any(not torch.equal(left[k],right[k]) for k in left):
                    raise ValueError('mixed behavior checkpoints')
            if data['condition']['embedding_sha256']!=episode['condition']['embedding_sha256']:raise ValueError('mixed conditioning')
        rows=[row for episode in episodes for row in episode['rows']];torch.manual_seed(7301)
        device=torch.device(args.device);torch.cuda.reset_peak_memory_stats()
        actor=make_action_head(DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update()),
                               rows[0]['features'].shape[-1],**model_kwargs(root/'weights/steve1/2x.model')[1]).to(device)
        actor.load_state_dict(data['initial_heads']['actor']);reference=copy.deepcopy(actor).requires_grad_(False)
        value=torch.nn.Linear(rows[0]['features'].shape[-1],1).to(device);value.load_state_dict(data['initial_heads']['value'])
        bundle=None
        if args.target_bundle:
            if args.resume:raise ValueError('experimental target bundles require a fresh optimizer')
            from minecraftrl.rl_horizon import checked_training_bundle
            bundle=checked_training_bundle(torch.load(args.target_bundle,map_location='cpu'),
                [digest(path) for path in paths],len(rows))
            value.load_state_dict(bundle['value'])
            report['target_bundle_sha256']=digest(args.target_bundle)
            report['critic_initialization']='gamma-specific fitted returns; OOF actor advantages'
        parameters=list(actor.parameters())+list(value.parameters());optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=0)
        prior_updates=0
        if args.resume:
            saved=torch.load(args.resume,map_location='cpu')
            for group in ('actor','value'):
                if saved[group].keys()!=data['initial_heads'][group].keys() or any(not torch.equal(saved[group][k],data['initial_heads'][group][k]) for k in saved[group]):
                    raise ValueError('resume checkpoint differs from collected behavior')
            optimizer.load_state_dict(saved['optimizer']);prior_updates=saved['updates']
            report.update(parent_checkpoint_sha256=digest(args.resume),prior_updates=prior_updates)
        rewards=[r['reward'] for r in rows];values=[r['value'] for r in rows]
        advantages,returns=episode_returns(episodes)
        if bundle:advantages,returns=bundle['advantages'],bundle['returns']
        adv=torch.tensor(advantages,device=device);ret=torch.tensor(returns,device=device)
        if adv.std(unbiased=False)>1e-8:adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        old=torch.tensor([r['old_logp'] for r in rows],device=device)
        def batch(indices):
            selected=[rows[i] for i in indices]
            features=torch.cat([r['features'] for r in selected],1).to(device)
            actions={k:torch.cat([r['action'][k] for r in selected],1).to(device) for k in selected[0]['action']}
            return features,actions
        max_error=0.
        with torch.no_grad():
            for start in range(0,len(rows),16):
                ids=list(range(start,min(start+16,len(rows))));features,actions=batch(ids)
                actual=actor.logprob(actions,guided(actor,features)).flatten()
                max_error=max(max_error,(actual-old[ids]).abs().max().item())
        if max_error>2e-4:raise ValueError('behavior probability recomputation mismatch: '+str(max_error))
        report.update(trajectory_sha256=digest(args.trajectory),samples=len(rows),reward_sum=sum(rewards),
                      trajectories=[dict(path=str(path),sha256=digest(path),seed=episode.get('seed'),samples=len(episode['rows']),
                                         status=episode['status'],reward_sum=sum(r['reward'] for r in episode['rows'])) for path,episode in zip(paths,episodes)],
                      behavior_logp_max_error=max_error,trainable_parameters=sum(p.numel() for p in parameters),
                      probability_forward='per-transition collection shape; gradients aggregated per minibatch',
                      epochs=args.epochs,learning_rate=1e-5,gamma=.99,gae_lambda=.95,clip=.2,reference_kl_coefficient=.01)
        if bundle:report.update(gamma=bundle['gamma'],gae_lambda=bundle['lam'])
        report['reward_config']=reward_config
        before=cpu_tree(actor.state_dict());history=[]
        for epoch in range(args.epochs):
            order=torch.randperm(len(rows)).tolist()
            for start in range(0,len(order),16):
                if stop.exists():raise InterruptedError('STOP requested before optimizer update')
                ids=order[start:start+16];features,actions=batch(ids);optimizer.zero_grad(set_to_none=True)
                pd=guided(actor,features);logp=actor.logprob(actions,pd).flatten()
                with torch.no_grad():ref=guided(reference,features)
                kl=actor.kl_divergence(ref,pd).mean();prediction=value(features[0:1]).flatten()
                policy_loss=ppo_loss(logp,old[ids],adv[ids],.2);value_loss=.5*(prediction-ret[ids]).square().mean()
                loss=policy_loss+value_loss+.01*kl
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                if stop.exists():raise InterruptedError('STOP requested before optimizer step')
                optimizer.step();report['updates']+=1
                entry=dict(update=report['updates'],epoch=epoch,loss=loss.item(),policy_loss=policy_loss.item(),
                           value_loss=value_loss.item(),reference_kl=kl.item(),gradient_norm=norm.item())
                history.append(entry)
                with (output/'updates.jsonl').open('a') as log:log.write(json.dumps(entry)+'\n')
        after=cpu_tree(actor.state_dict());delta=max((after[k]-before[k]).abs().max().item() for k in before)
        if bundle or reward_config['name']!='sparse':
            final_kl=[]
            with torch.no_grad():
                for start in range(0,len(rows),16):
                    ids=list(range(start,min(start+16,len(rows))));features,actions=batch(ids)
                    final_kl.extend(actor.kl_divergence(guided(reference,features),guided(actor,features)).flatten().cpu().tolist())
            import math
            if not final_kl or not all(math.isfinite(x) for x in final_kl):raise ValueError('nonfinite final KL')
            report['final_reference_kl_mean']=sum(final_kl)/len(final_kl)
            report['runtime_comparison_eligible']=report['final_reference_kl_mean']<=.03
        if delta<=0:raise ValueError('no action-head parameter changed; no learning claim')
        checkpoint=dict(schema=1,scope='head_only',actor=after,value=cpu_tree(value.state_dict()),
                        optimizer=cpu_tree(optimizer.state_dict()),updates=prior_updates+report['updates'],condition=data['condition'],
                        trajectory_sha256=report['trajectory_sha256'])
        if bundle:checkpoint['return_estimator']=dict(gamma=bundle['gamma'],lam=bundle['lam'],bundle_sha256=digest(args.target_bundle))
        torch.save(checkpoint,output/'checkpoint.pt')
        restored=torch.load(output/'checkpoint.pt',map_location=device)
        clone=copy.deepcopy(actor);clone.load_state_dict(restored['actor'])
        clone_value=copy.deepcopy(value);clone_value.load_state_dict(restored['value'])
        cloned_optimizer=torch.optim.AdamW(list(clone.parameters())+list(clone_value.parameters()),lr=1e-5,weight_decay=0)
        cloned_optimizer.load_state_dict(restored['optimizer'])
        with torch.no_grad():
            features,actions=batch(list(range(min(16,len(rows)))))
            error=(actor.logprob(actions,guided(actor,features))-clone.logprob(actions,guided(clone,features))).abs().max().item()
        if error!=0 or not cloned_optimizer.state:raise ValueError('checkpoint reload mismatch')
        report.update(completed=True,actor_max_abs_change=delta,checkpoint_reload_logp_error=error,
                      total_updates=prior_updates+report['updates'],
                      optimizer_state_reloaded=True,checkpoint_sha256=digest(output/'checkpoint.pt'),history=history)
    except BaseException as exc:
        report['error']=repr(exc);(output/'traceback.txt').write_text(traceback.format_exc())
    finally:
        report['torch_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        report['torch_peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
        publish_completion(stop,'RL training ended; inspect before re-arming.');write_json(output/'training.json',report)
    print(json.dumps(report));return 0 if report['completed'] else 2

if __name__=='__main__':raise SystemExit(main())
