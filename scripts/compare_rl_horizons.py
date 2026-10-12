"""Nine estimator diagnostics, seed-held-out critics, four fixed PPO target bundles."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
from audit_rl_pilot import local
from minecraftrl.rl_core import gae,validate_training_data,validate_collector_result
from minecraftrl.rl_value import fit_residual_value,seed_folds,calibration_passes
from minecraftrl.rl_horizon import censored_targets,checked_training_bundle
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

torch.set_num_threads(4)
base=ROOT/'results/rl-horizon-20261007';out=base/'diagnostic';out.mkdir(exist_ok=False)
assert (ROOT/'results/STOP').exists()
state=json.loads((ROOT/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
recovery=json.loads((base/'endpoint-recovery/recovery.json').read_text());assert recovery['completed']
assert digest(base/'endpoint-recovery/features.pt')==recovery['feature_file_sha256']
endpoints=torch.load(base/'endpoint-recovery/features.pt',map_location='cpu')
parent=torch.load(local(state['parent_checkpoint'],ROOT),map_location='cpu')
episodes=[];xs=[];hashes=[];ends={}
for item in state['training']['trajectories']:
    path=local(item['path'],ROOT);assert digest(path)==item['sha256']
    validate_collector_result(json.loads((path.parent/'summary.json').read_text()))
    e=torch.load(path,map_location='cpu');validate_training_data(e)
    assert e['seed'] in state['train_seeds'] and e['seed'] not in state['validation_seeds']+state['reserved_final_seeds']
    for group in ('actor','value'):
        assert all(torch.equal(e['initial_heads'][group][k],parent[group][k]) for k in parent[group])
    i=len(episodes);episodes.append(e);hashes.append(item['sha256'])
    xs.append(torch.cat([r['features'][0:1] for r in e['rows']],1).squeeze(0).double())
    if not e['terminal']:ends[i]=endpoints[item['sha256']]['features'].double()
zero_w=torch.zeros_like(parent['value']['weight'],dtype=torch.float64)
zero_b=torch.zeros_like(parent['value']['bias'],dtype=torch.float64)
def predict(model,x):return torch.nn.functional.linear(x,**model).flatten()
def target(i,gamma,bootstrap):
    e=episodes[i]
    return torch.tensor(censored_targets([r['reward'] for r in e['rows']],e['terminal'],gamma,bootstrap)[0],dtype=torch.float64)
def fit(indices,gamma):
    boot={i:.5 for i in indices if i in ends};history=[]
    x=torch.cat([xs[i] for i in indices])
    for iteration in range(3):
        y=torch.cat([target(i,gamma,boot.get(i,0.)) for i in indices])
        model=fit_residual_value(x,y,zero_w,zero_b,ridge=1.)
        new={i:float(predict(model,ends[i]).clamp(0,1)) for i in boot}
        history.append(dict(pass_index=iteration,used_bootstrap=boot,next_bootstrap=new));boot=new
    return model,history

diagnostics=[];critic_reports=[];bundles={}
with torch.no_grad():
    for gamma in (.99,.995,.999):
        oof={};boots={};folds=[]
        for seed,train,test in seed_folds(episodes):
            model,history=fit(train,gamma)
            for i in test:
                oof[i]=predict(model,xs[i]);boots[i]=float(predict(model,ends[i]).clamp(0,1)) if i in ends else 0.
            terminal=[i for i in test if episodes[i]['terminal']]
            if terminal:
                y=torch.cat([target(i,gamma,0) for i in terminal]);p=torch.cat([oof[i] for i in terminal])
                inherited=torch.cat([torch.tensor([r['value'] for r in episodes[i]['rows']],dtype=torch.float64) for i in terminal])
                mean=torch.cat([target(i,gamma,0) for i in train if episodes[i]['terminal']]).mean()
                folds.append(dict(seed=seed,test_indices=test,train_seeds=sorted({episodes[i]['seed'] for i in train}),
                    mse=(p-y).square().mean().item(),inherited_mse=(inherited-y).square().mean().item(),constant_mean=mean.item(),bootstrap_passes=history))
            else:folds.append(dict(seed=seed,test_indices=test,train_seeds=sorted({episodes[i]['seed'] for i in train}),mse=None,inherited_mse=None,bootstrap_passes=history))
            print('gamma',gamma,'held out',seed,flush=True)
        final,history=fit(list(range(len(episodes))),gamma)
        terminal_indices=[i for i,e in enumerate(episodes) if e['terminal']]
        y=torch.cat([target(i,gamma,0) for i in terminal_indices]);p=torch.cat([oof[i] for i in terminal_indices])
        old=torch.cat([torch.tensor([r['value'] for r in episodes[i]['rows']],dtype=torch.float64) for i in terminal_indices])
        const=torch.cat([torch.full_like(oof[i],next(f['constant_mean'] for f in folds if i in f['test_indices'])) for i in terminal_indices])
        early=torch.cat([torch.arange(len(xs[i])-1,-1,-1)>=100 for i in terminal_indices])
        start=torch.cat([torch.arange(len(xs[i]))<100 for i in terminal_indices])
        scores={label:dict(mse=(v-y).square().mean().item(),early_mse=(v[early]-y[early]).square().mean().item(),start100_mse=(v[start]-y[start]).square().mean().item()) for label,v in [('inherited',old),('calibrated',p),('constant',const)]}
        improved=sum(f['mse']<f['inherited_mse'] for f in folds if f['mse'] is not None)
        # Existing gate is reported, never rewritten to force acceptance; target changed at new gamma.
        critic_reports.append(dict(gamma=gamma,folds=folds,scores_on_completed_episodes=scores,
            inherited_critic_discount=.99,improved_seeds=improved,legacy_gate_diagnostic=calibration_passes(scores,improved,len(folds)),
            censored=[dict(episode=i,seed=episodes[i]['seed'],oof_bootstrap=boots[i],
                interval_violation_fraction=float(((oof[i]<0)|(oof[i]>torch.tensor(censored_targets([r['reward'] for r in episodes[i]['rows']],False,gamma,0)[2]))).double().mean())) for i in ends],
            full_fit_bootstrap_passes=history))
        for lam in (.95,.99,1.):
            advantages=[];returns=[];raw_old=[];lower=[];upper=[];mask=[]
            for i,e in enumerate(episodes):
                rewards=[r['reward'] for r in e['rows']]
                a,r=gae(rewards,oof[i].tolist(),boots[i],terminal=e['terminal'],gamma=gamma,lam=lam)
                aa,_=gae(rewards,[r['value'] for r in e['rows']],e['bootstrap'],terminal=e['terminal'],gamma=gamma,lam=lam)
                lo,_=gae(rewards,oof[i].tolist(),0.,terminal=e['terminal'],gamma=gamma,lam=lam)
                hi,_=gae(rewards,oof[i].tolist(),1.,terminal=e['terminal'],gamma=gamma,lam=lam)
                advantages+=a;returns+=r;raw_old+=aa;lower+=lo;upper+=hi
                mask += [len(rewards)-1-j>=100 for j in range(len(rewards))]
            def stats(values):
                v=torch.tensor(values,dtype=torch.float64);n=(v-v.mean())/(v.std(unbiased=False)+1e-8);m=torch.tensor(mask)
                return dict(mean=v.mean().item(),std=v.std(unbiased=False).item(),positive=int((v>0).sum()),
                    normalized_positive=int((n>0).sum()),early_normalized_positive=int((n[m]>0).sum()),early_count=int(m.sum()))
            diagnostics.append(dict(gamma=gamma,lam=lam,new_critic=stats(advantages),inherited_fixed_critic_sensitivity=stats(raw_old),
                censored_bootstrap_0_sensitivity=stats(lower),censored_bootstrap_1_sensitivity=stats(upper),
                direct_terminal_reward_coefficient_100=(gamma*lam)**100))
            if gamma in (.99,.999) and lam in (.95,1.):
                name=('g99' if gamma==.99 else 'g999')+('-l95' if lam==.95 else '-l1')
                bundle=dict(schema=1,gamma=gamma,lam=lam,trajectory_sha256=hashes,advantages=advantages,returns=returns,
                    value={k:v.float() for k,v in final.items()},scope='cross-fitted gamma-specific value, unchanged sparse reward',
                    parent_sha256=state['parent_sha256'])
                checked_training_bundle(bundle,hashes,len(advantages));torch.save(bundle,out/(name+'.pt'))
                bundles[name]=dict(path=str(out/(name+'.pt')),sha256=digest(out/(name+'.pt')),gamma=gamma,lam=lam)
        write_json(out/'progress.json',dict(critics=critic_reports,estimators=diagnostics))
for item in state['training']['trajectories']:assert digest(local(item['path'],ROOT))==item['sha256']
write_json(out/'comparison.json',dict(completed=True,samples=sum(len(x) for x in xs),episodes=len(episodes),
    terminal_episodes=sum(e['terminal'] for e in episodes),critics=critic_reports,estimators=diagnostics,bundles=bundles,
    actor_updates=0,linear_solves=3*3*(len(state['train_seeds'])+1),evaluation_used_for_fitting=False,
    timeout_targets_are_estimates=True,parent_sha256=state['parent_sha256']))
print('Completed estimator diagnostics and four immutable training bundles.',flush=True)
