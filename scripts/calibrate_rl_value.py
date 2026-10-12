"""Leave-one-world-out value comparisons on completed training collections only."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from minecraftrl.rl_value import discounted_returns,seed_folds,fit_residual_value,calibration_passes
from minecraftrl.rl_core import validate_training_data,validate_collector_result
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from audit_rl_pilot import local

p=argparse.ArgumentParser();p.add_argument('batch',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
base=args.batch.resolve();out=args.output.resolve();assert out.is_relative_to(ROOT/'results')
out.mkdir(exist_ok=False,parents=True);assert (ROOT/'results/STOP').exists()
s=json.loads((base/'experiment.json').read_text());t=s['training'];torch.set_num_threads(4)
assert s['completed'] and not s['errors']
parent=local(s['parent_checkpoint'],ROOT);checkpoint=torch.load(parent,map_location='cpu');w=checkpoint['value']['weight'];b=checkpoint['value']['bias']
assert digest(parent)==s['parent_sha256']
episodes=[];xs=[];ys=[];remaining=[];hashes={}
for item in t['trajectories']:
 path=local(item['path'],ROOT);assert digest(path)==item['sha256'];hashes[str(path)]=item['sha256']
 validate_collector_result(json.loads((path.parent/'summary.json').read_text()))
 e=torch.load(path,map_location='cpu');validate_training_data(e);episodes.append(e)
 assert e['seed']==item['seed'] and len(e['rows'])==item['samples']
 assert e['seed'] in s['train_seeds'] and e['seed'] not in s['validation_seeds']+s['reserved_final_seeds']
 assert all(torch.equal(e['initial_heads']['value'][k],checkpoint['value'][k]) for k in ('weight','bias'))
 xs.append(torch.cat([r['features'][0:1] for r in e['rows']],1).squeeze(0).double())
 ys.append(torch.tensor(discounted_returns([r['reward'] for r in e['rows']],e['bootstrap'],e['terminal']),dtype=torch.float64))
 remaining.append(torch.arange(len(e['rows'])-1,-1,-1))
assert {e['seed'] for e in episodes}==set(s['train_seeds']) and len(hashes)==len(episodes)
folds=[];all_y=[];all_old=[];all_new=[];all_constant=[];all_early=[];predictions=[]
with torch.no_grad():
 for seed,train,test in seed_folds(episodes):
  print('fit excluding '+str(seed),flush=True)
  x=torch.cat([xs[i] for i in train]);y=torch.cat([ys[i] for i in train]);fit=fit_residual_value(x,y,w,b,ridge=1.)
  tx=torch.cat([xs[i] for i in test]);ty=torch.cat([ys[i] for i in test]);early=torch.cat([remaining[i] for i in test])>=100
  old=torch.nn.functional.linear(tx,w.double(),b.double()).flatten();new=torch.nn.functional.linear(tx,**fit).flatten();constant=torch.full_like(ty,y.mean())
  mse=lambda pred: (pred-ty).square().mean().item()
  fold=dict(seed=seed,train_seeds=sorted({episodes[i]['seed'] for i in train}),excluded_episode_indices=test,
   train_samples=len(y),test_samples=len(ty),inherited_mse=mse(old),calibrated_mse=mse(new),constant_mse=mse(constant),
   fitting_mse=(torch.nn.functional.linear(x,**fit).flatten()-y).square().mean().item())
  folds.append(fold);all_y.append(ty);all_old.append(old);all_new.append(new);all_constant.append(constant);all_early.append(early)
  start=0
  for i in test:
   n=len(ys[i]);predictions.append(dict(episode_index=i,seed=seed,predictions=new[start:start+n].tolist()));start+=n
  write_json(out/'progress.json',dict(folds=folds))
 target=torch.cat(all_y);old=torch.cat(all_old);new=torch.cat(all_new);constant=torch.cat(all_constant);early=torch.cat(all_early)
 scores={name:dict(mse=(v-target).square().mean().item(),early_mse=(v[early]-target[early]).square().mean().item(),
  explained_variance=1-(v-target).var(unbiased=False).item()/target.var(unbiased=False).item()) for name,v in (('inherited',old),('calibrated',new),('constant',constant))}
 improved=sum(f['calibrated_mse']<f['inherited_mse'] for f in folds)
 passed=calibration_passes(scores,improved,len(folds))
 full=fit_residual_value(torch.cat(xs),torch.cat(ys),w,b,ridge=1.)
 torch.save(dict(value=full,scope='diagnostic value only; no actor',ridge=1.,parent_sha256=s['parent_sha256']),out/'value-only.pt')
for path,h in hashes.items():assert digest(path)==h
assert digest(parent)==s['parent_sha256'] and (ROOT/'results/STOP').exists()
report=dict(completed=True,gate_passed=bool(passed),actor_updates=0,optimizer_steps=0,linear_solves=len(folds)+1,
 training_seeds=s['train_seeds'],evaluation_used_for_fitting=False,samples=len(target),folds=folds,
 scores=scores,improved_seeds=improved,ridge=1.,gamma=.99,scope='cross-world prediction of observed returns; not policy performance',
 parent_sha256=s['parent_sha256'],trajectories=hashes,value_sha256=digest(out/'value-only.pt'))
write_json(out/'calibration.json',report);write_json(out/'out-of-fold-predictions.json',predictions)
print(json.dumps({k:v for k,v in report.items() if k not in ('folds','trajectories')}))
