"""Fixed nonlinear critic candidate, training-world cross-validation only."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from minecraftrl.rl_value import discounted_returns,seed_folds,calibration_passes
from minecraftrl.rl_core import validate_training_data,validate_collector_result
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from audit_rl_pilot import local
p=argparse.ArgumentParser();p.add_argument('batch',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
base=args.batch.resolve();out=args.output.resolve();assert out.is_relative_to(ROOT/'results') and (ROOT/'results/STOP').exists();out.mkdir(exist_ok=False,parents=True)
s=json.loads((base/'experiment.json').read_text());parent=local(s['parent_checkpoint'],ROOT);saved=torch.load(parent,map_location='cpu');torch.set_num_threads(4)
assert s['completed'] and not s['errors'] and digest(parent)==s['parent_sha256']
episodes=[];xs=[];ys=[];rs=[];hashes={}
for item in s['training']['trajectories']:
 path=local(item['path'],ROOT);assert digest(path)==item['sha256'];hashes[str(path)]=item['sha256']
 validate_collector_result(json.loads((path.parent/'summary.json').read_text()));e=torch.load(path,map_location='cpu');validate_training_data(e)
 assert e['seed']==item['seed'] and len(e['rows'])==item['samples']
 assert all(torch.equal(e['initial_heads']['value'][k],saved['value'][k]) for k in ('weight','bias'))
 assert e['seed'] in s['train_seeds'] and e['seed'] not in s['validation_seeds']+s['reserved_final_seeds']
 episodes.append(e);xs.append(torch.cat([r['features'][0:1] for r in e['rows']],1).squeeze(0).float())
 ys.append(torch.tensor(discounted_returns([r['reward'] for r in e['rows']],e['bootstrap'],e['terminal'])))
 rs.append(torch.arange(len(e['rows'])-1,-1,-1))
assert {e['seed'] for e in episodes}==set(s['train_seeds']) and len(hashes)==len(episodes)
folds=[];collected=[];predictions=[];steps=0
for seed,train,test in seed_folds(episodes):
 print('MLP fit excluding '+str(seed),flush=True)
 x=torch.cat([xs[i] for i in train]);y=torch.cat([ys[i] for i in train]);tx=torch.cat([xs[i] for i in test]);ty=torch.cat([ys[i] for i in test]);early=torch.cat([rs[i] for i in test])>=100
 mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-6);x=(x-mean)/scale;test_x=(tx-mean)/scale
 torch.manual_seed(40321)
 model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,1),torch.nn.Sigmoid())
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 for epoch in range(20):
  order=torch.randperm(len(x))
  for start in range(0,len(order),256):
   ids=order[start:start+256];opt.zero_grad(set_to_none=True);loss=(model(x[ids]).flatten()-y[ids]).square().mean()
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
 with torch.no_grad():
  new=model(test_x).flatten();old=torch.nn.functional.linear(tx,**saved['value']).flatten();constant=torch.full_like(ty,y.mean())
  assert torch.isfinite(new).all() and ((new>=0)&(new<=1)).all()
  mse=lambda v:(v-ty).square().mean().item()
  fold=dict(seed=seed,train_seeds=sorted({episodes[i]['seed'] for i in train}),train_samples=len(y),test_samples=len(ty),
   inherited_mse=mse(old),calibrated_mse=mse(new),constant_mse=mse(constant),fitting_mse=(model(x).flatten()-y).square().mean().item())
 folds.append(fold);collected.append((ty,old,new,constant,early));start=0
 for i in test:
  n=len(ys[i]);predictions.append(dict(episode_index=i,seed=seed,predictions=new[start:start+n].tolist()));start+=n
 torch.save(dict(model=model.state_dict(),mean=mean,scale=scale,excluded_seed=seed,scope='diagnostic critic; no actor'),out/f'value-excluding-{seed}.pt')
 write_json(out/'progress.json',dict(folds=folds,value_optimizer_steps=steps))
target,old,new,constant,early=[torch.cat([row[i] for row in collected]) for i in range(5)]
scores={name:dict(mse=(v-target).square().mean().item(),early_mse=(v[early]-target[early]).square().mean().item(),explained_variance=1-(v-target).var(unbiased=False).item()/target.var(unbiased=False).item()) for name,v in (('inherited',old),('calibrated',new),('constant',constant))}
improved=sum(f['calibrated_mse']<f['inherited_mse'] for f in folds)
passed=calibration_passes(scores,improved,len(folds))
for path,h in hashes.items():assert digest(path)==h
assert digest(parent)==s['parent_sha256'] and (ROOT/'results/STOP').exists()
report=dict(completed=True,gate_passed=bool(passed),actor_updates=0,value_optimizer_steps=steps,training_seeds=s['train_seeds'],evaluation_used_for_fitting=False,samples=len(target),folds=folds,scores=scores,improved_seeds=improved,
 architecture='2048-64-ReLU-1-sigmoid',epochs=20,learning_rate=.001,minibatch=256,seed=40321,weight_decay=.01,
 scope='fixed diagnostic critic candidate; no policy update or new gameplay',parent_sha256=s['parent_sha256'],trajectories=hashes)
write_json(out/'calibration.json',report);write_json(out/'out-of-fold-predictions.json',predictions)
print(json.dumps({k:v for k,v in report.items() if k not in ('folds','trajectories')}))
