"""Read-only training-signal diagnostics on the frozen training data."""
import argparse,json,statistics,sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root/'scripts'))
from audit_rl_pilot import local
from minecraftrl.rl_core import episode_returns
from minecraftrl.runtime import write_json
p=argparse.ArgumentParser();p.add_argument('batch',type=Path);args=p.parse_args();base=args.batch
t=json.loads((base/'training/training.json').read_text());assert t['completed'];torch.set_num_threads(1)
checkpoint=torch.load(base/'training/checkpoint.pt',map_location='cpu');weight=checkpoint['value']['weight'];bias=checkpoint['value']['bias'];episodes=[];all_adv=[];old_squared=[];new_squared=[]
with torch.no_grad():
 for item in t['trajectories']:
  data=torch.load(local(item['path'],root),map_location='cpu');adv,targets=episode_returns([data]);all_adv.extend(adv)
  features=torch.cat([r['features'][0:1] for r in data['rows']],1).squeeze(0)
  new=torch.nn.functional.linear(features,weight,bias).flatten();old=torch.tensor([r['value'] for r in data['rows']]);target=torch.tensor(targets)
  old_squared.extend((old-target).square().tolist());new_squared.extend((new-target).square().tolist())
  mass=sum(abs(v) for v in adv)
  episodes.append(dict(seed=data['seed'],status=data['status'],steps=len(adv),reward=sum(r['reward'] for r in data['rows']),bootstrap=data['bootstrap'],first_old_value=old[0].item(),first_new_value=new[0].item(),advantage_abs_mass=mass,last_64_advantage_abs_mass_fraction=sum(abs(v) for v in adv[-64:])/mass if mass else None))
values=torch.tensor(all_adv);history=t['history']
normalized=(values-values.mean())/(values.std(unbiased=False)+1e-8) if values.std(unbiased=False)>1e-8 else values
result=dict(updates=0,scope='descriptive training-data signal; not independent evaluation or full gradient attribution',episodes=episodes,samples=len(all_adv),normalized_positive_advantages=int((normalized>0).sum()),normalized_negative_advantages=int((normalized<0).sum()),positive_advantages=int((values>0).sum()),negative_advantages=int((values<0).sum()),zero_advantages=int((values==0).sum()),raw_advantage_std=values.std(unbiased=False).item(),absolute_advantage_quantiles={str(q):values.abs().quantile(q).item() for q in (0.,.25,.5,.75,.95,1.)},value_mse_against_fixed_training_targets_before=statistics.mean(old_squared),value_mse_against_fixed_training_targets_after=statistics.mean(new_squared),minibatch_reference_kl_max=max(r['reference_kl'] for r in history),gradient_clip_activated_updates=sum(r['gradient_norm']>1 for r in history),optimizer_updates_inspected=len(history))
write_json(base/'learning-signal.json',result);print(json.dumps({k:v for k,v in result.items() if k!='episodes'}))
