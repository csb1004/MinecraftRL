"""Read-only decomposition of the existing validation rollouts."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from audit_rl_pilot import local
from minecraftrl.runtime import write_json
p=argparse.ArgumentParser();p.add_argument('batch',type=Path);p.add_argument('diagnosis',type=Path);a=p.parse_args()
s=json.loads((a.batch/'experiment.json').read_text());metrics={}
with (a.diagnosis/'step-metrics.jsonl').open() as f:
 for line in f:
  r=json.loads(line)
  if r['split']=='eval':metrics[(r['label'],r['step'])]=r
runs=[r for r in s['runs'] if r['label'].startswith('validation-')];details=[];folders={}
for run in runs:
 print('phases '+run['label'],flush=True)
 folder=local(run['path'],ROOT);folders[run['label']]=folder
 first_mine=None;first_server=None;attack=0;forward=0;steps=0;mine_baseline=None
 with (folder/'events.jsonl').open() as f:
  for line in f:
   e=json.loads(line)
   if e['event']!='transition':continue
   steps+=1
   mined=sum(v for k,v in e['observation'].get('mine_block',{}).items() if k.endswith('_log'))
   if mine_baseline is None:mine_baseline=mined
   if mined>mine_baseline and first_mine is None:first_mine=steps
   if any(v for k,v in e['info']['workshop']['server_inventory'].items() if k.endswith('_log')) and first_server is None:first_server=steps
   attack+=bool(e['action']['attack'][0]);forward+=bool(e['action']['forward'][0])
 timing=json.loads((folder/'first-feature-timing.json').read_text())
 details.append(dict(label=run['label'],steps=steps,first_mined_log_stat=first_mine,first_server_log=first_server,
  mine_stat_to_inventory_steps=first_server-first_mine if first_mine is not None else None,
  attack_steps=attack,forward_steps=forward,first_backbone_forward_seconds=timing['seconds'],
  total_control_seconds=run['report']['elapsed_control_s']))
pairs=[]
for seed in s['validation_seeds']:
 for rep in (1,2,3):
  labels={phase:f'validation-{seed}-{rep}-{phase}' for phase in ('before','after')}
  ds={phase:torch.load(folders[label]/'trajectory.pt',map_location='cpu') for phase,label in labels.items()}
  first_feature=None;first_action=None;before_change=None;after_change=None
  for i,(x,y) in enumerate(zip(ds['before']['rows'],ds['after']['rows'])):
   if first_feature is None and not torch.equal(x['features'],y['features']):first_feature=i+1
   if first_action is None and x['env_action']!=y['env_action']:
    first_action=i+1
    before_change=metrics[(labels['before'],i+1)]['effective_action_changed']
    after_change=metrics[(labels['after'],i+1)]['effective_action_changed']
  pairs.append(dict(seed=seed,repeat=rep,first_feature_difference=first_feature,first_actual_action_difference=first_action,
   actor_swap_changes_action_on_before_features_at_first_branch=before_change,
   actor_swap_changes_action_on_after_features_at_first_branch=after_change,
   first_branch_feature_driven_with_no_actor_effect_on_either_input=first_action is not None and before_change is False and after_change is False))
result=dict(verified=True,gameplay_runs=0,optimizer_updates=0,
 scope='mining statistic is descriptive client observation; completion uses server inventory; saved feature swaps are not counterfactual future rollouts',
 runs=details,pairs=pairs)
write_json(a.diagnosis/'rollout-phases.json',result)
print(json.dumps(result))
