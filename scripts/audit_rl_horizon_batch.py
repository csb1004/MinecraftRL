"""Audit every actual comparison, retaining failures and unrun conditions."""
import json,statistics,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'external/steve1'))
from audit_rl_pilot import audit_run,local
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from minecraftrl.rl_expanded import validate_rollout
base=ROOT/'results/rl-horizon-20261007'
state=json.loads((base/'experiment.json').read_text())
assert state['completed'] and not state['errors']
source=json.loads((ROOT/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
assert digest(local(state['parent_checkpoint'],ROOT))==state['parent_sha256']
for item in source['training']['trajectories']:assert digest(local(item['path'],ROOT))==item['sha256']
heads={'parent':torch.load(local(state['parent_checkpoint'],ROOT),map_location='cpu')}
for name,report in state['training'].items():
    folder=base/('training-'+name)
    assert report['completed'] and report['epochs']==1 and report['samples']==4855 and report['updates']==304
    assert report['behavior_logp_max_error']<=2e-4 and report['checkpoint_reload_logp_error']==0
    assert digest(folder/'checkpoint.pt')==report['checkpoint_sha256']
    assert digest(base/'diagnostic'/(name+'.pt'))==report['target_bundle_sha256']
    heads[name]=torch.load(folder/'checkpoint.pt',map_location='cpu')
    assert heads[name]['return_estimator']['gamma']==report['gamma']
    assert all(v['step'].item()==report['updates'] for v in heads[name]['optimizer']['state'].values())
results=[]
for item in state['runs']:
    folder=local(item['path'],ROOT);head=heads[item['condition']]
    validate_rollout(item['report'],'eval')
    result=audit_run(folder,head['actor'],head['value'],'eval')
    assert not item['report']['rl_training_eligible']
    first=None
    with (folder/'events.jsonl').open() as f:
        for line in f:
            event=json.loads(line)
            if event['event']=='inference_finished':
                first=event['call_finished_s']-event['call_started_s'];break
    results.append(dict(seed=item['seed'],condition=item['condition'],repeat=item['repeat'],
        first_inference_s=first,path=item['path'],**result))
aggregates={}
for name in heads:
    runs=[r for r in results if r['condition']==name]
    seeds={}
    for seed in state['validation_seeds']:
        rows=[r for r in runs if r['seed']==seed];ok=[r for r in rows if r['success']]
        seeds[str(seed)]=dict(valid=len(rows),successes=len(ok),successful_ticks=[r['steps'] for r in ok],
            median_success_ticks=statistics.median(r['steps'] for r in ok) if ok else None,
            restricted_mean_control_s=statistics.mean(min(r['seconds'],180.) if r['success'] else 180. for r in rows) if rows else None)
    aggregates[name]=dict(valid=len(runs),successes=sum(r['success'] for r in runs),seeds=seeds)
comparisons=[]
parent=aggregates['parent']
for name,a in aggregates.items():
    if name=='parent':continue
    ratio={s:(a['seeds'][s]['median_success_ticks']/parent['seeds'][s]['median_success_ticks']
        if a['seeds'][s]['median_success_ticks'] and parent['seeds'][s]['median_success_ticks'] else None) for s in parent['seeds']}
    comparisons.append(dict(condition=name,success_delta=a['successes']-parent['successes'],per_seed_tick_ratio=ratio,
        descriptive_faster_in_both_seeds=a['valid']==4 and a['successes']==4 and all(x is not None and x<1 for x in ratio.values())))
divergences=[]
for repeat in (1,2):
    for seed in state['validation_seeds']:
        matches=[r for r in state['runs'] if r['repeat']==repeat and r['seed']==seed]
        original=next((r for r in matches if r['condition']=='parent'),None)
        if original is None:continue
        before=torch.load(local(original['path'],ROOT)/'trajectory.pt',map_location='cpu')
        for run in matches:
            if run['condition']=='parent':continue
            after=torch.load(local(run['path'],ROOT)/'trajectory.pt',map_location='cpu')
            feature=None;action=None
            for i,(left,right) in enumerate(zip(before['rows'],after['rows'])):
                if feature is None and not torch.equal(left['features'],right['features']):feature=i+1
                if action is None and left['env_action']!=right['env_action']:action=i+1
            divergences.append(dict(seed=seed,repeat=repeat,condition=run['condition'],
                first_feature_difference=feature,first_action_difference=action,
                compared_steps=min(len(before['rows']),len(after['rows'])),
                note='Descriptive actual trajectories; feature divergence does not identify its causal source.'))
assert all(c['state_identical'] for c in state['comparisons'].values())
assert digest(ROOT/'docs/research-spec.md')=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
report=dict(verified=True,actual_runs=len(results),results=results,aggregates=aggregates,comparisons=comparisons,
    trajectory_divergences=divergences,
    updates=sum(x['updates'] for x in state['training'].values()),technical_attempts=state['technical_attempts'],unrun=state['unrun'],
    identical_initial_state_comparisons=len(state['comparisons']),pixel_identical_comparisons=sum(c['pixel_identical'] for c in state['comparisons'].values()),
    default_policy_promoted=False,final_seeds_unrun=state['final_seeds_unrun'],
    scope='four matched experimental branches on two reused development worlds, not generalization proof')
write_json(base/'audit.json',report)
print(json.dumps(dict(verified=True,aggregates=aggregates,comparisons=comparisons),indent=2))
