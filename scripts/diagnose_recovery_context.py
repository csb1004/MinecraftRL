"""Frozen-feature search-context intervention; no gameplay or optimizer."""
import json,sys
from pathlib import Path
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import guided
from minecraftrl.runtime import write_json

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'external/steve1'))
from gym3.types import DictType
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
from steve1.VPT.lib.action_head import make_action_head

base=root/'results/approach-recovery-20261007';state=json.loads((base/'experiment.json').read_text())
assert state['completed'] and (root/'results/STOP').exists()
record=next(r for r in state['runs'] if r['label']=='recovery-v4')
run=root/record['path'].replace('\\','/').split('MinecraftRL/',1)[1]
data=torch.load(run/'trajectory.pt',map_location='cpu');rows=data['rows'];selected=[];raw_before=json.loads((run/'initial-state.json').read_text())['server']
index=0
for line in (run/'events.jsonl').open():
    e=json.loads(line)
    if e['event']!='transition':continue
    row=rows[index]
    if row['reward_context'][0]==1. and raw_before['wood_v3_kind']==1 and raw_before['wood_v3_reachable']==0:
        selected.append((index,row,min(raw_before['wood_v3_distance'],16.)/16.))
    index+=1;raw_before=e['info']['workshop']
actor=make_action_head(DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update()),
    rows[0]['features'].shape[-1],**model_kwargs(root/'weights/steve1/2x.model')[1]).cuda()
actor.load_state_dict(data['initial_heads']['actor']);actor.eval();torch.set_num_threads(1)
records=[]
with torch.no_grad():
    for index,row,distance in selected[::max(1,len(selected)//64)][:64]:
        features=row['features'].cuda();normal=features.clone()
        normal[:,:,-6:]=torch.tensor([0.,1.,0.,0.,0.,distance],device='cuda')
        search_pd=guided(actor,features);normal_pd=guided(actor,normal)
        old=actor.logprob({k:v.cuda() for k,v in row['action'].items()},search_pd).item()
        records.append(dict(step=index+1,old_logp_error=abs(old-row['old_logp']),
                            kl=actor.kl_divergence(search_pd,normal_pd).item(),
                            total_variation={k:(search_pd[k].exp()-normal_pd[k].exp()).abs().sum().item()/2 for k in search_pd}))
report=dict(eligible_frames=len(selected),samples=len(records),rows=records,optimizer_updates=0,
            scope='same frozen features and head; only six RM context values changed; not escape success')
baseline_record=next(r for r in state['runs'] if r['label']=='baseline-v3')
baseline_run=root/baseline_record['path'].replace('\\','/').split('MinecraftRL/',1)[1]
baseline=torch.load(baseline_run/'trajectory.pt',map_location='cpu')
first_difference=next((i for i,(a,b) in enumerate(zip(baseline['rows'],rows)) if a['env_action']!=b['env_action']),None)
first_entry=next((i+1 for i,r in enumerate(rows) if r['reward_components']['recovery']['event']=='entered'),None)
comparison=dict(first_action_difference=None if first_difference is None else first_difference+1,
                first_recovery_entry=first_entry,
                actor_weights_identical=all(torch.equal(v,baseline['initial_heads']['actor'][k]) for k,v in data['initial_heads']['actor'].items()))
if first_difference is not None:
    a=baseline['rows'][first_difference];b=rows[first_difference]
    comparison.update(context_at_difference_identical=a['reward_context']==b['reward_context'],
                      feature_max_abs_difference=(a['features']-b['features']).abs().max().item(),
                      difference_before_feedback=first_entry is None or first_difference+1<=first_entry)
report['trajectory_comparison']=comparison
if records:
    report['max_logp_error']=max(r['old_logp_error'] for r in records)
    report['mean_kl']=sum(r['kl'] for r in records)/len(records)
    report['mean_total_variation']={k:sum(r['total_variation'][k] for r in records)/len(records) for k in records[0]['total_variation']}
    assert report['max_logp_error']<1e-5
write_json(base/'context-intervention.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
