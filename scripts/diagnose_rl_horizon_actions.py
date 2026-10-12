"""Same-input, same-noise actor swaps after gameplay; no counterfactual rollout claim."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'external/steve1'))
from gym3.types import DictType
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
from steve1.VPT.lib.action_head import make_action_head
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import guided
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from audit_rl_pilot import local
base=ROOT/'results/rl-horizon-20261007';out=base/'action-diagnosis';out.mkdir(exist_ok=False)
state=json.loads((base/'experiment.json').read_text());assert state['completed']
assert (ROOT/'results/STOP').exists()
torch.set_num_threads(1);mapping=CameraHierarchicalMapping(n_camera_bins=11)
paths={'parent':local(state['parent_checkpoint'],ROOT)}
paths.update({name:base/('training-'+name)/'checkpoint.pt' for name in state['training']})
hashes={name:digest(path) for name,path in paths.items()};heads={}
for name,path in paths.items():
    checkpoint=torch.load(path,map_location='cpu')
    head=make_action_head(DictType(**mapping.get_action_space_update()),checkpoint['value']['weight'].shape[1],
        **model_kwargs(ROOT/'weights/steve1/2x.model')[1]).cuda().eval().requires_grad_(False)
    head.load_state_dict(checkpoint['actor']);heads[name]=head
records=[];details=[];max_error=0.;count=0
with torch.no_grad(),(out/'steps.jsonl').open('w') as stream:
    for run in state['runs']:
        data=torch.load(local(run['path'],ROOT)/'trajectory.pt',map_location='cpu')
        torch.manual_seed(data['seed']);torch.cuda.manual_seed_all(data['seed'])
        changed={name:0 for name in heads if name!='parent'}
        first={name:None for name in changed};n=len(data['rows'])
        for i,row in enumerate(data['rows']):
            features=row['features'].cuda();rng=torch.cuda.get_rng_state()
            probabilities={name:guided(head,features) for name,head in heads.items()}
            samples={}
            for name,head in heads.items():
                torch.cuda.set_rng_state(rng);samples[name]=head.sample(probabilities[name])
            actual=samples[run['condition']]
            assert all(torch.equal(v.cpu(),row['action'][k]) for k,v in actual.items()),(run['label'],i)
            saved={k:v.cuda() for k,v in row['action'].items()}
            logp=heads[run['condition']].logprob(saved,probabilities[run['condition']]).item()
            error=abs(logp-row['old_logp']);assert error<2e-4
            max_error=max(max_error,error);count+=1
            factored={name:mapping.to_factored({k:v[:,0].cpu().numpy() for k,v in sample.items()}) for name,sample in samples.items()}
            swaps={name:any((factored['parent'][k]!=factored[name][k]).any() for k in factored['parent']) for name in changed}
            for name,diff in swaps.items():
                changed[name]+=int(diff)
                if diff and first[name] is None:first[name]=i+1
            entry=dict(label=run['label'],seed=run['seed'],repeat=run['repeat'],condition=run['condition'],step=i+1,
                actor_swap_changes_effective_action=swaps)
            stream.write(json.dumps(entry)+'\n');records.append(entry)
        details.append(dict(label=run['label'],seed=run['seed'],condition=run['condition'],repeat=run['repeat'],steps=n,
            same_input_actor_swap_changes=changed,first_same_input_actor_swap_change=first))
        print('replayed',run['label'],n,flush=True)
        write_json(out/'progress.json',dict(completed_runs=len(details),replayed=count,last=details[-1]))
audit=json.loads((base/'audit.json').read_text());lookup={(r['label'],r['step']):r for r in records};branches=[]
for pair in audit['trajectory_divergences']:
    step=pair['first_action_difference']
    if step is None:continue
    key=lambda name:(f"eval-{pair['repeat']}-{name}-{pair['seed']}",step)
    before=lookup[key('parent')]['actor_swap_changes_effective_action'][pair['condition']]
    after=lookup[key(pair['condition'])]['actor_swap_changes_effective_action'][pair['condition']]
    branches.append(dict(**pair,actor_swap_changes_on_parent_input=before,actor_swap_changes_on_candidate_input=after,
        first_action_branch_without_actor_swap_effect=not before and not after))
for name,path in paths.items():assert digest(path)==hashes[name]
write_json(out/'diagnosis.json',dict(completed=True,environment_actions=0,optimizer_updates=0,replayed_actions=count,
    max_probability_error=max_error,checkpoints=hashes,runs=details,first_action_branches=branches,
    scope='Actual saved-action replay and one-step actor interventions; does not simulate alternative future observations.'))
