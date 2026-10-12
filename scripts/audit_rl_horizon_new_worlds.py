"""Audit the fixed new-world sample and its preregistered decision rule."""
import json,sys
from pathlib import Path
import torch
from PIL import Image,ImageChops
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'external/steve1'))
from audit_rl_pilot import audit_run,local
from minecraftrl.rl_expanded import validate_rollout
from minecraftrl.rl_validation import summarize_comparison
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from minecraftrl.rl_worlds import checked_world
torch.set_num_threads(2)
base=ROOT/'results/rl-horizon-new-worlds-20261007'
state=json.loads((base/'experiment.json').read_text())
assert state['completed'] and not state['errors'] and not state['unrun']
assert len(state['preparations'])==5 and len(state['runs'])==30
heads={}
for name,item in state['checkpoints'].items():
    path=local(item['path'],ROOT);assert digest(path)==item['sha256']
    heads[name]=torch.load(path,map_location='cpu')
for item in state['worlds'].values():assert digest(local(item['path'],ROOT))==item['sha256']
for seed,item in state['worlds'].items():checked_world(local(item['path'],ROOT),int(seed))
details=[];jars=set()
for preparation in state['preparations']:
    receipt=json.loads((local(preparation['path'],ROOT)/'runtime-jar-cache.json').read_text())
    jars.add(receipt['sha256'])
for run in state['runs']:
    folder=local(run['path'],ROOT);checkpoint=heads[run['condition']]
    jars.add(json.loads((folder/'runtime-jar-cache.json').read_text())['sha256'])
    validate_rollout(run['report'],'eval');assert not run['report']['rl_training_eligible']
    result=audit_run(folder,checkpoint['actor'],checkpoint['value'],'eval')
    first=None
    with (folder/'events.jsonl').open() as stream:
        for line in stream:
            event=json.loads(line)
            if event['event']=='inference_finished':
                first=event['call_finished_s']-event['call_started_s'];break
    data=torch.load(folder/'trajectory.pt',map_location='cpu')
    if run['report']['status']=='time_limit':
        assert not data['terminal'] and data['condition']['recurrent_commit_mode']=='dispatch_commit'
    details.append(dict(label=run['label'],condition=run['condition'],seed=run['seed'],repeat=run['repeat'],
        first_inference_s=first,bootstrap=data['bootstrap'],terminal=data['terminal'],**result))
    print('audited',run['label'],flush=True)
assert all(c['state_identical'] for c in state['comparisons'].values())
assert len(jars)==1,'Runtime JAR changed within matched experiment'
def actions(folder):
    with (folder/'events.jsonl').open() as stream:
        for line in stream:
            event=json.loads(line)
            if event['event']=='transition':yield event['action']
def first_branch(parent,other):
    difference=None;equal_pixels=None;compared=0
    for step,(a,b) in enumerate(zip(actions(parent),actions(other)),1):
        compared=step
        if a!=b:
            difference=step;filename=f'frames/{step-1:06d}.png'
            with Image.open(parent/filename) as first,Image.open(other/filename) as second:
                equal_pixels=ImageChops.difference(first,second).getbbox() is None
            break
    return dict(first_action_difference=difference,input_pixels_identical_at_branch=equal_pixels,compared_prefix_steps=compared)
by_key={(r['seed'],r['repeat'],r['condition']):r for r in state['runs']}
divergences=[]
for seed in state['seeds']:
    for repeat in (1,2):
        parent=local(by_key[seed,repeat,'parent']['path'],ROOT)
        for name in ('control','candidate'):
            other=local(by_key[seed,repeat,name]['path'],ROOT)
            divergences.append(dict(seed=seed,repeat=repeat,condition=name,**first_branch(parent,other)))
repeats=[]
for seed in state['seeds']:
    for name in ('parent','control','candidate'):
        rows=[by_key[seed,repeat,name] for repeat in (1,2)]
        branch=first_branch(*(local(r['path'],ROOT) for r in rows))
        repeats.append(dict(seed=seed,condition=name,statuses=[r['report']['status'] for r in rows],
            steps=[r['report']['steps'] for r in rows],**branch))
summary=summarize_comparison(state['runs'],state['seeds']);assert summary['complete']
assert digest(ROOT/'docs/research-spec.md')=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
write_json(base/'audit.json',dict(verified=True,summary=summary,runs=details,optimizer_updates=0,
    prepared_worlds=len(state['preparations']),actual_evaluations=len(details),actual_actions=sum(r['steps'] for r in details),
    technical_attempts=state['technical_attempts'],unrun=state['unrun'],
    initial_state_comparisons=len(state['comparisons']),pixel_identical_comparisons=sum(c['pixel_identical'] for c in state['comparisons'].values()),
    runtime_jar_sha256=next(iter(jars)),saved_world_seeds_verified=True,
    trajectory_divergences=divergences,within_policy_repeats=repeats,
    checkpoints=state['checkpoints'],final_seeds_unrun=state['final_seeds_unrun'],default_policy_promoted=False,
    scope='Five previously unused development worlds; independent of training, not a final locked evaluation.'))
print(json.dumps(summary,indent=2))
