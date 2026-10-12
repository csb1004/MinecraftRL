"""Held-out frozen-feature RM sensitivity before/after curriculum; no updates."""
import argparse,json,sys
from pathlib import Path
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import guided
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,default=root/'results/recovery-curriculum-20261007');base=p.parse_args().batch
state=json.loads((base/'experiment.json').read_text());assert state['completed'] and (root/'results/STOP').exists()
if not state['accepted_checkpoint']:
    write_json(base/'context-diagnosis.json',dict(evaluated=False,reason='no accepted trained checkpoint'));raise SystemExit(0)
import torch
sys.path.insert(0,str(root/'external/steve1'))
from gym3.types import DictType
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
from steve1.VPT.lib.action_head import make_action_head

def local(path):return root/path.replace('\\','/').split('MinecraftRL/',1)[1]
heldout=root/'results/rl-eval-104729-20261007T090125593992Z/run'
data=torch.load(heldout/'trajectory.pt',map_location='cpu');rows=data['rows'];indices=[];previous=json.loads((heldout/'initial-state.json').read_text())['server'];index=0
for line in (heldout/'events.jsonl').open():
    e=json.loads(line)
    if e['event']!='transition':continue
    if rows[index]['reward_context'][0]==1. and previous['wood_v3_kind']==1 and not previous['wood_v3_reachable']:
        indices.append((index,min(previous['wood_v3_distance'],16.)/16.))
    index+=1;previous=e['info']['workshop']
samples=indices[::max(1,len(indices)//64)][:64];assert samples
torch.set_num_threads(1);heads={};states={}
for name,path in (('parent',state['parent']),('candidate',state['accepted_checkpoint']['path'])):
    saved=torch.load(local(path),map_location='cpu');states[name]=saved['actor']
    head=make_action_head(DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update()),
        rows[0]['features'].shape[-1],**model_kwargs(root/'weights/steve1/2x.model')[1]).cuda()
    head.load_state_dict(saved['actor']);head.eval();heads[name]=head
records=[]
with torch.no_grad():
    for i,distance in samples:
        f=rows[i]['features'].cuda();normal=f.clone();normal[:,:,-6:]=torch.tensor([0.,1.,0.,0.,0.,distance],device='cuda')
        pd={name:guided(head,f) for name,head in heads.items()};entry=dict(step=i+1)
        for name,head in heads.items():
            alternate=guided(head,normal)
            entry[name]={k:(pd[name][k].exp()-alternate[k].exp()).abs().sum().item()/2 for k in pd[name]}
        entry['weight_change_kl']=heads['parent'].kl_divergence(pd['parent'],pd['candidate']).item();records.append(entry)
context_delta=max((states['candidate'][k][:,-6:]-v[:,-6:]).abs().max().item() for k,v in states['parent'].items() if v.ndim==2 and v.shape[-1]==rows[0]['features'].shape[-1])
report=dict(evaluated=True,samples=len(records),optimizer_updates=0,held_out_source=str(heldout),
            held_out_trajectory_sha256=digest(heldout/'trajectory.pt'),parent_sha256=state['parent_sha256'],candidate_sha256=state['accepted_checkpoint']['sha256'],
            source_used_for_training=False,context_weight_max_abs_change=context_delta,
            mean_context_total_variation={name:{k:sum(r[name][k] for r in records)/len(records) for k in ('camera','buttons')} for name in heads},
            mean_weight_change_kl=sum(r['weight_change_kl'] for r in records)/len(records),rows=records,
            scope='fixed old evaluation features only; not new rollout performance and not a training target')
write_json(base/'context-diagnosis.json',report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
