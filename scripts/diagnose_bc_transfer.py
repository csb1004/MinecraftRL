"""CPU-only fixed-observation comparison; evaluation features never train weights."""
import json,sys
from pathlib import Path
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import guided
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];base=root/'results/bc-comparison-20261008'
out=base/'fixed-wood-features.json'
if out.exists():raise FileExistsError(out)
state=json.loads((base/'experiment.json').read_text())
run=next(r for r in state['runs'] if r['label']=='long_wall-parent' and r['audited'])
def local(path):return root/path.replace('\\','/').split('MinecraftRL/',1)[1]
source=local(run['path'])/'trajectory.pt';data=torch.load(source,map_location='cpu')
assert data['split']=='eval';rows=data['rows'];indices=list(range(0,len(rows),max(1,len(rows)//64)))[:64]
sys.path.insert(0,str(root/'external/steve1'));torch.set_num_threads(1)
from gym3.types import DictType
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
from steve1.VPT.lib.action_head import make_action_head
heads={}
for name in ('parent','candidate'):
    saved=torch.load(local(state[name]),map_location='cpu')
    head=make_action_head(DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update()),
                         rows[0]['features'].shape[-1],**model_kwargs(root/'weights/steve1/2x.model')[1])
    head.load_state_dict(saved['actor']);head.eval().requires_grad_(False);heads[name]=head
measures=[]
with torch.no_grad():
    for i in indices:
        f=rows[i]['features'];pd={name:guided(head,f) for name,head in heads.items()}
        measures.append(dict(frame=i,kl=heads['parent'].kl_divergence(pd['parent'],pd['candidate']).item(),
            tv={k:.5*(pd['parent'][k].exp()-pd['candidate'][k].exp()).abs().sum().item() for k in pd['parent']}))
report=dict(samples=len(indices),source_sha256=digest(source),optimizer_updates=0,source_used_for_training=False,
            mean_kl=sum(m['kl'] for m in measures)/len(measures),
            mean_tv={k:sum(m['tv'][k] for m in measures)/len(measures) for k in ('camera','buttons')},
            scope='CPU frozen wood-goal evaluation observations; distribution change, not policy success')
write_json(out,report);print(json.dumps(report))
