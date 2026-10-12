"""CPU-only stage-wise BC loss, after gameplay ends; no optimizer or new rollout."""
import json,sys
from pathlib import Path
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import guided_minibatch as guided
from minecraftrl.runtime import write_json

root=Path(__file__).resolve().parents[1];out=root/'results/teacher-bc-training-20261008/stage-loss.json'
if out.exists():raise FileExistsError(out)
if not (root/'results/STOP').exists():raise RuntimeError('diagnostics require stopped gameplay')
torch.set_num_threads(1)
data=torch.load(root/'results/teacher-bc-features-20261008/features.pt',map_location='cpu')
sources={s['id']:json.loads((root/s['run']/'teacher-rows.json').read_text()) for s in data['plan']['sessions']}
sys.path.insert(0,str(root/'external/steve1'))
from gym3.types import DictType
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
from steve1.VPT.lib.action_head import make_action_head
heads={}
for label,path in [('parent',root/data['plan']['parent']),('candidate',root/'results/teacher-bc-training-20261008/checkpoint.pt')]:
    head=make_action_head(DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update()),
                          data['rows'][0]['features'].shape[-1],**model_kwargs(root/'weights/steve1/2x.model')[1])
    head.load_state_dict(torch.load(path,map_location='cpu')['actor']);heads[label]=head.eval()
groups={}
for row in data['rows']:
    stage=sources[row['session']][row['frame']]['reward_components']['stage_before']
    groups.setdefault((row['split'],stage),[]).append(row)
report=[]
with torch.no_grad():
    for (split,stage),rows in groups.items():
        scores={}
        for label,head in heads.items():
            loss=0.
            for i in range(0,len(rows),16):
                chunk=rows[i:i+16];f=torch.cat([r['features'] for r in chunk],1)
                a={k:torch.cat([r['action'][k] for r in chunk],1) for k in chunk[0]['action']}
                loss-=head.logprob(a,guided(head,f)).sum().item()
            scores[label]=loss/len(rows)
        report.append(dict(split=split,stage=stage,rows=len(rows),nll=scores))
write_json(out,dict(rows=report,optimizer_updates=0,scope='Teacher-state diagnostics, not visited student-state success'))
print(json.dumps(report))
