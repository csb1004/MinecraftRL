"""Post-game stage losses for the fixed final head/tail comparison, no updates."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.policy_tail import load_tail
from minecraftrl.rl_policy import guided_minibatch as guided
from minecraftrl.runtime import write_json

root=Path(__file__).resolve().parents[1];base=root/'results/diverse-bc-lr5e7-comparison-20261008'
out=base/'stage-loss.json'
if out.exists():raise FileExistsError(out)
batch=json.loads((base/'experiment.json').read_text())
assert len(batch['runs'])==4 and all(r['audited'] for r in batch['runs']) and (root/'results/STOP').exists()
assert not batch['errors'] or batch['errors']==["AssertionError('initial state/pixel mismatch')"]
torch.set_num_threads(1)
data=torch.load(root/'results/diverse-bc-features-20261008/features.pt',map_location='cpu')
sources={s['id']:json.loads((root/s['run']/'teacher-rows.json').read_text()) for s in data['plan']['sessions']}
sys.path.insert(0,str(root/'external/steve1'))
from gym3.types import DictType
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
from steve1.VPT.lib.action_head import make_action_head
from steve1.VPT.lib.torch_util import set_default_torch_device
from steve1.embed_conditioned_policy import MinecraftAgentPolicy
set_default_torch_device('cpu')
space=DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())
kwargs,head_kwargs=model_kwargs(root/'weights/steve1/2x.model')
skeleton=MinecraftAgentPolicy(space,kwargs,head_kwargs)
net=SimpleNamespace(lastlayer=skeleton.net.lastlayer,final_ln=skeleton.net.final_ln);del skeleton
tail=torch.nn.Sequential(net.lastlayer,net.final_ln).cuda().eval()
groups={}
for row in data['rows']:
    stage=sources[row['session']][row['frame']]['reward_components']['stage_before']
    groups.setdefault((row['split'],stage),[]).append(row)
report={key:dict(split=key[0],stage=key[1],rows=len(rows),nll={}) for key,rows in groups.items()}
with torch.no_grad():
    for label,path in [('original',root/data['plan']['parent']),
                       ('head',root/'results/diverse-bc-head-lr5e7-20261008/checkpoint.pt'),
                       ('tail',root/'results/diverse-bc-tail-lr5e7-20261008/checkpoint.pt')]:
        saved=torch.load(path,map_location='cpu')
        head=make_action_head(space,data['rows'][0]['features'].shape[-1],**head_kwargs).cuda().eval()
        head.load_state_dict(saved['actor'])
        if label=='tail':load_tail(net,saved)
        for key,rows in groups.items():
            loss=0.
            for i in range(0,len(rows),16):
                chunk=rows[i:i+16];f=torch.cat([r['features'] for r in chunk],1).cuda()
                if label=='tail':
                    with torch.cuda.amp.autocast():latent=torch.cat([tail(r['tail_input'].cuda()) for r in chunk],1)
                    f=torch.cat([latent.float(),f[...,-6:]],-1)
                a={k:torch.cat([r['action'][k] for r in chunk],1).cuda() for k in chunk[0]['action']}
                loss-=head.logprob(a,guided(head,f)).sum().item()
            report[key]['nll'][label]=loss/len(rows)
        del head,saved
write_json(out,dict(rows=list(report.values()),optimizer_updates=0,
                   scope='Teacher-state stage losses; no student-state coverage or task success claim'))
print(json.dumps(list(report.values())))
