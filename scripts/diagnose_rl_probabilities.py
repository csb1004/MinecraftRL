"""Read-only same-shape versus batched behavior-probability diagnosis; no optimizer."""
import argparse,json,sys
from pathlib import Path
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import guided
from minecraftrl.runtime import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('batch',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.action_head import make_action_head
    state=json.loads((a.batch/'batch.json').read_text());episodes=[]
    for run in state['runs']:
        if not run['label'].startswith('train-'):continue
        relative=run['path'].replace('\\','/').split('MinecraftRL/',1)[1]
        episodes.append(torch.load(root/relative/'trajectory.pt',map_location='cpu'))
    rows=[r for e in episodes for r in e['rows']]
    actor=make_action_head(DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update()),rows[0]['features'].shape[-1],
                          **model_kwargs(root/'weights/steve1/2x.model')[1]).cuda()
    actor.load_state_dict(episodes[0]['initial_heads']['actor']);actor.eval()
    old=torch.tensor([r['old_logp'] for r in rows],device='cuda');values={}
    with torch.no_grad():
        for size in (1,16):
            actual=[]
            for start in range(0,len(rows),size):
                selected=rows[start:start+size];features=torch.cat([r['features'] for r in selected],1).cuda()
                actions={k:torch.cat([r['action'][k] for r in selected],1).cuda() for k in selected[0]['action']}
                actual.append(actor.logprob(actions,guided(actor,features)).flatten())
            values[size]=torch.cat(actual)
    single=(values[1]-old).abs();batch=(values[16]-old).abs();worst=int(batch.argmax().item())
    report=dict(samples=len(rows),updates=0,same_shape_max_error=single.max().item(),batch16_max_error=batch.max().item(),
                batch_vs_same_shape_max_error=(values[16]-values[1]).abs().max().item(),
                worst_index=worst,old=old[worst].item(),same_shape=values[1][worst].item(),batch16=values[16][worst].item())
    write_json(a.batch/'probability-diagnosis.json',report);print(json.dumps(report))

if __name__=='__main__':main()
