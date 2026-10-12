"""One-use motion, dispatch and checkpoint audit for the original/local comparison."""
import json
from pathlib import Path
import torch
from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];base=root/'results/base-policy-comparison-20261008'
out=base/'motion-and-weights.json'
if out.exists():raise FileExistsError(out)
state=json.loads((base/'experiment.json').read_text());assert state['completed'] and not state['errors']
def local(path):return root/str(path).replace('\\','/').split('/MinecraftRL/',1)[-1]
rows=[]
for trial in state['runs']:
    run=local(trial['path']);checkpoint=local(state['parent' if trial['label'].endswith('-parent') else 'candidate'])
    assert trial['audited'] and digest(checkpoint)==trial['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu');data=torch.load(run/'trajectory.pt',map_location='cpu')
    assert data['split']=='eval'
    for name in ('actor','value'):
        assert state_fingerprint(saved[name])==state_fingerprint(data['initial_heads'][name])
    if trial['label'].endswith('-parent'):
        for tensor in saved['actor'].values():
            if tensor.ndim==2:assert torch.count_nonzero(tensor[:,-6:])==0
    del saved,data
    previous=json.loads((run/'initial-state.json').read_text())['server']
    initial=previous;distance=turn=0.;moved=rotated=count=0;minimum_y=maximum_y=previous['server_y']
    keys={k:0 for k in ('forward','back','left','right','jump','sprint','sneak','attack','use','inventory')}
    for line in (run/'events.jsonl').open():
        event=json.loads(line)
        if event['event']!='transition':continue
        current=event['info']['workshop'];count+=1
        for key in keys:keys[key]+=int(bool(event['action'].get(key,[0])[0]))
        step=sum((current['server_'+axis]-previous['server_'+axis])**2 for axis in 'xz')**.5
        yaw=abs((current['yaw']-previous['yaw']+180)%360-180)
        pitch=abs(current['pitch']-previous['pitch'])
        distance+=step;turn+=yaw+pitch;moved+=step>1e-5;rotated+=yaw+pitch>1e-5
        minimum_y=min(minimum_y,current['server_y']);maximum_y=max(maximum_y,current['server_y'])
        previous=current
    assert count==trial['report']['steps']
    rows.append(dict(label=trial['label'],steps=count,checkpoint_verified=True,commands=keys,
                     horizontal_path_blocks=distance,moving_transitions=moved,turning_transitions=rotated,
                     absolute_yaw_pitch_change_degrees=turn,minimum_y=minimum_y,maximum_y=maximum_y,
                     start=[initial['server_'+a] for a in 'xyz'],end=[previous['server_'+a] for a in 'xyz']))
write_json(out,dict(verified=True,rows=rows,
                   scope='Actual commands and server displacement; requested jump is not proof of a completed jump skill'))
print(json.dumps(rows))
