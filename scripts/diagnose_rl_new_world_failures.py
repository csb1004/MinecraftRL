"""Describe unsuccessful factual trajectories without changing policy or reward."""
import json
from pathlib import Path
from minecraftrl.runtime import write_json

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'results/rl-horizon-new-worlds-20261007'
state=json.loads((base/'experiment.json').read_text())
assert state['completed'] and (ROOT/'results/STOP').exists()
def local(path):
    text=str(path).replace('\\','/')
    if text.startswith('D:/DATA/Desktop/MinecraftRL/'):return ROOT/text.split('MinecraftRL/',1)[1]
    return Path(path)
results=[]
for run in state['runs']:
    if run['report']['server_target_reached']:continue
    folder=local(run['path']);initial=json.loads((folder/'initial-state.json').read_text())['server']
    start=[initial['server_'+axis] for axis in 'xyz'];positions=[];inventory=None;actions={}
    first_drop=None;last_health=None
    with (folder/'events.jsonl').open() as stream:
        for line in stream:
            event=json.loads(line)
            if event['event']!='transition':continue
            server=event['info']['workshop'];point=[server['server_'+axis] for axis in 'xyz'];positions.append(point)
            if first_drop is None and point[1]<=start[1]-5:first_drop=len(positions)
            inventory=server['server_inventory'];last_health=event['observation'].get('life_stats')
            for key in ('forward','back','left','right','jump','attack','use','sprint','sneak'):
                value=event['action'].get(key,0)
                if isinstance(value,list):
                    assert len(value)==1
                    value=value[0]
                assert value in (0,1)
                actions[key]=actions.get(key,0)+int(bool(value))
    assert len(positions)==run['report']['steps']
    tail=positions[len(positions)*3//4:]
    results.append(dict(label=run['label'],condition=run['condition'],repeat=run['repeat'],seed=run['seed'],
        status=run['report']['status'],steps=len(positions),initial_position=start,final_position=positions[-1],
        minimum_y=min(p[1] for p in positions),maximum_y=max(p[1] for p in positions),
        first_step_five_blocks_below_start=first_drop,
        last_quarter_position_ranges=[max(p[i] for p in tail)-min(p[i] for p in tail) for i in range(3)],
        final_inventory={k:v for k,v in inventory.items() if v},final_life_stats=last_health,
        action_active_steps=actions,initial_frame=str(folder/'frames/000000.png'),final_frame=str(folder/'frames/final.png')))
    print('described',run['label'],flush=True)
write_json(base/'failure-diagnosis.json',dict(completed=True,runs=results,environment_actions=0,optimizer_updates=0,
    scope='Observed positions, inventories and actions; descriptive evidence, not proof of a neural failure mechanism.'))
