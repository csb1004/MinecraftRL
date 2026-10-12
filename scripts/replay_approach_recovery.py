"""Offline detector diagnosis. Never writes/labels counterfactual training data."""
import json
from pathlib import Path
from minecraftrl.rl_core import ShapedWoodRewardV3
from minecraftrl.approach_recovery import ApproachRecovery
from minecraftrl.runtime import write_json

root=Path(__file__).resolve().parents[1]
cases=[('legacy-failure','rl-eval-2100059-20261007T052714309530Z',True),
       ('normal-v2','rl-eval-104729-20261007T063210789328Z',False),
       ('normal-v3','rl-eval-104729-20261007T063602634490Z',False)]
results=[]
for label,path,legacy in cases:
    detector=ApproachRecovery();events=[];active=0;count=0
    for line in (root/'results'/path/'run/events.jsonl').open():
        e=json.loads(line)
        if e['event']!='transition':continue
        s=e['info']['workshop'];count+=1
        if legacy:
            s=dict(s,**{'wood_v3_'+k:s['wood_'+k] for k in ('kind','distance','reachable','target_x','target_y','target_z')})
        stage=ShapedWoodRewardV3.state(s)[0];r=detector.update(s,stage,terminal=e['done'])
        if r['event']:events.append(dict(step=count,**r))
        active+=int(r['active'])
    results.append(dict(label=label,path=path,legacy_sensor_substitution=legacy,steps=count,events=events,active_snapshots=active))
write_json(root/'results/approach-recovery-replay.json',dict(scope='offline counterfactual detector only; zero actions/training',cases=results))
print(json.dumps(results))
