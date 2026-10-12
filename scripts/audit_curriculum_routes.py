"""Actual wall crossing and recovery acquisition, without policy intervention."""
import argparse,json
from pathlib import Path
from minecraftrl.runtime import write_json

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,default=root/'results/recovery-curriculum-20261007');base=p.parse_args().batch
state=json.loads((base/'experiment.json').read_text());rows=[]
for record in state['runs']:
    assert record['audited']
    run=Path(record['path']);world=json.loads(Path(record['world']['path']).with_suffix('.json').read_text())
    ox,oz=[16*v for v in world['chunk']];crossing=None;step=0
    previous=json.loads((run/'initial-state.json').read_text())['server']
    mirrored=world['curriculum']['mirror_z'];wall_end=world['curriculum']['wall_end']
    attack_steps=wall_attack_steps=gap_clear_steps=0;positions=[]
    for line in (run/'events.jsonl').open():
        e=json.loads(line)
        if e['event']!='transition':continue
        step+=1;s=e['info']['workshop']
        # Reflect continuous coordinates about z=8; block indices reflect about 7.5.
        x=s['server_x']-ox;z=s['server_z']-oz
        normalized_z=16-z if mirrored else z
        positions.append((x,normalized_z))
        gap_clear_steps+=int(normalized_z>=wall_end+1.3)
        attack=bool(e['action']['attack'][0]);attack_steps+=int(attack)
        hit_z=previous['hit_z']-oz
        normalized_hit_z=15-hit_z if mirrored else hit_z
        wall_attack_steps+=int(attack and previous['hit_x']==ox+7 and
            previous['hit_y'] in (201,202) and 1<=normalized_hit_z<=wall_end)
        if crossing is None and s['server_x']>=ox+8.3:
            crossing=dict(step=step,relative_position=[s['server_x']-ox,s['server_y'],s['server_z']-oz])
        previous=s
    audit=json.loads((run/'recovery-audit.json').read_text())
    rows.append(dict(label=record['label'],split=record['world']['split'],stage=record['world'].get('stage',world.get('curriculum_stage')),
                     world_sha256=record['world']['sha256'],steps=step,first_crossing=crossing,
                     entered=audit['entered'],acquisition_after_entry=audit['acquisition_after_entry'],
                     success=audit['success'],server_inventory_delta=audit['server_inventory_delta'],
                     events=audit['events'],attack_steps=attack_steps,wall_attack_steps=wall_attack_steps,
                     gap_clear_steps=gap_clear_steps,
                     max_relative_x=max(p[0] for p in positions),
                     max_gap_direction_z=max(p[1] for p in positions),
                     gap_clearance_threshold=wall_end+1.3))
write_json(base/'route-audit.json',dict(rows=rows,scope='actual crossing and inventory; no scripted action credit'))
print(json.dumps([{k:v for k,v in row.items() if k not in ('events','world_sha256')} for row in rows]))
