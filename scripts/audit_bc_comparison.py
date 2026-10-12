"""Summarize actual BC comparison paths, inventory and stop audits."""
import argparse,json
from pathlib import Path
from minecraftrl.runtime import write_json

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,default=root/'results/bc-comparison-20261008')
p.add_argument('--report-initial-mismatch',action='store_true')
args=p.parse_args();base=args.batch
if (base/'audit.json').exists():raise FileExistsError(base/'audit.json')
state=json.loads((base/'experiment.json').read_text())
if args.report_initial_mismatch:
    assert not state['completed'] and state['errors']==["AssertionError('initial state/pixel mismatch')"]
    assert len(state['runs'])==4 and all(r['audited'] for r in state['runs'])
else:assert state['completed'] and not state['errors']
rows=[]
for run in state['runs']:
    assert run['audited'];path=Path(run['path']);summary=run['report']
    shaping=json.loads((path/'shaping-audit.json').read_text())
    assert shaping['verified'] and shaping['no_policy_action_after_stop']
    assert run['release_evidence']['verified'] and run['release_evidence']['held_key_count']==0
    previous=json.loads((path/'initial-state.json').read_text())['server']
    count=attacks=wall_attacks=boundary_attacks=0;crossing=None;max_gap_z=min_gap_z=None
    world=json.loads(Path(run['world']['path']).with_suffix('.json').read_text())
    if run['world']['kind']=='long_wall':
        ox,oz=[16*x for x in world['chunk']];end=world['curriculum']['wall_end'];mirror=world['curriculum']['mirror_z']
    for line in (path/'events.jsonl').open():
        event=json.loads(line)
        if event['event']!='transition':continue
        count+=1;server=event['info']['workshop'];attack=bool(event['action']['attack'][0]);attacks+=attack
        if run['world']['kind']=='long_wall':
            z=server['server_z']-oz;z=16-z if mirror else z
            max_gap_z=z if max_gap_z is None else max(max_gap_z,z)
            min_gap_z=z if min_gap_z is None else min(min_gap_z,z)
            hz=previous['hit_z']-oz;hz=15-hz if mirror else hz
            wall_attacks+=int(attack and previous['hit_x']==ox+7 and previous['hit_y'] in (201,202) and 1<=hz<=end)
            boundary_attacks+=int(attack and previous['hit_y'] in range(201,205) and
                (previous['hit_x']-ox in (0,15) or previous['hit_z']-oz in (0,15)))
            if crossing is None and server['server_x']>=ox+8.3:crossing=count
        previous=server
    assert count==summary['steps']
    rows.append(dict(label=run['label'],steps=count,success=summary['server_target_reached'],
        inventory_delta=summary['inventory_final_delta'],status=summary['status'],elapsed_s=summary['elapsed_control_s'],
        inference_ms=summary['inference_ms'],cycle_ms=summary['control_cycle_ms'],device_vram=summary['device_vram'],
        attacks=attacks,wall_attacks=wall_attacks if run['world']['kind']=='long_wall' else None,
        boundary_attacks=boundary_attacks if run['world']['kind']=='long_wall' else None,
        first_crossing=crossing,max_gap_z=max_gap_z,min_gap_z=min_gap_z,stages=shaping['stages'],input_released=True,
        no_dispatch_after_stop=True,remaining_owned_pids=summary['process_cleanup']['remaining_live_pids']))
report=dict(rows=rows,comparisons=state['comparisons'],optimizer_updates=0,
            experiment_completed=state['completed'],experiment_errors=state['errors'],
            comparison_valid={k:v['state_identical'] and v['pixel_identical'] for k,v in state['comparisons'].items()},
            scope='Actual server inventory/path evidence; two development worlds, one trial per policy per world')
write_json(base/'audit.json',report);print(json.dumps(rows))
