"""Copy existing wall geometry at a previously observed stall pose; no runtime policy hints."""
import json
from pathlib import Path
from minecraftrl.rl_fixture import make_arena
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];out=root/'results/blind-recovery-20261012/worlds'
out.mkdir(exist_ok=False)
for case in ('far','corner'):
    run=root/'results/waypoint-validation-20261010'/case/'run'
    report=json.loads((run/'summary.json').read_text());world=report['initial_world']
    switches=json.loads((run/'waypoint-switches.json').read_text());step=switches[0]['sequence']
    rows=json.loads((run/'waypoint-rows.json').read_text());g=rows[step-1]['server_after']
    x=g['server_x']-16*world['chunk'][0];z=g['server_z']-16*world['chunk'][1]
    mirror=world['curriculum']['mirror_z'];yaw=g['yaw']
    if mirror:z=16-z;yaw=180-yaw
    yaw=(yaw+180)%360-180
    source=root/'results/diverse-teacher-worlds-20261008'/('development-'+case+'.zip')
    archive=out/(case+'.zip')
    result=make_arena(source,archive,'recovery',wall_end=world['curriculum']['wall_end'],mirror_z=mirror,start_xz=[x,z],start_yaw=yaw)
    result.update(experimental_split='development',pose_source=dict(run=str(run.relative_to(root)),step=step,
        rows_sha256=digest(run/'waypoint-rows.json'),position=[g['server_'+k] for k in 'xyz'],yaw=g['yaw'],pitch=g['pitch']),
        reconstruction='same fixture at archived stall xz/yaw; pitch -13, fresh neural memory; live stall must be observed again')
    write_json(archive.with_suffix('.json'),result)
    print(json.dumps(dict(case=case,position=result['position'],yaw=result['start_pose']['yaw'],sha256=result['sha256'])))
