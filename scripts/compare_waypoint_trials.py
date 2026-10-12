"""Summarize audited trials without crediting an uncalled skill or zero-action arrival."""
import json, math
from pathlib import Path
import numpy as np
import torch
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1]
out=root/'results/waypoint-comparison-20261010/comparison.json'
if out.exists():raise ValueError('comparison already exists')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
trials={};sources={}
for condition,folder in (('initial','waypoint-validation-20261010'),('focused','waypoint-focused-validation-20261010'),
        ('focused_at_start','waypoint-initial-validation-20261010')):
    for case in ('far','corner'):
        run=root/'results'/folder/case/'run'
        report=read(run/'summary.json');audit=read(run/'waypoint-audit.json')
        assert audit['verified']
        rows=read(run/'waypoint-rows.json');nav=[r for r in rows if r['mode']=='waypoint']
        forward=[r for r in nav if np.asarray(r['action'].get('forward',0)).item()]
        misaligned=sum(abs(math.degrees(math.atan2(r['goal'][0],r['goal'][1])))>45 for r in forward)
        navigation_path=sum(math.hypot(r['server_after']['server_x']-r['server_before']['server_x'],
            r['server_after']['server_z']-r['server_before']['server_z']) for r in nav)
        result=dict(steps=len(rows),mode_steps=report['mode_steps'],status=report['status'],
            start_mode=report['condition'].get('waypoint_start','stall'),
            skill_evaluated=bool(nav),route_started=report['route_started'],route_completed=report['route_completed'],
            waypoints_reached=sum(r['actions']>0 for r in report['route_receipts']),
            already_satisfied=sum(r['actions']==0 for r in report['route_receipts']),
            inventory_delta=report['inventory_final_delta'],root_success=report['root_inventory_success'],
            forward_actions=len(forward),forward_heading_error_gt45=misaligned,
            navigation_path_blocks=navigation_path,
            navigation_attack_actions=sum(bool(np.asarray(r['action']['attack']).item()) for r in nav),
            navigation_jump_actions=sum(bool(np.asarray(r['action']['jump']).item()) for r in nav),
            goals=audit['goals'],elapsed_control_s=report['elapsed_control_s'],
            inference_ms=report['inference_ms'],control_cycle_ms=report['control_cycle_ms'],
            gpu=report['gpu'],device_vram=report['device_vram'])
        trials[condition+'_'+case]=result
        for name in ('summary.json','waypoint-rows.json','waypoint-audit.json'):
            sources[str((run/name).relative_to(root))]=digest(run/name)

prefixes={}
for case in ('far','corner'):
    runs=[root/'results'/folder/case/'run' for folder in ('waypoint-validation-20261010','waypoint-focused-validation-20261010')]
    rows=[read(run/'waypoint-rows.json') for run in runs]
    limit=min(sum(r['mode']=='root' for r in group) for group in rows)
    first_frame=first_action=first_pose=None
    for i in range(limit):
        pair=[group[i] for group in rows]
        if first_frame is None and digest(runs[0]/pair[0]['frame'])!=digest(runs[1]/pair[1]['frame']):first_frame=i+1
        if first_action is None and pair[0]['action']!=pair[1]['action']:first_action=i+1
        if first_pose is None and any(pair[0]['server_before'][k]!=pair[1]['server_before'][k] for k in ('server_x','server_y','server_z','yaw','pitch')):first_pose=i+1
    prefixes[case]=dict(compared_root_actions=limit,first_frame_difference=first_frame,
        first_action_difference=first_action,first_before_pose_difference=first_pose,
        identical_prefix=not any((first_frame,first_action,first_pose)),
        initial_frame_sha256=[digest(run/'frames/000000.png') for run in runs])

torch.set_num_threads(1)
paths=[root/'results'/name/'checkpoint.pt' for name in ('waypoint-training-20261010','waypoint-training-goal-focused-20261010')]
weights=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
visual={};goal={}
for name,tensor in weights[0]['actor'].items():
    after=weights[1]['actor'][name]
    if '.visual.' in name:
        error=(after-tensor*.1).abs().max().item();assert error==0
        visual[name]=error
    elif '.goal.' in name:goal[name]=(after-tensor).abs().max().item()
    else:assert torch.equal(after,tensor),name
assert visual and goal and any(goal.values())
assert weights[0]['parent_sha256']==weights[1]['parent_sha256']
for p in paths:sources[str(p.relative_to(root))]=digest(p)
write_json(out,dict(trials=trials,root_prefix_comparison=prefixes,
    focused_visual_scale_max_errors=visual,focused_goal_max_changes=goal,source_sha256=sources,
    api_calls=0,scope='Development diagnostics; uncalled waypoint skill excluded from skill success/failure denominator.'))
print(json.dumps(dict(trials=trials,root_prefix_comparison=prefixes)))
