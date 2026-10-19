"""Read-only metrics for the local detour development comparison."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def summarize(attempt):
    run=attempt/'run';read=lambda name:json.loads((run/name).read_text(encoding='utf-8-sig'))
    summary=read('summary.json');rows=read('goal-recovery-rows.json')
    scalar=lambda v:scalar(v[0]) if isinstance(v,list) else v
    movement=lambda r:any(scalar(r['action'].get(k,0)) for k in ('forward','back','left','right'))
    distance=lambda r:math.hypot(r['server_after']['server_x']-r['server_before']['server_x'],
                               r['server_after']['server_z']-r['server_before']['server_z'])
    def phase(r):
        if r['owner']=='planner_neutral':return 'neutral'
        if 'local_route' in r:return 'routed_approach'
        if 'waypoint_target' in r:return 'waypoint'
        return r.get('subgoal_context',{}).get('skill_id','collect_wood')
    phases={}
    for key in dict.fromkeys(map(phase,rows)):
        subset=[r for r in rows if phase(r)==key]
        phases[key]=dict(actions=len(subset),movement_commands=sum(movement(r) for r in subset),
            low_motion_commands=sum(movement(r) and distance(r)<.02 for r in subset),
            path_blocks=sum(distance(r) for r in subset),
            wood_known_actions=sum(r['server_after']['wood_v3_kind']>0 for r in subset),
            interaction_ready_actions=sum(r['server_after']['wood_v3_reachable']>0 for r in subset))
    first=lambda k:next((i+1 for i,r in enumerate(rows) if r['server_after'][k]),None)
    result=dict(label=attempt.name,status=summary['status'],execution_valid=summary['execution_valid'],
        elapsed_control_s=summary['elapsed_control_s'],steps=summary['steps'],
        server_target_reached=summary['server_target_reached'],inventory_final_delta=summary['inventory_final_delta'],
        planner_calls=summary['planner_calls'],first_target=first('wood_v3_kind'),first_reachable=first('wood_v3_reachable'),
        phases=phases,stop_reason=summary['stop_reason'],direct_action_overrides=summary['direct_action_overrides'],
        process_cleanup=summary['process_cleanup'],release=read('input-release-state.json'),
        source='Actual committed server observations; low motion is not alone proof of a wall collision')
    if (run/'navigation-receipts.json').exists():
        result['waypoints']=[dict(target=r['method']['arguments']['target'],outcome=r['outcome'],
            diagnosis=r['assessment']['diagnosis'],steps=r['committed_steps']) for r in read('navigation-receipts.json')]
    result['planner_pauses']=[e.get('reason') for e in read('planner-events.json') if e['event']=='paused']
    (attempt/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.attempt),indent=2))
