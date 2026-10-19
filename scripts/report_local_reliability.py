"""Summarize real committed success, separately from after-release saved inventory."""
import argparse
from collections import Counter
import json
from pathlib import Path


def report(attempt):
    read=lambda name:json.loads((attempt/name).read_text(encoding='utf-8-sig'))
    summary=read('run/summary.json');rows=read('run/goal-recovery-rows.json')
    logs=lambda obs:sum(v for k,v in obs['server_inventory'].items() if k.endswith('_log'))
    start=rows[0]['server_before'];end=rows[-1]['server_after']
    baseline=logs(start);gain=logs(end)-baseline
    events=[json.loads(s) for s in (attempt/'run/events.jsonl').read_text(encoding='utf-8').splitlines()]
    initial=next(e['monotonic_s'] for e in events if e['event']=='episode_started')
    success_events=[e for e in events if e['event']=='target_reached']
    first=lambda predicate:next((i+1 for i,r in enumerate(rows) if predicate(r)),None)
    stop_index=next(i for i,e in enumerate(events) if e['event']=='stop_acknowledged')
    post_stop=sum(e['event'] in ('inference_started','policy_dispatch_started','control_dispatch_started') for e in events[stop_index+1:])
    normal=(summary['status']=='target_reached' and summary['execution_valid']
        and summary['worker_exit_code']==0 and summary['server_target_reached'] and gain>=1
        and summary['elapsed_control_s']<=180 and len(success_events)==1)
    saved=None
    if (attempt/'world-preservation.json').exists():
        inventories=read('world-preservation.json')['saved_inventory']
        saved={k:sum(x['Count'] for x in v if x['id'].endswith('_log')) for k,v in inventories.items()}
    result=dict(label=attempt.name,status=summary['status'],execution_valid=summary['execution_valid'],
        worker_exit_code=summary['worker_exit_code'],normal_episode_success=normal,
        elapsed_control_s=summary['elapsed_control_s'],steps=len(rows),server_log_gain=gain,
        inventory_gain_step=first(lambda r:logs(r['server_after'])>baseline),
        success_elapsed_s=success_events[0]['monotonic_s']-initial if success_events else None,
        first_discovery_step=first(lambda r:r['server_after']['wood_v3_kind']>0),
        first_reachable_step=first(lambda r:r['server_after']['wood_v3_reachable']>0),
        initial_known=bool(start['wood_v3_kind']),planner_calls=summary['planner_calls'],
        phases=dict(Counter(r.get('local_route',{}).get('stage',
            'waypoint' if 'waypoint_target' in r else r.get('subgoal_context',{}).get('phase','unknown')) for r in rows)),
        tempered_neural_actions=sum(r.get('action_temperature',1)<1 for r in rows),
        saved_log_counts=saved,post_stop_actions=post_stop,
        held_keys_after_release=read('run/input-release-state.json')['held_key_count'],
        remaining_owned_pids=summary['process_cleanup']['remaining_live_pids'],
        planner_pauses=[e.get('reason') for e in read('run/planner-events.json') if e['event']=='paused'])
    assert post_stop==0 and result['held_keys_after_release']==0 and not result['remaining_owned_pids']
    (attempt/'reliability.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);a=p.parse_args()
    print(json.dumps(report(a.attempt),indent=2))
