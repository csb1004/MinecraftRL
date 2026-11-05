"""Independent inventory/tick audit and descriptive S2 development comparison."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def analyze(batch):
    plan=read(batch/'plan.json');runs=[];issues=[]
    for condition in plan['runs']:
        folder=batch/condition['name']
        if not (folder/'summary.json').exists():continue
        summary=read(folder/'summary.json')
        if 'worker_exit_code' not in summary:continue
        if not summary.get('execution_valid') or summary['worker_exit_code']!=0:
            issues.append(dict(run=condition['name'],reason='technical_or_cancelled',status=summary.get('status')));continue
        rows=read(folder/'rm-rows.json');queue=summary['goal_queue']
        if plan.get('protocol')=='common_recovery_r1':
            start_receipt=read(folder/'recovery-start.json')
            assert start_receipt['world_sha256']==condition['world_sha256'], 'restored world differs from plan'
            assert start_receipt['task']==condition['task'] and start_receipt['split']==condition['role']
            assert summary['condition']['sampling_seed']==condition['sampling_seed'], 'policy sampling seed differs'
        goals={g['id']:g for g in queue['program']['goals']}
        initial=read(folder/'initial-state.json')['server'];previous=initial
        seen={};completed=set();transitions=0;ticks=Counter();offsets=Counter();frames=[];steps=[]
        mining_segments=0;mining_length=0;longest_mining=0;last_target=None;attack=0
        last_server=None;dispatch_after_ack=0;acked=False
        event_hash=hashlib.sha256()
        with (folder/'events.jsonl').open('rb') as stream:
            for line in stream:
                event_hash.update(line)
                event=json.loads(line)
                if event['event']=='stop_acknowledged':acked=True
                if event['event']=='policy_dispatch_started' and acked:dispatch_after_ack+=1
                if event['event']=='cycle_timing':frames.append(event['frame_write_ms'])
                if event['event']=='policy_dispatch_finished':steps.append((event['call_finished_s']-event['call_started_s'])*1000)
                if event['event']!='transition':continue
                row=rows[transitions];transitions+=1;server=event['info']['workshop'];last_server=server
                assert event['step']==row['step']==transitions
                assert event['action']==row['env_action'] and event['reward']==row['reward']
                key=row['goal_id'];goal=goals[key]
                assert key not in completed, 'action after goal completion'
                if key not in seen:
                    assert all(dep in completed for dep in goal['depends_on'])
                    seen[key]=sum(previous['server_inventory'].get(i,0) for i in goal['items'])
                    assert seen[key]==queue['records'][key]['baseline']
                gain=max(0,sum(server['server_inventory'].get(i,0) for i in goal['items'])-seen[key])
                reached=gain>=goal['count']
                assert event['reward']==float(reached), 'reward does not match authoritative new inventory'
                if reached:completed.add(key)
                ticks[(server['server_tick']-previous['server_tick'],server['client_tick']-previous['client_tick'])]+=1
                offsets[server['server_tick']-server['client_tick']]+=1
                target=tuple(server.get('wood_mining_'+axis) for axis in 'xyz') if server.get('wood_server_mining') else None
                if target is None:mining_length=0
                elif target!=last_target:mining_segments+=1;mining_length=1
                else:mining_length+=1
                longest_mining=max(longest_mining,mining_length);last_target=target
                button=event['action'].get('attack',0)
                while isinstance(button,list):button=button[0]
                attack+=bool(button)
                previous=server
        assert transitions==summary['steps']==len(rows)
        assert len(completed)==summary['completed_goal_count']
        assert not dispatch_after_ack
        assert summary['release']=='neutral_command_sent' and not summary['process_cleanup']['remaining_live_pids']
        release=read(folder/'input-release-state.json')
        assert release['verified'] and release['held_key_count']==0, 'held input after release'
        variant=condition['variant'] if condition['candidate'] else 'initial'
        per_goal=[dict(id=key,status=record['status'],reason=record['reason'],executed=key in seen,
                      gain=record['progress'],count=goals[key]['count']) for key,record in queue['records'].items()]
        from statistics import median
        item=dict(name=condition['name'],seed=condition['seed'],role=condition['role'],variant=variant,
            case_id=condition.get('case_id',str(condition['seed'])),task=condition.get('task','collection'),
            status=summary['status'],end_reason=summary.get('s2_end_reason'),
            program_completed=summary['program_completed'],completed_goals=len(completed),executed_goals=len(seen),
            unexecuted_goals=len(goals)-len(seen),steps=transitions,control_s=summary['elapsed_control_s'],
            inference_ms=summary['inference_ms'],cycle_ms=summary['control_cycle_ms'],gpu=summary['gpu'],
            device_vram=summary.get('device_vram'),phase_counts=dict(Counter(r['stage'] for r in rows)),
            inventory_delta=summary['inventory_final_delta'],goals=per_goal,attack_actions=attack,
            mining_segments=mining_segments,longest_mining_segment=longest_mining,
            server_log_harvest_delta=last_server.get('server_log_harvests',0)-initial.get('server_log_harvests',0),
            tick_deltas={str(k):v for k,v in ticks.items()},tick_offsets=dict(offsets),
            frame_write_median_ms=median(frames) if frames else None,env_step_median_ms=median(steps) if steps else None,
            initial_pose={k:initial.get(k) for k in ('server_x','server_y','server_z','yaw','pitch','server_tick','world_time')},
            events_sha256=event_hash.hexdigest(),
            summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest())
        item['phase_by_skill']={skill:dict(Counter(r['stage'] for r in rows if r['skill']==skill))
                                for skill in sorted({r['skill'] for r in rows})}
        item['progress_fractions']=sorted({r['phase_context'][5] for r in rows})
        item['goal_quantities']=sorted({r['semantic_goal']['quantity'] for r in rows})
        item['final_pose']={k:last_server.get(k) for k in ('server_x','server_y','server_z','yaw','pitch','server_tick','world_time')}
        runtime=read(folder/'runtime-jar-cache.json')
        item['runtime_jar_sha256']=runtime['sha256']
        item['launcher_sha256']=runtime['launcher_sha256']
        runs.append(item)
    aggregate={}
    for role,variant in sorted({(r['role'],r['variant']) for r in runs}):
        group=[r for r in runs if (r['role'],r['variant'])==(role,variant)]
        aggregate[role+'/'+variant]=dict(runs=len(group),completed=sum(r['program_completed'] for r in group),
            completed_goals=sum(r['completed_goals'] for r in group),executed_goals=sum(r['executed_goals'] for r in group),
            unexecuted_goals=sum(r['unexecuted_goals'] for r in group),steps=sum(r['steps'] for r in group),
            endings=dict(Counter(r['end_reason'] or r['status'] for r in group)))
    pairs=[]
    try:
        import numpy as np
        from PIL import Image
        for base in [r for r in runs if r['role']=='development' and r['variant']=='initial']:
            with Image.open(batch/base['name']/'frames/000000.png') as im: left=np.asarray(im.convert('RGB'))
            for other in [r for r in runs if r['role']=='development' and r['case_id']==base['case_id'] and r['variant']!='initial']:
                with Image.open(batch/other['name']/'frames/000000.png') as im:right=np.asarray(im.convert('RGB'))
                changed=np.any(left!=right,axis=-1)
                pairs.append(dict(seed=base['seed'],case_id=base['case_id'],task=base['task'],variant=other['variant'],initial_pose_equal=base['initial_pose']==other['initial_pose'],
                    initial_pixels_equal=not bool(changed.any()),changed_pixel_fraction=float(changed.mean())))
    except ImportError:issues.append(dict(reason='pixel_comparison_dependency_missing'))
    runtime_hashes=sorted({r['runtime_jar_sha256'] for r in runs})
    launcher_hashes=sorted({r['launcher_sha256'] for r in runs})
    assert len(runtime_hashes)<=1 and len(launcher_hashes)<=1, 'runtime changed between conditions'
    result=dict(protocol=plan.get('protocol','S2'),completed_valid=len(runs),planned=len(plan['runs']),runs=runs,aggregate=aggregate,
                runtime_jar_sha256=runtime_hashes,launcher_sha256=launcher_hashes,
                initial_state_pairs=pairs,issues=issues,claim='descriptive_development_only',automatic_promotion=False)
    (batch/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (batch/'run-metrics.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['name','seed','role','variant','status','end_reason','program_completed','steps','control_s',
                'inference_median_ms','inference_p95_ms','cycle_median_ms','cycle_p95_ms','torch_peak_mib','device_peak_mib']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in runs:
            row={k:r[k] for k in fields[:9]}
            for label,source in [('inference',r['inference_ms']),('cycle',r['cycle_ms'])]:
                for stat in ('median','p95'):row[label+'_'+stat+'_ms']=source[stat]
            row['torch_peak_mib']=r['gpu']['torch_peak_allocated_mib']
            row['device_peak_mib']=(r['device_vram'] or {}).get('sampled_peak_used_mib')
            writer.writerow(row)
    print(json.dumps(dict(completed_valid=len(runs),aggregate=aggregate,issues=issues),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    analyze(parser.parse_args().batch)
