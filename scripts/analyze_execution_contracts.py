"""Read-only E2 raw server/contract audit with descriptive development metrics."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from minecraftrl.execution_returns import check_committed_evidence,validate_execution_trace,REWARD_PROFILE
from minecraftrl.goal_queue import GoalQueue
from minecraftrl.rm_program import CAPABILITIES


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def analyze(batch):
    plan=read(batch/'plan.json');runs=[];issues=[]
    for case in plan['runs']:
        folder=batch/case['name']
        if not (folder/'summary.json').exists():continue
        summary=read(folder/'summary.json')
        if 'worker_exit_code' not in summary:continue
        if not summary.get('execution_valid') or summary['worker_exit_code']!=0:
            issues.append(dict(name=case['name'],reason='technical_or_cancelled',status=summary.get('status')));continue
        receipt=read(folder/'recovery-start.json');rows=read(folder/'rm-rows.json')
        assert receipt['world_sha256']==case['world_sha256']
        assert receipt['task']==case['task'] and receipt['split']==case['role']
        assert summary['condition']['sampling_seed']==case['sampling_seed']
        initial=read(folder/'initial-state.json')['server'];previous=initial
        index=0;acked=False;poststop=0;event_hash=hashlib.sha256()
        with (folder/'events.jsonl').open('rb') as stream:
            for line in stream:
                event_hash.update(line);event=json.loads(line)
                if event['event']=='stop_acknowledged':acked=True
                if event['event']=='policy_dispatch_started' and acked:poststop+=1
                if event['event']!='transition':continue
                row=rows[index];index+=1
                check_committed_evidence(row,previous,event)
                previous=event['info']['workshop']
                row['phase']=row['stage']
        assert index==len(rows)==summary['steps'] and poststop==0
        assert summary['release']=='neutral_command_sent' and not summary['process_cleanup']['remaining_live_pids']
        release=read(folder/'input-release-state.json')
        assert release['verified'] and release['held_key_count']==0
        compiled=summary['goal_queue']['goal_spec']
        if rows:
            validate_execution_trace(dict(rows=rows,goal_spec=compiled,program=compiled['program'],
                program_sha256=compiled['program_sha256'],variant='nonlinear-on',
                context_fields=summary['condition']['context_fields'],bank_metadata=summary['condition']['bank'],
                recovery_task=case['task'],reward_profile=REWARD_PROFILE))
            assert rows[-1]['execution_after']['completed']==summary['program_completed']
        else:
            q=GoalQueue(compiled['source'],CAPABILITIES);q.observe(initial);q.select()
            assert q.completed and summary.get('completed_on_reset')
        events=[json.loads(line) for line in (folder/'execution-events.jsonl').read_text().splitlines()] if rows else []
        assert len(events)==len(rows)
        assert sum(r['program_reward'] for r in rows)==summary['program_reward']
        for row,event in zip(rows,events):
            assert event['request']==row['execution_request'] and event['outcome']==row['execution_after']
        records=summary['goal_queue']['records'];executed={r['goal_id'] for r in rows}
        runs.append(dict(name=case['name'],case_id=case['case_id'],seed=case['seed'],role=case['role'],
            data_role=case['data_role'],candidate=case['candidate'],model=case['model'],task=case['task'],
            status=summary['status'],end_reason=summary.get('s2_end_reason'),
            completed=summary['program_completed'],steps=index,control_s=summary['elapsed_control_s'],
            inference_ms=summary['inference_ms'],cycle_ms=summary['control_cycle_ms'],gpu=summary['gpu'],
            device_vram=summary.get('device_vram'),inventory_delta=summary['inventory_final_delta'],
            goals=[dict(id=k,**r,executed=k in executed) for k,r in records.items()],
            failure_category=summary.get('failure_category'),phase_counts=dict(Counter(r['stage'] for r in rows)),
            initial_pose={k:initial.get(k) for k in ('server_x','server_y','server_z','yaw','pitch','server_tick','world_time')},
            events_sha256=event_hash.hexdigest(),runtime_jar_sha256=read(folder/'runtime-jar-cache.json')['sha256'],
            raw_mismatches=0,post_stop_dispatches=0))
    aggregate={}
    for role,model in sorted({(r['role'],r['model']) for r in runs}):
        group=[r for r in runs if (r['role'],r['model'])==(role,model)]
        aggregate[role+'/'+model]=dict(runs=len(group),completed=sum(r['completed'] for r in group),
            steps=sum(r['steps'] for r in group),constraint_violations=sum(r['failure_category']=='constraint_violation' for r in group))
    assert len({r['runtime_jar_sha256'] for r in runs})<=1
    result=dict(protocol=plan['protocol'],runs=runs,aggregate=aggregate,issues=issues,
        verified_rows=sum(r['steps'] for r in runs),claim='descriptive development; not sealed final evaluation',promoted=False)
    (batch/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verified_rows=result['verified_rows'],aggregate=aggregate,issues=issues),indent=2))
    if issues:raise ValueError('invalid runs remain outside performance aggregation')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    analyze(parser.parse_args().batch)
