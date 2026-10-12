"""Audit actual packet/attempt persistence plus the existing full transition audit."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from audit_fresh_target_handoff import audit
from minecraftrl.replanning import PlanHistory,ReplanGuard,make_snapshot
from minecraftrl.recovery_draft import planning_context,validate_packet
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def audit_memory(run,checkpoint):
    run=run.resolve();destination=run/'recovery-memory-audit.json'
    if destination.exists():raise FileExistsError(destination)
    audit(run,checkpoint)
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    previous=read(run/'fresh-target-handoff-audit.json')
    rows=read(run/'goal-recovery-rows.json');history=PlanHistory(run/'planner-history.json').snapshot()
    events=read(run/'planner-events.json');attempts=read(run/'recovery-attempts.json')
    assert history['schema']==2 and len(history['attempts'])==len(attempts)
    if history['requests']:
        packet=read(run/'planner-input.json');validate_packet(packet)
        sent=next(e['packet'] for e in events if e['event']=='wait_started')
        assert packet==sent
        request=history['requests'][0];memory=PlanHistory()
        memory.data.update(revision=packet['planner_context']['history_revision'],
            requests=[dict(deepcopy(request),consumed=False)],
            failures=[deepcopy(r) for r in history['failures']
                      if r['snapshot']['sequence']<=request['snapshot']['sequence']])
        guard=ReplanGuard(memory,request['capabilities'],allow_goal_conditioning=True)
        assert planning_context(guard,request['request_id'])==packet['planner_context']
        assert packet['planner_context']['failed_attempts']
        assert packet['planner_context']['available_choices']==['explore_then_resume','remain_paused']
    for saved,receipt in zip(history['attempts'],attempts):
        assert saved['request_id']==receipt['request_id']
        assert saved['committed_steps']==receipt['committed_transitions']
        assert saved['before']['sequence']==receipt['start_step']-1
        if saved['after'] is not None:assert saved['after']['sequence']==receipt['end_step']
        for snapshot in (saved['before'],saved['after']):
            if snapshot is None:continue
            observation=rows[snapshot['sequence']-1]['server_after']
            rebuilt=make_snapshot(observation,**{k:snapshot[k] for k in (
                'world_id','episode_id','goal','task','generation','sequence','elapsed_s','status',
                'release_sequence','held_key_count','baseline_count')})
            assert rebuilt==snapshot
        classification=receipt['assessment']['classification']
        expected=('completed' if classification=='goal_completed' else
                  'interrupted' if classification in ('stopped','control_budget_exhausted') else
                  'unverified' if classification=='observation_unreliable' else
                  'partial' if classification in ('target_acquired','interaction_ready',
                     'recovery_incomplete_with_progress','recovery_incomplete_with_target_evidence') else 'failed')
        assert saved['outcome']==expected and saved['reason']==classification
    result=dict(previous,history_schema=2,memory_packet_verified=bool(history['requests']),
        execution_receipts=len(history['attempts']),outcomes=[r['outcome'] for r in history['attempts']],
        memory_source_sha256={n:digest(run/n) for n in ('planner-history.json','planner-events.json',
            'goal-recovery-rows.json','recovery-attempts.json')})
    if history['requests']:result['memory_source_sha256']['planner-input.json']=digest(run/'planner-input.json')
    write_json(destination,result)
    print(json.dumps({k:result[k] for k in ('verified','steps','log_gain','memory_packet_verified',
        'execution_receipts','outcomes','post_stop_dispatches','held_keys_after_release','remaining_owned_pids')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('checkpoint',type=Path)
    args=parser.parse_args();audit_memory(args.run,args.checkpoint)
