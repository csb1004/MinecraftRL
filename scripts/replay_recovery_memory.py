"""Replay recorded observations through attempt memory; no game or model calls."""
import json
from pathlib import Path

from minecraftrl.replanning import PlanHistory,ReplanGuard,make_snapshot
from minecraftrl.recovery_draft import planning_context,validate_packet
from minecraftrl.recovery_evidence import assess_stall
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root=Path(__file__).resolve().parents[1]
out=root/'results/recovery-memory-20261010/replay';out.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
results=[]
for stamp in ('20261009T144548946737Z','20261009T150414577383Z'):
    run=root/f'results/goal-recovery-eval-104761-{stamp}/run'
    rows=read(run/'goal-recovery-rows.json');events=read(run/'planner-events.json')
    old=read(run/'planner-history.json');request=old['requests'][0]
    original=request['snapshot'];folder=out/stamp;folder.mkdir()
    names=('goal-recovery-rows.json','planner-events.json','planner-history.json','recovery-attempts.json')
    sources={n:digest(run/n) for n in names}

    def snap(sequence,elapsed):
        obs=rows[sequence-1]['server_after']
        # Only the original two neutral transitions establish release. Later observations
        # stay unverified for dispatch; this script cannot fabricate a new neutral step.
        return make_snapshot(obs,world_id=original['world_id'],episode_id=original['episode_id'],
            goal='wood',task=None,generation=original['generation'],sequence=sequence,elapsed_s=elapsed,
            status='stalled',release_sequence=original['release_sequence']+(sequence>original['sequence']),
            held_key_count=obs.get('held_key_count'),baseline_count=original['baseline_count'])

    guard=ReplanGuard(PlanHistory(folder/'history.json'),request['capabilities'],allow_goal_conditioning=True)
    initial=snap(original['sequence'],original['elapsed_s'])
    guard.record_failure(initial,request['evidence'],'collect_wood')
    req=guard.request(initial,request['evidence'])
    memory=planning_context(guard,req['request_id'])
    old_packet=next(e['packet'] for e in events if e['event']=='wait_started')
    packet=dict(old_packet,request_id=req['request_id'],runtime_binding=initial,planner_context=memory)
    validate_packet(packet);write_json(folder/'first-planner-input.json',packet)
    reviewed=next(e for e in events if e['event']=='reviewed')
    fresh=snap(reviewed['current']['sequence'],reviewed['current']['elapsed_s'])
    proposal=dict(reviewed['proposal'],request_id=req['request_id'])
    verdict=guard.review(req['request_id'],proposal,fresh,now_elapsed_s=reviewed['elapsed_s'])
    assert verdict['accepted']
    attempt=read(run/'recovery-attempts.json')[0]
    assessed=next(e for e in events if e['event']=='recovery_assessed')
    before=snap(attempt['start_step']-1,reviewed['elapsed_s'])
    after=snap(attempt['end_step'],assessed['elapsed_s'])
    guard.record_attempt(req['request_id'],before,after,outcome='partial',
        reason=attempt['assessment']['classification'],committed_steps=attempt['committed_transitions'])
    restall=next(e for e in events if e['event']=='resumed_collection_stalled')
    current=snap(restall['step'],restall['elapsed_s']);evidence=restall['evidence']
    guard.record_failure(current,evidence,'collect_wood')
    next_request=guard.request(current,evidence)
    next_memory=planning_context(guard,next_request['request_id'])
    assessment=assess_stall(rows[restall['step']-1]['server_after'],evidence,baseline_count=0)
    next_packet=dict(request_id=next_request['request_id'],evidence_id=evidence['event_id'],
        root_goal=dict(goal='wood',target_count=1),root_reference=current['goal_ref'],assessment=assessment,
        status='stalled',remaining_control_s=180-current['elapsed_s'],runtime_binding=current,
        planner_context=next_memory)
    validate_packet(next_packet);write_json(folder/'next-planner-input.json',next_packet)
    assert not next_memory['call_model'] and next_memory['available_choices']==['remain_paused']
    duplicate=dict(proposal,request_id=next_request['request_id'],evidence_ids=[evidence['event_id']],
                   reason='A different explanation does not change the method.')
    rejected=guard.review(next_request['request_id'],duplicate,current,now_elapsed_s=current['elapsed_s'])
    assert 'same_failed_attempt' in rejected['reasons']
    assert PlanHistory(folder/'history.json').snapshot()==guard.history.snapshot()
    assert all(digest(run/n)==sha for n,sha in sources.items())
    results.append(dict(source_run=str(run.relative_to(root)),source_sha256=sources,
        initial_failed_methods=len(memory['failed_attempts']),
        next_failed_methods=len(next_memory['failed_attempts']),attempts=len(next_memory['attempts']),
        duplicate_reasons=rejected['reasons'],next_available_choices=next_memory['available_choices'],
        model_calls=0,new_actions=0,scope='Offline counterfactual request; no new release or game evidence.'))
write_json(out/'report.json',dict(verified=True,rows=results,api_calls=0,game_actions=0,optimizer_updates=0,
    source_sha256={n:digest(root/n) for n in ('src/minecraftrl/replanning.py','src/minecraftrl/recovery_draft.py',
        'scripts/replay_recovery_memory.py')}))
print(json.dumps([{k:v for k,v in r.items() if k!='source_sha256'} for r in results]))
