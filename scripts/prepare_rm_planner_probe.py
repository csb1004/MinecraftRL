"""Prepare a reviewable archived full-plan API probe. Does not call any provider."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from minecraftrl.rm_planner import fixture_plan,payload,RMPlanScheduler,validate_plan,planning_packet
from minecraftrl.replanning import PlanHistory,ReplanGuard,blocked_method
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def prepare(replay,output):
    root=Path(__file__).resolve().parents[1];replay=replay.resolve();output=output.resolve()
    if not replay.is_relative_to(root/'results') or not output.is_relative_to(root/'results'):
        raise ValueError('probe must remain in project results')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    evidence_report=read(replay.parent/'report.json')
    if not evidence_report['verified']:raise ValueError('verified archived reconstruction required')
    provenance=next(row for row in evidence_report['rows'] if replay.name in row['source_run'])
    run=root/provenance['source_run']
    for name,sha in provenance['source_sha256'].items():
        if digest(run/name)!=sha:raise ValueError('original observed trajectory changed')
    old_packet=read(replay/'next-planner-input.json')
    history=PlanHistory(replay/'history.json').snapshot()
    # The old second request was an offline duplicate-check test, never a real call.
    # Preserve real failure/attempt evidence, create a new clearly archived proposal request.
    history['requests']=history['requests'][:1];history['decisions']=history['decisions'][:1]
    assert len(history['failures'])==2 and len(history['attempts'])==1
    output.mkdir(exist_ok=False)
    local_history=PlanHistory(output/'archived-history.json');local_history.commit(history)
    guard=ReplanGuard(local_history,history['requests'][0]['capabilities'],allow_goal_conditioning=True)
    s=old_packet['runtime_binding'];event=history['failures'][-1]['evidence']
    request=guard.request(s,event)
    scheduler=RMPlanScheduler(output/'archived-root.json',s)
    packet=planning_packet(guard,request['request_id'],scheduler);packet['mode']='archived_review'
    call=payload(packet)
    write_json(output/'packet.json',packet);write_json(output/'request-payload.json',call)
    checks=[]
    for label,unsupported in (('registered_plan',False),('missing_navigation_skill',True)):
        plan=fixture_plan(packet,unsupported=unsupported);validate_plan(plan,packet)
        node=sorted((n for n in plan['steps'] if not n['depends_on']),key=lambda n:-n['priority'])[0]
        cap=packet['execution_registry'].get(node['skill_id'])
        reason=('missing_capability' if cap is None else
                'same_failed_attempt' if blocked_method(s,cap,history) else 'advisory_supported')
        selected=dict(node_id=node['id'],reason=reason,execution_authorized=False)
        write_json(output/(label+'-proposal.json'),plan)
        checks.append(dict(label=label,selection=selected))
    assert [c['selection']['reason'] for c in checks]==['same_failed_attempt','missing_capability']
    write_json(output/'local-review.json',dict(completed=True,checks=checks,actual_api_calls=0,
        actual_new_games=0,scope='Archived failed episode; no fabricated fresh neutral or execution authority'))
    sources=[root/'src/minecraftrl'/name for name in ('rm_planner.py','semantic.py','replanning.py','memory.py')]
    sources += [root/'scripts/run_rm_planner_probe.py',Path(__file__)]
    inputs=[replay/'history.json',replay/'next-planner-input.json',replay.parent/'report.json',
            output/'packet.json',output/'request-payload.json']+[run/name for name in provenance['source_sha256']]
    manifest=dict(prepared=True,model='gpt-5.3-codex',maximum_calls=1,max_output_tokens=2048,
        timeout_s=20,retries=0,new_game_actions=0,paid_authorization_remaining=0,
        source_run=str(run),source_packet=str(replay/'next-planner-input.json'),actual_previous_attempts=len(packet['memory']['attempts']),
        source_sha256={str(p.relative_to(root)):digest(p) for p in sources+inputs},
        scope='Archived full-plan generation only; separate explicit paid authorization required')
    write_json(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('replay',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.replay,a.output)))
