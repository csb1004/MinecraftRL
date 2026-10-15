"""Check history semantics on a saved game; never call an API or generate motor actions."""
import json
from pathlib import Path
from minecraftrl.adaptive_route import AdaptiveRoute
from minecraftrl.exploration_route import pose_only
from minecraftrl.navigation_choice import choice_packet,validate_choice
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    root=Path(__file__).resolve().parents[1];base=root/'results/adaptive-recovery-20261014'
    run=base/'far/adaptive_llm/run';out=base/'route-history-replay.json'
    if out.exists():raise FileExistsError('replay exists')
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    assert read('waypoint-audit.json')['verified']
    names=('waypoint-rows.json','waypoint-switches.json','exploration-route.json','candidate-events.json')
    original={name:digest(run/name) for name in names}
    rows=read('waypoint-rows.json');switches=read('waypoint-switches.json');recorded=read('exploration-route.json')
    start=switches[0]['sequence'];route=AdaptiveRoute(pose_only(rows[start]['server_before']),
        [[r['server_before']['server_'+k] for k in 'xyz'] for r in rows[:start]])
    applied={e['sequence']:e for e in read('candidate-events.json') if e['event']=='choice_applied'}
    requests={}
    for folder in (run/'candidate-exchange').glob('request-*'):
        packet=json.loads((folder/'request.json').read_text())['packet']
        reply=json.loads((folder/'response.json').read_text())['proposal']
        validate_choice(reply,packet)  # Archived v1 must remain readable.
        requests[packet['sequence']]=(packet,reply)
    nav_steps=0;request_comparison=[]
    for i,row in enumerate(rows[start:],start):
        for switch in switches:
            if switch['sequence']==i and switch.get('reason')=='root_probe_no_access':
                route.replan(row['server_before'],sequence=nav_steps,reason='root_probe_no_access')
        if i in applied:
            event=applied[i];old,reply=requests[nav_steps]
            packet=choice_packet(route,old['request_id'],nav_steps)
            validate_choice(reply,packet)
            request_comparison.append(dict(nav_sequence=nav_steps,
                archived_failure_count=len(old['previous_failures']),
                corrected_failure_count=len(packet['previous_failures']),
                interrupted_candidate_count=len(packet['interrupted_candidates']),
                navigation_failure_trigger_satisfied=bool(route.failures)))
            route.choose(event['candidate_index'],sequence=nav_steps,source='llm')
        if row['mode'] in ('waypoint','root_probe') and not row['terminal']:
            if row['mode']=='waypoint':assert row['target']==route.targets[route.index]
            route.observe_action(row['server_before'],row['server_after'],row['action'],sequence=i+1)
            if row['mode']=='waypoint':nav_steps+=1;route.observe(row['server_after'],sequence=nav_steps)
    failures=[v for v in recorded['failures'] if v['reason']!='root_probe_no_access']
    interruptions=[v for v in recorded['failures'] if v['reason']=='root_probe_no_access']
    assert route.failures==failures and len(failures)==1
    assert route.interruptions==interruptions and len(interruptions)==2
    assert route.targets==recorded['targets'] and route.decisions==recorded['decisions']
    assert route.blocked_directions==recorded['blocked_directions']
    assert request_comparison[1]['archived_failure_count']==1
    assert request_comparison[1]['corrected_failure_count']==0
    assert not request_comparison[1]['navigation_failure_trigger_satisfied']
    assert original=={name:digest(run/name) for name in names}
    result=dict(verified=True,archived_steps=len(rows),navigation_steps=nav_steps,
        genuine_navigation_failures=route.failures,interrupted_candidates=route.interruptions,
        request_comparison=request_comparison,original_unchanged=True,new_api_calls=0,new_game_actions=0,
        scope='Replay saved poses/actions and old accepted choices; validates history classification, not post-fix live performance',
        source_sha256=original)
    write_json(out,result);print(json.dumps(result))


if __name__=='__main__':main()
