"""Replay every committed action, relative goal, reward and STOP boundary of a real trial."""
import argparse,json,math,sys
from pathlib import Path
import torch
import numpy as np
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import guided
from minecraftrl.rl_core import SHAPED_REWARDS,tick_delta
from minecraftrl.runtime import jsonable,write_json
from minecraftrl.waypoint_actor import navigation_head,append_goal
from minecraftrl.waypoints import goal_features,RouteProgress
from minecraftrl.exploration_route import ExplorationRoute,pose_only
from minecraftrl.adaptive_route import AdaptiveRoute
from minecraftrl.navigation_choice import validate_choice
from minecraftrl.stall_detection import StallDetector
from minecraftrl.stall_replay import wood_sample
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--actor',type=Path,default=Path('results/waypoint-training-20261010/checkpoint.pt'));a=p.parse_args()
    root=Path(__file__).resolve().parents[1];run=a.run.resolve();out=run/'waypoint-audit.json'
    if not run.is_relative_to(root/'results') or out.exists():raise ValueError('new project audit required')
    read=lambda name:json.loads((run/name).read_text())
    report=read('summary.json');rows=read('waypoint-rows.json');switches=read('waypoint-switches.json')
    assert report['controller']=='steve1_waypoint_development' and 'error' not in report
    assert report['rl_updates']==report['direct_action_overrides']==0
    assert 0<=report['actual_llm_calls']<=2
    assert not report['process_cleanup']['remaining_live_pids']
    release=read('input-release-state.json');assert release['verified'] and release['held_key_count']==0
    data=torch.load(run/'waypoint-trajectory.pt',map_location='cpu');saved=torch.load(a.actor,map_location='cpu')
    assert not data['eligible'] and saved['schema']=='waypoint_actor_v1'
    assert digest(a.actor)==report['condition']['waypoint_actor_sha256']
    parent=root/saved['parent'];assert digest(parent)==saved['parent_sha256']
    weights=torch.load(parent,map_location='cpu');torch.set_num_threads(1)
    sys.path.insert(0,str(root/'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.action_head import make_action_head
    from steve1.VPT.lib.actions import ActionTransformer
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
    mapper=CameraHierarchicalMapping(n_camera_bins=11);transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
    base=make_action_head(DictType(**mapper.get_action_space_update()),2054,**model_kwargs(root/'weights/steve1/2x.model')[1]).cuda().eval()
    base.load_state_dict(weights['actor']);nav=navigation_head(base);nav.load_state_dict(saved['actor']);del weights
    previous=read('initial-state.json')['server'];initial=previous;rm=SHAPED_REWARDS['wood-potential-v4'](previous)
    start=switches[0]['sequence'] if switches else None
    planner=report['condition'].get('waypoint_planner');adaptive=planner in ('adaptive_rule','adaptive_llm')
    history_schema=report['condition'].get('route_history_schema',1)
    assert history_schema in (1,2)
    blind=planner=='visit_memory' or adaptive
    choices={};delivered={}
    for exchange in (run/'candidate-exchange').glob('request-*'):
        request=json.loads((exchange/'request.json').read_text())['packet']
        response=exchange/'response.json'
        if response.exists():
            reply=validate_choice(json.loads(response.read_text())['proposal'],request)
            assert json.loads((exchange/'broker-result.json').read_text())['status']=='delivered'
            delivered[request['sequence']]=reply['candidate_index']
    if adaptive and (run/'candidate-events.json').exists():
        plan_events=read('candidate-events.json');waiting=None
        for event in plan_events:
            if event['event']=='request':
                waiting=event['sequence'];assert waiting>0
                assert rows[waiting-1]['mode']=='planner_neutral' and rows[waiting-1]['neutral_phase']=='release'
            elif event['event']=='reply':
                assert waiting==event['sequence']  # No policy transition while the request was outstanding.
            elif event['event']=='choice_applied':
                assert event['sequence']==waiting+1
                assert rows[waiting]['mode']=='planner_neutral' and rows[waiting]['neutral_phase']=='refresh'
            if event['event']=='choice_applied':choices[event['sequence']]=event
    route=None
    if switches:
        if blind:
            pose=initial if start==0 else rows[start-1]['server_after']
            route=(AdaptiveRoute if adaptive else ExplorationRoute)(pose_only(pose),[[r['server_before']['server_'+k] for k in 'xyz'] for r in rows[:start]])
        else:route=RouteProgress(switches[0]['targets'])
    detector=StallDetector();detector.update(wood_sample(initial,{},str(run)));trigger=None
    n=dispatch=inferences=neutral_dispatch=neutral_encodings=nav_steps=0;stopped=accepted=False;error=distance=0.;goal_details={}
    for line in (run/'events.jsonl').open():
        e=json.loads(line);kind=e['event']
        if kind in ('inference_started','policy_dispatch_started','control_dispatch_started','context_encoding_started'):
            assert not stopped and not accepted
            inferences+=kind=='inference_started';dispatch+=kind=='policy_dispatch_started'
            neutral_dispatch+=kind=='control_dispatch_started';neutral_encodings+=kind=='context_encoding_started'
        if kind=='stop_acknowledged':stopped=True
        if kind=='target_reached':accepted=True
        if kind!='transition':continue
        raw=rows[n];row=data['rows'][n];n+=1
        assert not stopped and e['step']==n and row['env_action']==raw['action']==e['action']
        assert raw['server_before']==previous and raw['server_after']==e['info']['workshop']
        assert row['waypoint_goal']==raw['goal']
        if adaptive and route is not None:
            for switch in switches:
                if switch['sequence']==n-1 and switch.get('reason')=='root_probe_no_access':
                    assert rows[n-2]['mode']=='root_probe'
                    route.replan(previous,sequence=nav_steps,reason='root_probe_no_access')
                    if history_schema==1:
                        # Offline compatibility only: six archived games recorded interruptions as failures.
                        route.failures.append(route.interruptions.pop())
            if n-1 in choices:
                chosen=choices[n-1];assert chosen['nav_sequence']==nav_steps
                assert delivered[nav_steps]==chosen['candidate_index']
                route.choose(chosen['candidate_index'],sequence=nav_steps,source='llm')
        context=torch.tensor(rm.context(),dtype=row['features'].dtype)
        assert torch.equal(row['features'][0,0,-6:],context) and torch.equal(row['features'][1,0,-6:],context)
        head=base;features=row['features'].cuda()
        if raw['mode']=='waypoint':
            target=route.targets[route.index] if adaptive else route.observe(previous,sequence=n-1-start)
            assert target is not None and raw['target']==target
            assert raw['goal']==goal_features(previous,target)
            head=nav;features=append_goal(features,raw['goal'])
            key=str(route.index);entry=goal_details.setdefault(key,dict(target=target,actions=0,forward=0,attack=0,yaw=0,minimum_distance=1e9))
            entry['actions']+=1
            entry['forward']+=bool(np.asarray(raw['action'].get('forward',0)).item())
            entry['attack']+=bool(np.asarray(raw['action'].get('attack',0)).item())
            entry['yaw']+=bool(np.asarray(raw['action']['camera']).reshape(-1)[1]);after=raw['server_after']
            entry['minimum_distance']=min(entry['minimum_distance'],math.hypot(target[0]-after['server_x'],target[2]-after['server_z']))
            if not raw['terminal']:
                if adaptive:
                    route.observe_action(previous,after,raw['action'],sequence=n)
                    nav_steps+=1;route.observe(after,sequence=nav_steps)
                else:route.observe(after,sequence=n-start)
        else:
            assert raw['goal'] is None
            if raw['mode']=='resumed' and not adaptive:assert route.completed
            if adaptive and raw['mode']=='root_probe' and not raw['terminal']:
                route.observe_action(previous,raw['server_after'],raw['action'],sequence=n)
        if raw['mode']=='planner_neutral':
            assert row['context_only'] and row['action'] is None and row['old_logp'] is None
            assert all(np.asarray(v).sum()==0 and np.all(np.asarray(v)==0) for v in e['action'].values())
            assert raw['server_after']['held_key_count']==0
        else:
            with torch.no_grad():
                action={k:v.cuda() for k,v in row['action'].items()};logp=head.logprob(action,guided(head,features)).item()
                error=max(error,abs(logp-row['old_logp']))
            mapped={k:v[:,0].numpy() for k,v in row['action'].items()}
            assert jsonable(transformer.policy2env(mapper.to_factored(mapped)))==e['action']
        after=raw['server_after'];tick_delta(previous,after);parts=rm.advance(after,terminal=e['done'])
        assert jsonable(parts)==raw['reward_components']==e['info']['rl_reward']
        assert parts['total']==raw['reward']==row['reward']==e['reward']
        assert raw['terminal']==row['terminal']==bool(e['done'] or rm.completed)
        if raw['mode']=='root' and not raw['terminal']:
            decision=detector.update(wood_sample(after,raw['action'],str(run),previous=previous,terminal=False))
            if decision['event']=='stalled':trigger=decision
        distance+=math.hypot(after['server_x']-previous['server_x'],after['server_z']-previous['server_z']);previous=after
    assert stopped and n==dispatch+neutral_dispatch==report['steps']==len(rows)==len(data['rows'])
    assert dispatch<=inferences<=dispatch+1 and neutral_dispatch<=neutral_encodings<=neutral_dispatch+1
    assert error<2e-4,error
    if adaptive:
        probes=[]
        for switch in switches[1:]:
            sequence=switch['sequence'];g=rows[sequence]['server_before'] if sequence<len(rows) else rows[-1]['server_after']
            if switch['mode']=='resumed':
                from minecraftrl.rl_core import ShapedWoodRewardV3
                assert ShapedWoodRewardV3.state(g)[0] in ('mine','collect')
            if switch['mode']=='root_probe':
                p=[g['server_'+k] for k in 'xyz'];assert len(probes)<2
                if probes:assert math.dist(p[::2],probes[-1][::2])>3
                probes.append(p)
                remaining=rows[sequence:];count=0
                for probe_row in remaining:
                    if probe_row['mode']!='root_probe':break
                    count+=1
                assert count<=100
                origin=rows[start]['server_before'];origin_xz=[origin['server_x'],origin['server_z']]
                assert math.dist(p[::2],origin_xz)>=3
                evidence=[v for v in rows[start:sequence] if v['mode'] in ('waypoint','root_probe')][-20:]
                assert len(evidence)==20 and all(math.dist([v['server_after']['server_x'],v['server_after']['server_z']],origin_xz)>=3 for v in evidence)
                assert any(r['sequence']<=sum(x['mode']=='waypoint' for x in rows[:sequence]) for r in report['route_receipts'])
        assert sum(r['mode'] in ('root_probe','waypoint') for r in rows)<=400
    assert bool(route and route.completed)==report['route_completed'] and (route.receipts if route else [])==report['route_receipts']
    assert rm.completed==accepted==report['root_inventory_success']
    if switches and 'trigger' in switches[0] and report['condition'].get('waypoint_start','stall')=='stall':
        assert trigger is not None and jsonable(trigger)==switches[0]['trigger']
    if blind:
        recorded=read('exploration-route.json')
        assert recorded['decisions']==(route.decisions if route else [])
        assert recorded['failures']==(route.failures if route else [])
        assert recorded['targets']==(route.targets if route else [])
        if adaptive:assert recorded['blocked_directions']==(route.blocked_directions if route else [])
        if adaptive and history_schema==2:assert recorded['interruptions']==(route.interruptions if route else [])
    result=dict(verified=True,steps=n,action_logp_max_error=error,status=report['status'],
        root_success=rm.completed,route_completed=report['route_completed'],route_receipts=report['route_receipts'],
        actual_llm_calls=report['actual_llm_calls'],neutral_transitions=neutral_dispatch,
        goals=goal_details,path_blocks=distance,stop_dispatches=0,keys_remaining=0,owned_pids_remaining=[],
        inference_ms=report['inference_ms'],control_cycle_ms=report['control_cycle_ms'],gpu=report['gpu'],
        scope='Learned action and RM/goal/STOP consistency; not an autonomous route planner',
        source_sha256={name:digest(run/name) for name in ('summary.json','waypoint-rows.json','waypoint-trajectory.pt','events.jsonl')})
    write_json(out,result);print(json.dumps(result))


if __name__=='__main__':main()
