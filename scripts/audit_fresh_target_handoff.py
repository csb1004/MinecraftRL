"""Audit committed neural/neutral transitions and planning pauses, without gameplay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from minecraftrl.policies import model_kwargs
from minecraftrl.replanning import PlanHistory
from minecraftrl.recovery_evidence import assess_attempt
from minecraftrl.rl_core import SHAPED_REWARDS,tick_delta
from minecraftrl.rl_policy import guided,state_fingerprint
from minecraftrl.runtime import jsonable,write_json
from minecraftrl.stall_detection import StallDetector
from minecraftrl.stall_replay import wood_sample,log_count
from minecraftrl.world_store import digest


def audit(run,checkpoint,exchange=None):
    root=Path(__file__).resolve().parents[1];run=run.resolve()
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    report=read(run/'summary.json');raw=read(run/'goal-recovery-rows.json')
    rm_plan=report.get('condition',{}).get('planning',{}).get('protocol')=='rm_plan_v1'
    subgoals=report.get('condition',{}).get('planning',{}).get('abstract_navigation',False)
    waypoint=report.get('condition',{}).get('planning',{}).get('waypoint_integration',False)
    destination=run/('rm-navigation-audit.json' if waypoint else 'subgoal-audit.json' if subgoals else 'rm-plan-audit.json' if rm_plan else 'fresh-target-handoff-audit.json')
    if destination.exists():raise FileExistsError(destination)
    data=torch.load(run/'trajectory.pt',map_location='cpu')
    weights=torch.load(checkpoint,map_location='cpu')
    assert report['controller']=='steve1_planned_goal_conditioning'
    assert report['rl_updates']==0 and not report['rl_training_eligible']
    actual_llm_calls=report['actual_llm_calls'];broker=None
    if report['planner_provider'] in ('windows_mailbox','windows_rm_plan_mailbox'):
        assert exchange is not None
        exchange=exchange.resolve()
        if report['planner_provider']=='windows_rm_plan_mailbox':
            assert report['planner_calls']<=1, 'This live audit is bounded to one separately authorized call'
            exchange=exchange/'request-001'
        receipt=read(exchange/'api-receipt.json') if (exchange/'api-receipt.json').exists() else None
        actual_llm_calls=receipt['network_calls'] if receipt else 0
        assert actual_llm_calls in (0,1)
        if not report['planner_live']:assert actual_llm_calls==0
        if report['planner_calls']:
            broker=read(exchange/'broker-result.json')
            assert broker['provider_invocations']<=1 and broker['retries']==0
            assert not broker['execution_authorized'] and broker['new_game_actions']==0
            if report['planner_live'] and broker['status']=='delivered':assert actual_llm_calls==1
    else:
        assert report['planner_provider'] in ('fixed_local_fixture','fixed_rm_plan_fixture','fixed_neural_subgoal_fixture','fixed_rm_navigation_fixture') and actual_llm_calls==0
    assert data['scope']=='planning_goal_conditioning_diagnostic' and not data['eligible'] and data['split']=='eval'
    fingerprints={k:state_fingerprint(weights[k]) for k in ('actor','value','backbone_tail','backbone_temporal')}
    assert fingerprints==data['initial_head_fingerprints']==report['initial_head_fingerprints']
    assert state_fingerprint(data['initial_recurrent_state'])==data['condition']['initial_recurrent_sha256']
    assert report['release']=='neutral_command_sent' and not report['process_cleanup']['remaining_live_pids']
    release=read(run/'input-release-state.json');assert release['verified'] and release['held_key_count']==0
    assert len(raw)==len(data['rows'])==report['steps']>0
    decisions=read(run/'stall-decisions.json');assert len(decisions)==len(raw)
    initial=read(run/'initial-state.json');previous=start=initial['server']
    reward=SHAPED_REWARDS[data['reward_config']['name']](start)
    detector=StallDetector()
    diagnostic_wait=report.get('condition',{}).get('intervention',{}).get('diagnostic_initial_goal_budget_ticks')
    if diagnostic_wait is not None:
        assert diagnostic_wait==32 and report['condition']['planning']['navigation_strategy']=='local_passability'
        protocol=read(run.parent/'protocol.json')
        assert protocol['diagnostic_initial_goal_budget_ticks']==32 and 'NOT default-policy' in protocol['scope']
        detector.goal_budget_ticks=diagnostic_wait
    detector.update(wood_sample(start,{},str(run)))
    local_execution=report.get('condition',{}).get('planning',{}).get('local_execution')
    if local_execution:
        from minecraftrl.local_execution import configure_stall,LOCAL_STALL_WINDOWS,LOCAL_GOAL_WAIT,mining_temperature
        assert local_execution['stall_windows']==LOCAL_STALL_WINDOWS
        assert local_execution['goal_wait_ticks']==LOCAL_GOAL_WAIT
        configure_stall(detector,wood_sample(start,{},str(run)).stage)
    camera_receipts=[]
    camera_detection=report.get('condition',{}).get('planning',{}).get('wood_detection')
    if camera_detection:
        from minecraftrl.visible_wood import validate_acquisition
        assert start['wood_camera_rays_enabled']==1
        receipt=validate_acquisition(start)
        if receipt:camera_receipts.append(receipt)
    sys.path.insert(0,str(root/'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.action_head import make_action_head
    from steve1.VPT.lib.actions import ActionTransformer
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
    mapper=CameraHierarchicalMapping(n_camera_bins=11);transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
    actor=make_action_head(DictType(**mapper.get_action_space_update()),data['rows'][0]['features'].shape[-1],
        **model_kwargs(root/'weights/steve1/2x.model')[1]).cuda()
    actor.load_state_dict(weights['actor']);actor.requires_grad_(False);del weights
    if waypoint:
        from minecraftrl.waypoint_actor import append_goal
        from minecraftrl.navigation_model import load_navigation,append_motion
        from minecraftrl.navigation_refinement import motion_goal
        from minecraftrl.waypoints import goal_features
        planning=report['condition']['planning'];path=Path(planning['navigation_actor'])
        assert digest(path)==planning['navigation_actor_sha256']
        nav_saved=torch.load(path,map_location='cpu')
        assert nav_saved['schema'] in ('waypoint_actor_v1','navigation_refinement_v1') and nav_saved['parent_sha256']==digest(checkpoint)
        assert nav_saved['embedding_sha256']==digest(run/'goal-embed.npy')
        navigation,refined=load_navigation(actor,nav_saved);navigation.requires_grad_(False)
        if refined:assert planning['navigation_input']=='motion_goal_10' and nav_saved['motion_dim']==10
        nav_counts={};schedule=read(run/'rm-schedule.json')
    embeds={mode:hashlib.sha256(np.load(run/name,allow_pickle=False).tobytes()).hexdigest()
            for mode,name in (('wood','goal-embed.npy'),('intervention','intervention-embed.npy'))}
    skill_embeds={}
    if subgoals:
        for name,item in read(run/'neural-skills.json').items():
            sha=hashlib.sha256(np.load(run/(name+'-embed.npy'),allow_pickle=False).tobytes()).hexdigest()
            assert sha==item['embedding_sha256']==item['capability']['arguments']['embedding_sha256']
            skill_embeds[name]=sha
    n=neural=neutral=policy_dispatches=control_dispatches=0;max_error=0.
    stopped=accepted=waiting=False;started=None;wait_started=None;wait_s=None
    blocked_activity=[];sums={k:0. for k in ('task','shaping','time_cost','failure')};dispatch_times=[]
    for line in (run/'events.jsonl').open(encoding='utf-8'):
        e=json.loads(line);kind=e['event']
        if kind=='episode_started':started=e['monotonic_s']
        if kind=='planner_wait_started':
            assert not waiting;waiting=True;wait_started=e['monotonic_s']
        if kind in ('planner_response_received','planner_paused','stop_acknowledged') and waiting:
            waiting=False;wait_s=e['monotonic_s']-wait_started
        if kind=='stop_acknowledged':stopped=True
        if kind=='target_reached':accepted=True
        if kind in ('inference_started','context_encoding_started','policy_dispatch_started','control_dispatch_started'):
            assert not stopped and not accepted
            if waiting:blocked_activity.append(e)
        if kind in ('policy_dispatch_started','control_dispatch_started'):
            dispatch_times.append(e['monotonic_s']-started)
            policy_dispatches+=kind=='policy_dispatch_started';control_dispatches+=kind=='control_dispatch_started'
        if kind!='transition':continue
        row=raw[n];feature=data['rows'][n];n+=1
        assert not stopped and not accepted and e['step']==n
        assert row['server_before']==previous and row['server_after']==e['info']['workshop']
        assert row['action']==feature['env_action']==e['action'] and row['owner']==e['controller']
        assert row['reward_context']==feature['reward_context']==reward.context()
        expected=torch.tensor(reward.context(),dtype=feature['features'].dtype)
        assert torch.equal(feature['features'][0,0,-6:],expected) and torch.equal(feature['features'][1,0,-6:],expected)
        expected_embed=embeds[row['mode']]
        is_waypoint=waypoint and row.get('waypoint_goal') is not None
        if is_waypoint:
            if refined:
                tag=row['rm_plan'];key=(tag['revision'],tag['node_id']);count=nav_counts.get(key,0)
                plan=schedule['plans'][tag['revision']-1]
                node=next(v for v in plan['proposal']['steps'] if v['id']==tag['node_id'])
                prior=raw[n-2]['server_before'] if n>1 else None
                local=row.get('local_route')
                target=local['target'] if local else row['waypoint_target']
                remaining=node['budget_ticks']-count
                if local:
                    receipt=next(v['receipt'] for v in plan['nodes'] if v['id']==tag['node_id'])
                    remaining=min(100,node['budget_ticks']-((n-1)-receipt['before']['sequence']))
                expected_goal=motion_goal(previous,target,prior,remaining=remaining)
                nav_counts[key]=count+1
            else:expected_goal=goal_features(previous,row.get('local_route',{}).get('target',row.get('waypoint_target')))
            assert feature['waypoint_goal']==row['waypoint_goal']==expected_goal
            expected_embed=embeds['wood']
        elif subgoals:
            from minecraftrl.subgoal_skills import navigation_state
            tag=row['subgoal_context'];assert tag==feature['subgoal_context']
            observed=navigation_state(previous,log_count(start))
            if 'gui_open' not in tag['abstract_state']:observed.pop('gui_open')  # Preserved pre-fix failure, closed frames only.
            assert tag['abstract_state']==observed
            if tag['phase']=='subgoal':expected_embed=skill_embeds[tag['skill_id']]
        if row.get('local_route') is not None:
            from minecraftrl.local_navigation import checked_grid,validate_candidate
            from minecraftrl.subgoal_skills import navigation_state
            assert planning['navigation_strategy']=='local_passability'
            local=row['local_route'];assert local==feature['local_route']
            tag=row['rm_plan'];plan=schedule['plans'][tag['revision']-1]
            cap=plan['registry'][row['subgoal_context']['skill_id']]
            assert cap['arguments']['navigation_actor_sha256']==planning['navigation_actor_sha256']
            assert row['subgoal_context']==feature['subgoal_context']
            assert row['subgoal_context']['abstract_state']==navigation_state(previous,log_count(start))
            grid=checked_grid(previous)
            if local['choice'] is not None:
                assert local['choice']['target']==local['target'] and validate_candidate(previous,local['target'])
            elif local['target'] is not None:
                assert abs(previous['server_y']-local['target'][1])>.15  # Carry the last verified ground goal while airborne.
            expected_embed=embeds['wood']
        # PlanningRecoveryController.act explicitly restores the root embedding
        # for release/refresh, even after a text-conditioned subgoal fails.
        if waypoint and row['owner']=='planner_neutral':expected_embed=embeds['wood']
        assert feature['goal_conditioning']==dict(mode=row['mode'],embedding_sha256=expected_embed)
        assert row['embedding_sha256']==expected_embed
        assert not row['loss_eligible']
        if row['owner']=='planner_neutral':
            neutral+=1
            assert row['context_only'] and feature['context_only'] and feature['old_logp'] is None and feature['action'] is None
            assert all(np.count_nonzero(v)==0 for v in row['action'].values())
            assert row['server_after']['held_key_count']==0 and e['inference_ms'] is None
            assert len(row['memory_before'])==len(row['memory_after'])==64
        else:
            neural+=1
            assert row['owner']=='steve1_'+row['mode']
            with torch.no_grad():
                action={k:v.cuda() for k,v in feature['action'].items()}
                head=navigation if is_waypoint else actor
                features=feature['features'].cuda()
                assert torch.isfinite(features).all() and torch.isfinite(torch.tensor(feature['old_logp']))
                if is_waypoint:features=(append_motion if refined else append_goal)(features,row['waypoint_goal'])
                temperature=feature.get('action_temperature',1.)
                assert temperature==row.get('action_temperature',1.)
                if local_execution:
                    assert temperature==mining_temperature(previous,row.get('subgoal_context',{}).get('phase'))
                else:assert temperature==1.
                pd=guided(head,features,temperature=temperature);assert all(torch.isfinite(v).all() for v in pd.values())
                actual=head.logprob(action,pd).item();assert np.isfinite(actual)
            max_error=max(max_error,abs(actual-feature['old_logp']))
            converted=transformer.policy2env(mapper.to_factored({k:v[:,0].numpy() for k,v in feature['action'].items()}))
            assert jsonable(converted)==row['action']
        server=row['server_after'];assert tick_delta(previous,server)==1
        if camera_detection:
            assert server['wood_camera_rays_enabled']==1
            receipt=validate_acquisition(server)
            if receipt:camera_receipts.append(receipt)
        parts=reward.advance(server,terminal=e['done'])
        assert jsonable(parts)==jsonable(row['reward_components'])==jsonable(feature['reward_components'])==jsonable(e['info']['rl_reward'])
        assert parts['total']==row['reward']==feature['reward']==e['reward']
        assert row['terminal']==feature['terminal']==bool(e['done'] or reward.completed)
        for key in sums:sums[key]+=parts[key]
        decision=detector.update(wood_sample(server,e['action'],str(run),previous=previous,terminal=row['terminal']))
        decision['step']=n;assert jsonable(decision)==decisions[n-1]
        previous=server
    assert stopped and not blocked_activity and n==neural+neutral==policy_dispatches+control_dispatches
    assert neural==policy_dispatches and neutral==control_dispatches==report['neutral_control_transitions']
    assert max_error<=2e-4 and max(dispatch_times)<180
    assert report['server_tick_deltas']==[1]*n and sums==report['reward_component_sums']
    switches=read(run/'goal-switches.json');planner=read(run/'planner-events.json')
    if broker is not None:
        request=read(exchange/'request.json')
        started_request=next(e for e in planner if e['event']=='wait_started')
        assert request['packet']==started_request['packet']
        assert digest(exchange/'request.json')==broker['request_sha256']
        responses=[e for e in planner if e['event']=='response_received']
        if responses:
            assert len(responses)==1 and broker['status']=='delivered'
            assert read(exchange/'response.json')['proposal']==broker['proposal']==responses[0]['reply']
        assert read(exchange/'closed.json')['closed']
    history=PlanHistory(run/'planner-history.json').snapshot()
    assert len(history['requests'])==report['planner_calls']<=(3 if rm_plan else 1)
    gains=log_count(previous)-log_count(start)
    assert bool(gains>0)==reward.completed==accepted==report['server_target_reached']
    intervention=[r for r in raw if r['mode']=='intervention'];assert len(intervention)<=(1800 if subgoals else 600 if rm_plan else 200)
    if rm_plan:
        if subgoals:
            from audit_subgoal_schedule import verify_schedule
        else:
            from audit_rm_schedule import verify_schedule
        result=verify_schedule(run,report,raw,data,history,planner)
        result.update(verified=True,steps=n,neural_steps=neural,neutral_steps=neutral,
            log_gain=gains,goal_completed=accepted,action_head_logp_max_error=max_error,
            policy_dispatches_during_wait=0,post_stop_dispatches=0,post_stop_inferences=0,
            held_keys_after_release=0,remaining_owned_pids=[],last_dispatch_s=max(dispatch_times),
            actual_llm_calls=actual_llm_calls,optimizer_updates=0,inference_ms=report['inference_ms'],
            control_cycle_ms=report['control_cycle_ms'],gpu=report['gpu'],device_vram=report['device_vram'],
            checkpoint_sha256=digest(checkpoint),worker_exit_code=report['worker_exit_code'],
            runtime_error=report.get('error'),execution_valid=report['execution_valid'])
        if camera_detection:result['validated_camera_acquisitions']=camera_receipts
        if diagnostic_wait is not None:
            result.update(diagnostic_initial_goal_budget_ticks=diagnostic_wait,
                scope='Targeted real-game recovery integration with shorter initial stall wait; NOT default-policy performance or training success')
        write_json(destination,result);print(json.dumps(result));return
    neutral_rows=[r for r in raw if r['owner']=='planner_neutral']
    if report['planner_calls']==0:
        assert not switches and not intervention
        if report['status']=='error':
            assert report['error']=='ValueError: invalid release evidence' and neutral==1
            assert not report['execution_valid']
        elif report['status']=='planning_paused':
            assert neutral==1 and not report['execution_valid']
            assert any(e.get('reason') in ('insufficient_remaining_time','neutral_release_unverified') for e in planner)
        else:
            assert report['execution_valid'] and report['worker_exit_code']==0 and neutral==0
    elif report['status']=='cancelled':
        assert not report['execution_valid']
        if switches:
            assert neutral==2 and history['requests'][0]['consumed']
        else:
            assert neutral==1 and not intervention and not history['requests'][0]['consumed']
            assert not any(e['event']=='response_received' for e in planner)
    elif report['status']=='planning_paused':
        assert neutral in (1,2) and not report['execution_valid'] and report['worker_exit_code']==2
        assert any(e['event']=='paused' for e in planner)
        if intervention:
            assert neutral==2 and len(switches)>=1
            assert history['requests'][0]['consumed'] and history['decisions'][0]['accepted']
        else:
            assert not switches and not any(e['accepted'] for e in history['decisions'])
    else:
        assert report['execution_valid'] and report['worker_exit_code']==0
        assert neutral==2 and neutral_rows[0]['memory_after']==neutral_rows[1]['memory_before']
        assert len(history['decisions'])==1 and history['decisions'][0]['accepted']
        assert history['requests'][0]['consumed'] and len(switches)>=1
        assert switches[0]['reason']=='planner_review_accepted'
        assert 0<=wait_s<20
    from PIL import Image
    attempts=report['recovery_attempts']
    if attempts:
        assert len(attempts)==1 and attempts==read(run/'recovery-attempts.json')
        receipt=attempts[0];window=raw[receipt['start_step']-1:receipt['end_step']]
        assert window==intervention
        assert receipt['committed_transitions']==len(window)
        before=window[0]['server_before'] if window else switches[0]['server_before']
        assessment=assess_attempt(before,[r['server_after'] for r in window],
            reason=receipt['assessment']['boundary_reason'],baseline_count=log_count(start),budget_ticks=200)
        assert assessment==receipt['assessment']
        assert receipt['root_reference']==history['requests'][0]['snapshot']['goal_ref']
        assert receipt['remaining_calls']==receipt['remaining_interventions']==0
        assessed=[e for e in planner if e['event']=='recovery_assessed']
        assert len(assessed)==1 and all(assessed[0][k]==v for k,v in receipt.items())
        if receipt['next_phase']=='paused':
            assert report['status']=='planning_paused' and receipt['end_step']==len(raw) and len(switches)==1
        elif receipt['next_phase']=='resumed':
            assert assessment['current_target_tracked']
            assert assessment['classification'] in ('interaction_ready','target_acquired')
            if assessment['classification']=='target_acquired':
                assert assessment['target_acquired_on_last_transition']
        # No extra exploration action may follow the first fresh acquisition.
        fresh=[i for i,r in enumerate(window,1) if r['server_before']['wood_v3_kind']==0 and r['server_after']['wood_v3_kind']==1]
        if fresh:assert fresh==[len(window)]
    else:assert not intervention
    resumed=read(run/'resumed-stall-decisions.json');expected_resumed=[]
    if attempts and attempts[0]['next_phase']=='resumed':
        end=attempts[0]['end_step'];monitor=StallDetector()
        monitor.update(wood_sample(raw[end-1]['server_after'],{},str(run)+':resumed'))
        for i,row in enumerate(raw[end:],end+1):
            if row['terminal'] or decisions[i-1].get('event')=='stop':continue
            result=monitor.update(wood_sample(row['server_after'],row['action'],str(run)+':resumed',previous=row['server_before']))
            result['step']=i;expected_resumed.append(result)
        if report['status']=='planning_paused':
            assert expected_resumed[-1]['event']=='stalled'
            assert len(history['failures'])==2
    assert jsonable(expected_resumed)==resumed
    for switch in switches:
        assert switch['committed']==(switch['before_action']<=n)
        pixels=np.asarray(Image.open(run/switch['frame']))
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==switch['input_image_sha256']
        if switch['committed']:
            row=raw[switch['before_action']-1]
            assert row['server_before']==switch['server_before'] and row['mode']==switch['mode']
    result=dict(verified=True,steps=n,neural_steps=neural,neutral_steps=neutral,status=report['status'],
        log_gain=gains,goal_completed=accepted,intervention_steps=len(intervention),
        planner_wait_s=wait_s,action_head_logp_max_error=max_error,policy_dispatches_during_wait=0,
        post_stop_dispatches=0,post_stop_inferences=0,held_keys_after_release=0,remaining_owned_pids=[],
        worker_exit_code=report['worker_exit_code'],actual_llm_calls=actual_llm_calls,optimizer_updates=0,
        planner_provider=report['planner_provider'],broker_status=broker['status'] if broker else None,
        runtime_error=report.get('error'),handoff_exercised=bool(intervention),
        recovery_attempts=attempts,
        resumed_collection_transitions=len(resumed),
        last_dispatch_s=max(dispatch_times),inference_ms=report['inference_ms'],control_cycle_ms=report['control_cycle_ms'],
        context_encoding_ms=report['context_encoding_ms'],gpu=report['gpu'],device_vram=report['device_vram'],
        checkpoint_sha256=digest(checkpoint),source_sha256={name:digest(run/name) for name in (
            'summary.json','trajectory.pt','events.jsonl','goal-recovery-rows.json','planner-events.json','planner-history.json',
            'recovery-attempts.json','resumed-stall-decisions.json') if (run/name).exists()},
        scope='Actual planning handoff integration; no causal performance or optimizer improvement claim')
    write_json(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','device_vram','gpu')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('checkpoint',type=Path)
    p.add_argument('--exchange',type=Path)
    a=p.parse_args();audit(a.run,a.checkpoint,a.exchange)
