"""Independent raw-observation checks for neural subgoals and their plan journal."""
from copy import deepcopy
import hashlib
import json

import numpy as np
from PIL import Image

from minecraftrl.replanning import make_snapshot, goal_satisfied
from minecraftrl.rm_planner import validate_packet, validate_plan
from minecraftrl.stall_replay import log_count
from minecraftrl.subgoal_planner import blocked_subgoal
from minecraftrl.subgoal_skills import SKILLS, completion, navigation_state
from minecraftrl.world_store import digest
from minecraftrl.waypoint_skill import NAMES,assess_navigation,same_failed_route
from minecraftrl.waypoints import arrived


def verify_schedule(run,report,rows,trajectory,history,events):
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    schedule=read('rm-schedule.json');start=read('initial-state.json')['server'];root=schedule['root']
    assert root['baseline']==log_count(start)
    assert root['progress']==max(0,log_count(rows[-1]['server_after'])-log_count(start))
    assert root['completed']==report['server_target_reached'] and schedule['active_node'] is None
    assert len(schedule['plans'])<=report['planner_calls']<=report['condition']['planning']['maximum_calls']
    def at(step):return rows[step-1]['server_after'] if step else start
    def verify_snapshot(s):
        obs=at(s['sequence'])
        actual=make_snapshot(obs,world_id=s['world_id'],episode_id=s['episode_id'],goal='wood',task=None,
            generation=s['generation'],sequence=s['sequence'],elapsed_s=s['elapsed_s'],status=s['status'],
            release_sequence=s['release_sequence'],held_key_count=obs['held_key_count'],baseline_count=root['baseline'])
        actual['navigation']=navigation_state(obs,root['baseline'])
        assert s==actual
    coverage_states={}
    if report['condition']['planning'].get('navigation_strategy') in ('coverage','coverage_motion'):
        from minecraftrl.spatial_memory import SpatialMemory
        from minecraftrl.exploration_coverage import score_candidates,coverage_delta
        requests=[e['packet'] for e in events if e['event']=='wait_started']
        if requests:
            binding=requests[0]['runtime_binding']
            memory=SpatialMemory(binding['world_id'],binding['episode_id'],source='privileged_server',dimension='overworld')
            needed={0}
            for plan in schedule['plans']:
                for node in plan['nodes']:
                    r=node.get('receipt')
                    if r and r.get('type')=='neural_waypoint':needed.update((r['before']['sequence'],r['after']['sequence']))
            by_step={p['runtime_binding']['sequence']:p for p in requests}
            motion=report['condition']['planning'].get('navigation_strategy')=='coverage_motion';route=None
            for step in range(len(rows)+1):
                obs=at(step);memory.observe([obs['server_'+a] for a in 'xyz'],obs['server_tick'])
                if motion and route is not None and step:
                    row=rows[step-1]
                    route.observe_action(row['server_before'],row['server_after'],row['action'] if 'waypoint_goal' in row else {},sequence=step)
                if step in needed:coverage_states[step]=memory.summary()
                if step in by_step:
                    context=by_step[step]['waypoint_context']
                    if motion:
                        from minecraftrl.adaptive_route import AdaptiveRoute
                        from minecraftrl.exploration_route import pose_only
                        route=AdaptiveRoute(pose_only(obs),[[r['server_before']['server_'+a] for a in 'xyz'] for r in rows[:step]],
                            blocked_directions=route.blocked_directions if route else None,windowed_motion=True)
                        assert route.blocked_directions==context['motion_evidence']
                        for actual,expected in zip(context['candidates'],route.candidates):
                            assert actual['target']==expected['target'] and actual['admissible']==expected['admissible']
                    ranked,evidence=score_candidates(memory,context['candidates'])
                    assert evidence==context['exploration']
                    assert [c['cost'] for c in ranked]==[c['cost'] for c in context['candidates']]
    packets={};plans={};waits=[]
    for e in events:
        if e['event']=='wait_started':
            p=e['packet'];validate_packet(p);verify_snapshot(p['runtime_binding'])
            if p.get('waypoint_context',{}).get('source')=='privileged_local_passability_v1':
                from minecraftrl.local_navigation import validate_candidate
                obs=at(p['runtime_binding']['sequence']);context=p['waypoint_context']
                assert context['local_navigation']==obs['local_navigation']
                for c in context['candidates']:
                    if c['admissible']:assert validate_candidate(obs,c['target'])
            request=next(r for r in history['requests'] if r['request_id']==p['request_id'])
            assert p['runtime_binding']==request['snapshot'] and p['execution_registry']==request['capabilities']
            packets[p['request_id']]=p
            assert read(f'planner-input-{len(packets):02d}.json')==p
        elif e['event']=='response_received':
            validate_plan(e['reply'],packets[e['reply']['request_id']]);waits.append(e['wait_s'])
        elif e['event']=='plan_installed':
            p=e['plan'];validate_plan(p,packets[p['request_id']]);verify_snapshot(e['current'])
            saved=schedule['plans'][e['schedule_revision']-1]
            assert saved['proposal']==p and saved['applied_snapshot']==e['current']
            plans[e['schedule_revision']]=saved
        elif e['event']=='node_selected' and e['selection']['node'] is not None:
            node=e['selection']['node'];current=e['current'];verify_snapshot(current)
            plan=plans[e['schedule_revision']];step=e['step'];assert node in plan['proposal']['steps']
            past=deepcopy(history)
            past['failures']=[f for f in past['failures'] if f['snapshot']['sequence']<=step]
            past['attempts']=[a for a in past['attempts'] if a['after'] and a['after']['sequence']<=step]
            previous=deepcopy(schedule['plans'][:e['schedule_revision']])
            for old in previous:
                for n in old['nodes']:
                    if n['receipt'] and n['receipt']['end_sequence']>step:n['receipt']=None
            assert not blocked_subgoal(current,plan['registry'][node['skill_id']],past,previous)
            for dep in node['depends_on']:
                r=next(n['receipt'] for n in plan['nodes'] if n['id']==dep)
                assert r and r['end_sequence']<=step
                if r.get('type') in ('neural_subgoal','neural_waypoint'):assert r['outcome']=='completed'
                else:assert r['classification'] in ('target_acquired','interaction_ready') and r['end_sequence']==step
            if node['id']=='root' and node['depends_on'] and any(
                    n['id'] in node['depends_on'] and n['skill_id'] in (*SKILLS,*NAMES) for n in plan['proposal']['steps']):
                assert current['navigation']['interaction_ready']
            if node['skill_id'] in NAMES:
                cap=plan['registry'][node['skill_id']]
                for old in previous:
                    for old_node in old['nodes']:
                        r=old_node.get('receipt')
                        if r:assert not same_failed_route(cap,at(step),r,world_id=current['world_id'],episode_id=current['episode_id'])
    receipts=[]
    navigation_receipts=[]
    for revision,plan in enumerate(schedule['plans'],1):
        for n in plan['nodes']:
            r=n.get('receipt')
            if r and r.get('type')=='neural_waypoint':
                node=next(x for x in plan['proposal']['steps'] if x['id']==n['id'])
                before,after=r['before'],r['after'];verify_snapshot(before);verify_snapshot(after)
                window=rows[before['sequence']:after['sequence']]
                assert r['method']==plan['registry'][node['skill_id']]
                assert len(window)==r['committed_steps']<=node['budget_ticks']
                assessment=assess_navigation(r['method'],window,reason=r['reason'])
                if r['already_satisfied']:
                    assert not window and arrived(at(after['sequence']),r['method']['arguments']['target'])
                    assessment.update(outcome='completed',diagnosis='already_satisfied')
                assert assessment==r['assessment'] and assessment['outcome']==r['outcome']
                assert r['root_completed']==goal_satisfied(after)
                if 'exploration' in r:
                    assert coverage_states
                    assert r['exploration']==coverage_delta(coverage_states[before['sequence']],coverage_states[after['sequence']])
                for row in window:
                    assert row['rm_plan']['revision']==revision and row['rm_plan']['node_id']==n['id']
                    assert row['waypoint_target']==r['method']['arguments']['target'] and row['mode']=='intervention'
                navigation_receipts.append(r)
                continue
            if not r or r.get('type')!='neural_subgoal':continue
            node=next(x for x in plan['proposal']['steps'] if x['id']==n['id'])
            before,after=r['before'],r['after'];verify_snapshot(before);verify_snapshot(after)
            window=rows[before['sequence']:after['sequence']]
            assert len(window)==r['committed_steps']<=node['budget_ticks']
            assert r['method']==plan['registry'][node['skill_id']]
            assert r['predicate_satisfied']==completion(node['skill_id'],before['navigation'],after['navigation'])
            assert r['root_completed']==goal_satisfied(after)
            assert r['already_satisfied']==(r['reason']=='already_satisfied')
            if r['already_satisfied']:assert not window and r['predicate_satisfied']
            if r['outcome']=='completed':assert r['predicate_satisfied']
            for row in window:
                assert row['rm_plan']['revision']==revision and row['rm_plan']['node_id']==n['id']
                assert row['subgoal_context']['skill_id']==node['skill_id'] and row['mode']=='intervention'
            for row in window[:-1]:
                assert not completion(node['skill_id'],before['navigation'],navigation_state(row['server_after'],root['baseline']))
            receipts.append(dict(skill_id=node['skill_id'],outcome=r['outcome'],reason=r['reason'],
                start=before['sequence']+1,end=after['sequence'],committed_steps=len(window),
                before=before['navigation'],after=after['navigation']))
    for row,saved in zip(rows,trajectory['rows']):
        tag=row['rm_plan'];assert tag==saved['rm_plan'] and tag['root_reference']==root['reference']
        if row['owner']=='planner_neutral':continue
        if tag['revision']==0:assert tag['node_id']=='root' and row['mode']=='wood';continue
        plan=plans[tag['revision']];node=next(n for n in plan['proposal']['steps'] if n['id']==tag['node_id'])
        if node['skill_id'] in NAMES:
            assert row['waypoint_target']==plan['registry'][node['skill_id']]['arguments']['target']
            assert row['waypoint_goal']==saved['waypoint_goal']
        else:assert row['subgoal_context']['skill_id']==node['skill_id']
        assert row['mode']==('wood' if node['id']=='root' else 'intervention')
    from minecraftrl.recovery_evidence import assess_attempt
    for receipt in report['recovery_attempts']:
        window=rows[receipt['start_step']-1:receipt['end_step']]
        if not window:continue
        plan=plans[window[0]['rm_plan']['revision']]
        node=next(n for n in plan['proposal']['steps'] if n['id']==window[0]['rm_plan']['node_id'])
        assessment=assess_attempt(window[0]['server_before'],[r['server_after'] for r in window],
            reason=receipt['assessment']['boundary_reason'],baseline_count=root['baseline'],budget_ticks=node['budget_ticks'])
        assert assessment==receipt['assessment'] and receipt['committed_transitions']==len(window)
    for s in report['switches']:
        assert s['committed']==(s['before_action']<=len(rows))
        assert hashlib.sha256(np.asarray(Image.open(run/s['frame'])).tobytes()).hexdigest()==s['input_image_sha256']
        if s['committed']:assert rows[s['before_action']-1]['embedding_sha256']==s['embedding_sha256']
    assert len(packets)==report['planner_calls'] and len(plans)==len(schedule['plans'])
    if navigation_receipts:assert read('navigation-receipts.json')==navigation_receipts
    return dict(status=report['status'],protocol='rm_plan_v1',planner_calls=report['planner_calls'],
        installed_plans=len(plans),planner_wait_s=waits,root_preserved=True,subgoal_receipts=receipts,
        navigation_receipts=navigation_receipts,
        source_sha256={name:digest(run/name) for name in ('summary.json','trajectory.pt','events.jsonl',
            'goal-recovery-rows.json','planner-events.json','planner-history.json','rm-schedule.json','neural-skills.json')
            if (run/name).exists()},
        scope='Neural option integration; privileged server predicates; actual_llm_calls records provider use; no causal performance claim')
