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
    packets={};plans={};waits=[]
    for e in events:
        if e['event']=='wait_started':
            p=e['packet'];validate_packet(p);verify_snapshot(p['runtime_binding'])
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
                if r.get('type')=='neural_subgoal':assert r['outcome']=='completed'
                else:assert r['classification'] in ('target_acquired','interaction_ready') and r['end_sequence']==step
            if node['id']=='root' and node['depends_on'] and any(
                    n['id'] in node['depends_on'] and n['skill_id'] in SKILLS for n in plan['proposal']['steps']):
                assert current['navigation']['interaction_ready']
    receipts=[]
    for revision,plan in enumerate(schedule['plans'],1):
        for n in plan['nodes']:
            r=n.get('receipt')
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
        assert row['subgoal_context']['skill_id']==node['skill_id']
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
    return dict(status=report['status'],protocol='rm_plan_v1',planner_calls=report['planner_calls'],
        installed_plans=len(plans),planner_wait_s=waits,root_preserved=True,subgoal_receipts=receipts,
        source_sha256={name:digest(run/name) for name in ('summary.json','trajectory.pt','events.jsonl',
            'goal-recovery-rows.json','planner-events.json','planner-history.json','rm-schedule.json','neural-skills.json')
            if (run/name).exists()},
        scope='Neural option integration; privileged server predicates; actual_llm_calls records provider use; no causal performance claim')
