"""Independent inventory/transition/dependency checks for the new plan protocol.

The caller first audits every neural log-probability, official action, reward/RM
transition, server tick and the STOP/cleanup boundary using the existing auditor.
"""
from copy import deepcopy
import json

from minecraftrl.recovery_evidence import assess_attempt
from minecraftrl.replanning import blocked_method,fingerprint,goal_satisfied,make_snapshot
from minecraftrl.rm_planner import validate_plan,validate_packet
from minecraftrl.stall_replay import log_count
from minecraftrl.world_store import digest


def verify_schedule(run,report,rows,trajectory,history,events):
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    schedule=read('rm-schedule.json');start=read('initial-state.json')['server']
    root=schedule['root'];packets={};plans={};waits=[]
    assert root['baseline']==log_count(start)
    assert root['progress']==max(0,log_count(rows[-1]['server_after'])-log_count(start))
    assert root['completed']==report['server_target_reached']
    assert schedule['active_node'] is None
    assert len(schedule['plans'])<=report['planner_calls']<=report['condition']['planning']['maximum_calls']
    def at(step):return rows[step-1]['server_after'] if step else start
    def snapshot(s,step):
        return make_snapshot(at(step),world_id=s['world_id'],episode_id=s['episode_id'],goal='wood',task=None,
            generation=s['generation'],sequence=step,elapsed_s=s['elapsed_s'],status=s['status'],
            release_sequence=s['release_sequence'],held_key_count=at(step)['held_key_count'],baseline_count=root['baseline'])
    for event in events:
        kind=event['event'];step=event['step']
        if kind=='wait_started':
            packet=event['packet'];validate_packet(packet)
            assert packet['runtime_binding']==snapshot(packet['runtime_binding'],step)
            request=next(r for r in history['requests'] if r['request_id']==packet['request_id'])
            assert packet['runtime_binding']==request['snapshot']
            assert packet['execution_registry']==request['capabilities']
            packets[packet['request_id']]=packet
            assert read(f'planner-input-{len(packets):02d}.json')==packet
        elif kind=='response_received':
            value=event['reply'];validate_plan(value,packets[value['request_id']]);waits.append(event['wait_s'])
        elif kind=='plan_installed':
            value=event['plan'];validate_plan(value,packets[value['request_id']])
            assert event['current']==snapshot(event['current'],step)
            revision=event['schedule_revision'];plans[revision]=value
            saved=schedule['plans'][revision-1]
            assert saved['proposal']==value and saved['applied_snapshot']==event['current']
            assert saved['applied_snapshot']['goal_ref']==root['reference']
        elif kind=='node_selected':
            selected=event['selection'];current=event['current']
            assert current==snapshot(current,step)
            value=plans[event['schedule_revision']]
            if selected['node'] is None:
                if selected['reason']=='missing_capability':
                    assert any(n['skill_id'] not in packets[value['request_id']]['execution_registry'] for n in value['steps'])
                continue
            node=selected['node'];assert node in value['steps']
            cap=packets[value['request_id']]['execution_registry'][node['skill_id']]
            past=deepcopy(history)
            past['failures']=[f for f in history['failures'] if f['snapshot']['sequence']<=step]
            past['attempts']=[a for a in history['attempts'] if a['after'] and a['after']['sequence']<=step]
            assert not blocked_method(current,cap,past)
            # Actual recorded subgoal boundary must precede a dependent root selection.
            if node['depends_on']:
                receipts=[a for a in report['recovery_attempts'] if a['request_id']==value['request_id'] and a['end_step']==step]
                assert node['id']=='root' and len(receipts)==1
                assert receipts[0]['assessment']['classification'] in ('target_acquired','interaction_ready')
    assert len(packets)==report['planner_calls'] and len(plans)==len(schedule['plans'])
    for row,saved in zip(rows,trajectory['rows']):
        tag=row['rm_plan'];assert tag==saved['rm_plan'] and tag['root_reference']==root['reference']
        if row['owner']=='planner_neutral':continue
        if tag['revision']==0:assert tag['node_id']=='root' and row['mode']=='wood';continue
        node=next(n for n in plans[tag['revision']]['steps'] if n['id']==tag['node_id'])
        assert node['skill_id'] in ('collect_wood','explore_then_resume')
        assert row['mode']==('wood' if node['id']=='root' else 'intervention')
    for receipt in report['recovery_attempts']:
        window=rows[receipt['start_step']-1:receipt['end_step']]
        if not window:continue  # STOP before the first dispatch is checked by the common auditor.
        revision=window[0]['rm_plan']['revision'];plan=plans[revision]
        node=next(n for n in plan['steps'] if n['id']==window[0]['rm_plan']['node_id'])
        result=assess_attempt(window[0]['server_before'],[r['server_after'] for r in window],
            reason=receipt['assessment'].get('boundary_reason','stop'),baseline_count=root['baseline'],
            budget_ticks=node['budget_ticks'])
        assert result==receipt['assessment'] and receipt['committed_transitions']==len(window)
        assert all(r['mode']=='intervention' for r in window)
        fresh=[i for i,r in enumerate(window,1) if r['server_before']['wood_v3_kind']==0 and r['server_after']['wood_v3_kind']==1]
        if fresh:assert fresh==[len(window)]
    from PIL import Image
    import hashlib
    import numpy as np
    for switch in report['switches']:
        assert switch['committed']==(switch['before_action']<=len(rows))
        assert hashlib.sha256(np.asarray(Image.open(run/switch['frame'])).tobytes()).hexdigest()==switch['input_image_sha256']
    return dict(status=report['status'],protocol='rm_plan_v1',planner_calls=report['planner_calls'],
        installed_plans=len(plans),planner_wait_s=waits,root_preserved=True,
        unsupported_nodes=sum(n['support']=='unsupported' for p in schedule['plans'] for n in p['nodes']),
        recovery_attempts=report['recovery_attempts'],
        source_sha256={name:digest(run/name) for name in ('summary.json','trajectory.pt','events.jsonl',
            'goal-recovery-rows.json','planner-events.json','planner-history.json','rm-schedule.json') if (run/name).exists()},
        scope='Actual fixed-plan integration and neural actions; not LLM generation or learning improvement')
