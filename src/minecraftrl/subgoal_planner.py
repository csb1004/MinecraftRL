"""Observed option receipts over the existing root scheduler. No motor actions."""
from copy import deepcopy

from .replanning import (blocked_method, context, fingerprint, goal_satisfied, integer,
                         retry_scope, same_retry_conditions, validate_snapshot)
from .rm_planner import RMPlanScheduler, fixture_plan
from .subgoal_skills import SKILLS, completion


def motor_condition(cap):
    args=cap.get('arguments',{})
    return (cap['policy_ref'],args.get('embedding_sha256',args.get('explore_sha256',cap['skill_id'])),
            args.get('navigation_actor_sha256'))


def blocked_subgoal(current,cap,history,plans):
    if blocked_method(current,cap,history):return True
    receipts=[(r['method'],[r['before'],r['after']]) for r in history['attempts']
              if r['outcome'] in ('failed','partial') and r['committed_steps']]
    for plan in plans:
        for node in plan['nodes']:
            r=node.get('receipt')
            if r and r.get('type')=='neural_subgoal' and r['outcome']=='failed' and r['committed_steps']:
                receipts.append((r['method'],[r['before'],r['after']]))
    return any(motor_condition(method)==motor_condition(cap) and
               any(s is not None and same_retry_conditions(current,s) for s in snapshots)
               for method,snapshots in receipts)


def subgoal_fixture(packet):
    value=fixture_plan(packet)
    blocked={c['skill_id'] for c in packet['memory']['blocked_methods']}
    nav=packet['runtime_binding']['navigation']
    order=['leave_stalled_area','locate_wood','approach_wood']
    if nav['target_known']:order.remove('locate_wood')
    order=[name for name in order if name not in blocked]
    # A fixture does not invent an executor after the available options are exhausted.
    if not order:
        value.update(decision='pause',steps=[],reason='No untried registered condition in fixed fixture.')
        return value
    nodes=[]
    for name in order:
        nodes.append(dict(id=name,objective=SKILLS[name]['scope'],skill_id=name,priority=80,
            depends_on=[nodes[-1]['id']] if nodes else [],entry='active_root',
            completion=SKILLS[name]['completion'],budget_s=30.,budget_ticks=200,on_failure='replan'))
    root=value['steps'][-1];root['depends_on']=[nodes[-1]['id']]
    value['steps']=nodes+[root]
    return value


class SubgoalScheduler(RMPlanScheduler):
    def select(self,current,history):
        for _ in range(8):
            selected=super().select(current,history)
            node=selected['node']
            if node is None:return selected
            cap=self.state['plans'][-1]['registry'][node['skill_id']]
            if blocked_subgoal(current,cap,history,self.state['plans']):
                return self.pause('same_failed_condition')
            if node['skill_id'] not in SKILLS:return selected
            nav=current['navigation']
            if completion(node['skill_id'],nav,nav):
                self.finish_subgoal(node,current,current,reason='already_satisfied')
                continue
            if node['skill_id']=='approach_wood' and not nav['target_known']:
                return self.pause('approach_requires_known_target')
            return selected
        return self.pause('no_executable_subgoal')

    def finish_subgoal(self,node,before,after,*,reason):
        validate_snapshot(before);validate_snapshot(after)
        plan=self.state['plans'][-1]
        original=next(n for n in plan['proposal']['steps'] if n['id']==node['id'])
        count=after['sequence']-before['sequence']
        if (node!=original or node['skill_id'] not in SKILLS or self.state['active_node']!=node['id']
                or context(before)!=context(after) or not integer(count)
                or after['tick']-before['tick']!=count or after['elapsed_s']<before['elapsed_s']
                or reason not in ('predicate_satisfied','budget_exhausted','stop','environment_done','already_satisfied')):
            raise ValueError('unbound subgoal outcome')
        satisfied=completion(node['skill_id'],before['navigation'],after['navigation'])
        if (reason=='already_satisfied' and (count or not satisfied)) or (
                reason=='predicate_satisfied' and (not count or not satisfied)):
            raise ValueError('unproven subgoal completion')
        self.update_timing(after)
        interrupted=reason in ('stop','environment_done')
        outcome=('interrupted' if interrupted else 'completed' if satisfied else
                 'unrun' if not count else 'failed')
        receipt=dict(type='neural_subgoal',request_id=plan['proposal']['request_id'],node_id=node['id'],
            method=deepcopy(plan['registry'][node['skill_id']]),before=deepcopy(before),after=deepcopy(after),
            committed_steps=count,outcome=outcome,reason=reason,predicate_satisfied=satisfied,
            root_completed=goal_satisfied(after),already_satisfied=reason=='already_satisfied',
            classification='interaction_ready' if after['navigation']['interaction_ready'] else outcome,
            end_sequence=after['sequence'],execution_authorized=False,rm_effect='preserve')
        state=next(n for n in plan['nodes'] if n['id']==node['id'])
        state.update(status=outcome,receipt=receipt)
        self.state['active_node']=None;self.observe_root(after)
        return deepcopy(receipt)

    def request_after_failure(self,guard,snapshot):
        """Only this saved runtime receipt, not a model claim, can open another request."""
        validate_snapshot(snapshot)
        receipts=[n['receipt'] for n in self.state['plans'][-1]['nodes'] if n.get('receipt')]
        # Proposal listing order need not match dependency/execution order.
        receipt=max(receipts,key=lambda item:item['end_sequence'],default=None)
        if (not receipt or receipt.get('type') not in ('neural_subgoal','neural_waypoint') or receipt['outcome']!='failed'
                or not receipt['committed_steps'] or retry_scope(receipt['after'])!=retry_scope(snapshot)
                or not 0<=snapshot['tick']-receipt['after']['tick']<=4
                or snapshot['sequence']<receipt['after']['sequence']):
            raise ValueError('no fresh failed subgoal')
        request=next(r for r in guard.history.data['requests'] if r['request_id']==receipt['request_id'])
        if not request['consumed']:raise ValueError('subgoal request was not reviewed')
        evidence=dict(event_id=fingerprint(receipt)+':subgoal-ended',event='subgoal_ended',
            observation_valid=True,tick=receipt['after']['tick'],reason=receipt['reason'],
            source_request_id=receipt['request_id'],node_id=receipt['node_id'],outcome='failed',
            receipt_sha256=fingerprint(receipt))
        return guard._request(snapshot,evidence)
