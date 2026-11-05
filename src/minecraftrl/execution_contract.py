"""Caller-owned task meaning and safe boundaries for future asynchronous clients."""
from copy import deepcopy
import re

from .goal_contract import digest, fields
from .goal_spec import assert_goal, policy_condition


def execution_request(queue, phase, task):
    contract=assert_goal(queue,queue.active)
    if contract is None or 'effect' not in contract:
        raise ValueError('execution requests require goal_spec_v2')
    record=queue.records[queue.active]
    request=dict(schema='execution_request_v1',goal_id=queue.active,revision=record['revision'],
        semantic_sha256=contract['sha256'],source_sha256=queue.goal_spec['source_sha256'],
        effect=contract['effect'],target_ref=contract['target_ref'],items=contract['items'],
        quantity=contract['quantity'],baseline=record['baseline'],preserve=contract['preserve'],
        prerequisites=contract['prerequisites'],phase=phase,task=task,
        policy_condition=policy_condition(queue,phase,task),
        completion_owner='caller',observation_contract='fixed_server_inventory_and_masked_phase_v1')
    return dict(deepcopy(request),request_sha256=digest(request))


class BoundaryUpdates:
    """Single-owner mailbox. Inference never observes a partially applied revision."""
    def __init__(self):
        self.pending=[];self.seen=set();self.events=[]

    def submit(self, request):
        if not isinstance(request,dict) or request.get('kind')!='reprioritize':
            raise ValueError('semantic changes require a new run')
        fields(request,'request_id kind goal_id expected_revision priority','boundary update')
        for key in ('request_id','goal_id'):
            if not isinstance(request[key],str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',request[key]):
                raise ValueError('invalid boundary identity')
        if request['request_id'] in self.seen:raise ValueError('duplicate boundary request')
        if type(request['expected_revision']) is not int or request['expected_revision']<1:
            raise ValueError('invalid expected revision')
        if type(request['priority']) is not int or not 0<=request['priority']<=100:
            raise ValueError('invalid boundary priority')
        self.seen.add(request['request_id']);self.pending.append(deepcopy(request))

    def apply(self, queue, *, inference_pending):
        if inference_pending:return []
        results=[]
        while self.pending:
            request=self.pending.pop(0)
            try:
                queue.reprioritize(request['goal_id'],request['priority'],request['expected_revision'])
                result=dict(request,status='applied',tick=queue.last_tick)
            except (ValueError,KeyError) as exc:
                result=dict(request,status='rejected',reason=str(exc),tick=queue.last_tick)
            results.append(result);self.events.append(result)
        return deepcopy(results)


def classify_failure(evidence):
    """An observation of failure is not authorization or proof that RL is needed."""
    reason=evidence.get('reason','unknown')
    if reason in ('cancelled','stop_file','user_stop'):category='interrupted'
    elif evidence.get('binding_valid') is False:category='contract_mismatch'
    elif evidence.get('resource_interference') is True:category='execution_resource'
    elif evidence.get('observations_valid') is False:category='observation_unavailable'
    elif reason in ('unsupported_skill','skill_target_mismatch'):category='unsupported'
    elif reason=='preservation_violated':category='constraint_violation'
    elif reason in ('missing_prerequisite_items','missing_preserved_items'):category='precondition_missing'
    elif evidence.get('execution_valid') is False:category='execution_error'
    else:category='unresolved_stall'
    next_steps={
        'interrupted':['remain_stopped'],'contract_mismatch':['repair_contract'],
        'execution_resource':['check_resources'],'observation_unavailable':['verify_available_evidence'],
        'unsupported':['declare_capability_gap'],'constraint_violation':['review_constraints_and_policy'],
        'precondition_missing':['plan_prerequisites'],'execution_error':['diagnose_runtime'],
        'unresolved_stall':['compare_goal_condition','inspect_skill_evidence','propose_bounded_training'],
    }
    return dict(schema='failure_evidence_v1',category=category,reason=reason,evidence=deepcopy(evidence),
        next_steps=next_steps[category],automatic_training_authorized=False,
        capability_failure_confirmed=False)
