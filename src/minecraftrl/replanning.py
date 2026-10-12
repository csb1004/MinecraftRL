"""Bounded, observation-only replanning contracts. Never dispatches game input."""
import copy
import hashlib
import json
import math
from pathlib import Path
import uuid

from .memory import atomic_json, file_lock
from .tasks import GOALS, TaskState


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def collection_capability(goal, policy_ref):
    if goal not in GOALS or not isinstance(policy_ref, str) or not 1 <= len(policy_ref) <= 256:
        raise ValueError('invalid collection binding')
    return dict(skill_id='collect_' + goal, goal=goal, policy_ref=policy_ref,
                arguments={}, max_budget_s=60., max_budget_ticks=600,
                completion='authoritative_goal_inventory',
                scope='collection only; no target selection or detour control')


def goal_conditioning_capability(policy_ref, wood_sha256, explore_sha256):
    """Opt-in experimental condition switch, never a general navigation skill."""
    collection_capability('wood',policy_ref)
    for value in (wood_sha256, explore_sha256):
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('goal conditioning requires actual embedding fingerprints')
    return dict(skill_id='explore_then_resume',goal='wood',policy_ref=policy_ref,
        arguments=dict(wood_sha256=wood_sha256,explore_sha256=explore_sha256),
        max_budget_s=30.,max_budget_ticks=200,completion='authoritative_goal_inventory',
        scope='opt-in single exploration condition window; no directional or guaranteed detour skill')


def make_snapshot(obs, *, world_id, episode_id, goal, task, generation, sequence, elapsed_s,
                  status, release_sequence=None, held_key_count=None, baseline_count=0):
    """Trusted runtime/replay adapter. No fixture geometry or dialogue facts."""
    if obs.get('server_available') != 1 or obs.get('error', 0) != 0:
        raise ValueError('server observation unavailable')
    if goal not in GOALS:
        raise ValueError('unsupported goal')
    position = [obs['server_' + axis] for axis in 'xyz']
    if any(not number(v, -30000000) for v in position):
        raise ValueError('invalid server position')
    kind = obs['wood_v3_kind'] if goal == 'wood' else 0
    if kind not in (0,1,2) or obs.get('server_alive') not in (0,1):
        raise ValueError('invalid target/life observation')
    target = [obs['wood_v3_target_' + axis] for axis in 'xyz'] if kind in (1, 2) else None
    if target is not None and any(not number(v, -30000000) or int(v) != v for v in target):
        raise ValueError('invalid observed target')
    inventory = [obs['server_inventory'].get(k, 0) for k in GOALS[goal][1]]
    if any(not number(v) or int(v) != v for v in inventory):
        raise ValueError('invalid goal inventory')
    if not number(obs['server_tick']) or int(obs['server_tick']) != obs['server_tick']:
        raise ValueError('invalid server tick')
    if held_key_count is not None:
        if not number(held_key_count) or int(held_key_count)!=held_key_count:
            raise ValueError('invalid monitor key count')
        held_key_count=int(held_key_count)  # MineRL numeric monitor fields are floats.
    task = copy.deepcopy(task)
    if task is not None:
        if task['goal'] != goal:
            raise ValueError('task and policy goal disagree')
        goal_ref = dict(kind='task',task_id=task['task_id'],revision=task['revision'])
        baseline_count = task['baseline']
    else:
        goal_ref = dict(kind='episode',episode_id=episode_id,goal=goal)
    snapshot = dict(world_id=world_id,episode_id=episode_id,goal=goal,goal_ref=goal_ref,task=task,
                    generation=generation,sequence=sequence,tick=int(obs['server_tick']),elapsed_s=elapsed_s,
                    status=status,release_sequence=release_sequence,held_key_count=held_key_count,
                    baseline_count=baseline_count,observation_valid=True,
                    facts=dict(region=[math.floor(v/4.) for v in position],target=target,
                               selected_item=obs['selected_item'],goal_inventory=int(sum(inventory)),
                               alive=obs.get('server_alive') == 1))
    resources = obs['server_inventory']
    if any(not isinstance(k,str) or not number(v) or int(v)!=v for k,v in resources.items()):
        raise ValueError('invalid resource inventory')
    snapshot['facts']['resources'] = {k:int(v) for k,v in resources.items() if v}
    reachable = obs.get('wood_v3_reachable') if goal=='wood' else None
    if reachable is not None and reachable not in (0,1):
        raise ValueError('invalid target reachability')
    snapshot['facts']['target_reachable'] = None if reachable is None else bool(reachable)
    validate_snapshot(snapshot)
    return snapshot


def validate_snapshot(s):
    required = {'world_id','episode_id','goal','goal_ref','task','generation','sequence','tick','elapsed_s',
                'status','release_sequence','held_key_count','baseline_count','observation_valid','facts'}
    if not isinstance(s, dict) or not required<=set(s) or set(s)-required-{'navigation'}:
        raise ValueError('invalid snapshot fields')
    for key in ('world_id','episode_id'):
        if not isinstance(s[key], str) or not 1 <= len(s[key]) <= 256:
            raise ValueError('invalid context identity')
    if s['goal'] not in GOALS or s['status'] not in (
            'running','stalled','user_paused','cancelled','completed','superseded','stopped'):
        raise ValueError('invalid goal/status')
    if (any(not integer(s[k]) for k in ('sequence','tick','baseline_count')) or
            (s['generation'] is not None and not integer(s['generation'])) or not number(s['elapsed_s'])):
        raise ValueError('invalid observation timing')
    if type(s['observation_valid']) is not bool:
        raise ValueError('invalid observation validity')
    if any(s[k] is not None and not integer(s[k]) for k in ('release_sequence','held_key_count')):
        raise ValueError('invalid release evidence')
    facts = s['facts']
    base_facts = {'region','target','selected_item','goal_inventory','alive'}
    if (not isinstance(facts, dict) or not base_facts <= set(facts)
            or set(facts)-base_facts-{'resources','target_reachable'}):
        raise ValueError('invalid observed facts')
    if not isinstance(facts['region'], list) or len(facts['region']) != 3 or any(type(v) is not int for v in facts['region']):
        raise ValueError('invalid region')
    target = facts['target']
    if target is not None and (not isinstance(target, list) or len(target) != 3 or
                              any(not number(v, -30000000) or int(v) != v for v in target)):
        raise ValueError('invalid target')
    if not isinstance(facts['selected_item'], str) or not 1 <= len(facts['selected_item']) <= 128:
        raise ValueError('invalid selected item')
    if not integer(facts['goal_inventory']) or type(facts['alive']) is not bool:
        raise ValueError('invalid inventory/life facts')
    if 'resources' in facts:
        resources=facts['resources']
        if (not isinstance(resources,dict) or len(resources)>128 or any(
                not isinstance(k,str) or not 1<=len(k)<=128 or not integer(v,1) for k,v in resources.items())):
            raise ValueError('invalid resource facts')
    if facts.get('target_reachable') is not None and type(facts['target_reachable']) is not bool:
        raise ValueError('invalid reachability facts')
    if s['task'] is not None:
        task = s['task']
        if not isinstance(task,dict):
            raise ValueError('invalid task snapshot')
        schema = 3 if 'failure' in task else 2 if 'plan' in task else 1
        TaskState.restore(dict(schema=schema,world_id=s['world_id'],task=task,last_observation=s['sequence']),s['world_id'])
    expected = dict(kind='episode',episode_id=s['episode_id'],goal=s['goal']) if s['task'] is None else dict(
        kind='task',task_id=s['task']['task_id'],revision=s['task']['revision'])
    if s['goal_ref'] != expected or (s['task'] is not None and s['task']['goal'] != s['goal']):
        raise ValueError('invalid goal reference')
    if 'navigation' in s:
        from .subgoal_skills import validate_navigation
        nav=validate_navigation(s['navigation'])
        if (s['goal']!='wood' or nav['tick']!=s['tick'] or nav['target']!=facts['target']
                or nav['inventory_gain']!=facts['goal_inventory']-s['baseline_count']
                or [math.floor(v/4.) for v in nav['position']]!=facts['region']
                or nav['reachable']!=bool(nav['kind'] and facts.get('target_reachable'))):
            raise ValueError('abstract navigation is not bound to snapshot')
    fingerprint(s)  # Reject NaN/non-JSON values in receipts as well.


def active(s):
    task = s['task']
    if s['status'] not in ('running','stalled') or not s['facts']['alive']:
        return False
    if task is not None:
        return task['status'] == 'running' or (task['status'] == 'paused' and
                                               task.get('failure',{}).get('reason') == 'no_progress')
    return True


def context(s):
    return {k:s[k] for k in ('world_id','episode_id','goal_ref','generation')}


def goal_satisfied(s):
    count = s['task']['target_count'] if s['task'] is not None else 1
    return s['facts']['goal_inventory'] - s['baseline_count'] >= count


def failed_attempt_key(s, capability):
    return fingerprint(dict(world=s['world_id'],episode=s['episode_id'],goal=s['goal_ref'],
                            skill=capability['skill_id'],policy=capability['policy_ref'],facts=s['facts']))


def retry_scope(s):
    # A pause increments TaskState.revision without changing the user's goal.
    # Keep the original revision in the receipt, but not in retry equivalence.
    goal = s['goal_ref'] if s['task'] is None else dict(kind='task',task_id=s['task']['task_id'],goal=s['goal'])
    return dict(world=s['world_id'],episode=s['episode_id'],goal=goal)


def method_key(capability):
    return fingerprint({k:capability.get(k,{}) for k in ('skill_id','policy_ref','arguments')})


def same_retry_conditions(a,b):
    if retry_scope(a)!=retry_scope(b):return False
    if any(a['facts'][k]!=b['facts'][k] for k in ('target','goal_inventory','alive')):return False
    # Missing legacy evidence cannot prove that a retry has new preconditions.
    return all(a['facts'].get(k) is None or b['facts'].get(k) is None or a['facts'][k]==b['facts'][k]
               for k in ('resources','target_reachable'))


def blocked_method(s,capability,data):
    key=method_key(capability)
    for row in data['failures']:
        same_method=(row['skill_id']==capability['skill_id'] and row['policy_ref']==capability['policy_ref']
                     and ('arguments' not in row or method_key(row)==key))
        if same_method and same_retry_conditions(s,row['snapshot']):return True
    for row in data['attempts']:
        if row['outcome'] not in ('failed','partial') or not row['committed_steps']:continue
        if method_key(row['method'])==key and any(same_retry_conditions(s,x) for x in (row['before'],row['after']) if x):
            return True
    return False


def validate_attempt(row,requests,decisions):
    if set(row)!={'request_id','method','before','after','outcome','reason','committed_steps'}:
        raise ValueError('invalid attempt receipt fields')
    request=next((r for r in requests if r['request_id']==row['request_id']),None)
    decision=next((d for d in decisions if d['request_id']==row['request_id']),None)
    if (not request or not decision or not decision['accepted'] or not isinstance(row['method'],dict)
            or decision.get('method')!=row['method']):
        raise ValueError('attempt lacks accepted method review')
    before,after=row['before'],row['after'];validate_snapshot(before)
    if context(before)!=context(request['snapshot']) or before['facts']!=request['snapshot']['facts']:
        raise ValueError('attempt differs from reviewed context')
    if any(before[k]<request['snapshot'][k] for k in ('sequence','tick','elapsed_s')):
        raise ValueError('attempt predates review')
    if (row['outcome'] not in ('failed','partial','completed','interrupted','unverified')
            or not isinstance(row['reason'],str) or not 1<=len(row['reason'])<=160
            or not integer(row['committed_steps'])):
        raise ValueError('invalid attempt outcome')
    if row['committed_steps']==0 and row['outcome'] not in ('interrupted','unverified'):
        raise ValueError('unexecuted attempt is not a failure or success')
    if row['committed_steps']>row['method']['max_budget_ticks']:
        raise ValueError('attempt exceeds method tick budget')
    if after is None:
        if row['outcome']!='unverified':raise ValueError('missing attempt end evidence')
    else:
        validate_snapshot(after)
        if (context(after)!=context(before) or after['sequence']-before['sequence']!=row['committed_steps']
                or after['tick']<before['tick'] or after['elapsed_s']<before['elapsed_s']):
            raise ValueError('invalid attempt observation interval')
        if ((row['outcome']=='completed' and not goal_satisfied(after))
                or (row['outcome'] in ('failed','partial') and goal_satisfied(after))):
            raise ValueError('completion must match authoritative inventory')
    fingerprint(row)


def memory_facts(s):
    facts=copy.deepcopy(s['facts'])
    resources=facts.pop('resources',None)
    if resources is not None:
        facts.update(resources=dict(sorted(resources.items())[:8]),
                     omitted_resources=max(0,len(resources)-8),resources_sha256=fingerprint(resources))
    return facts


class PlanHistory:
    """Single-session sidecar; optimistic revision guard plus native file lock."""
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.data = self._read()

    def _read(self):
        empty = dict(schema=2,revision=0,failures=[],requests=[],decisions=[],attempts=[])
        if self.path is None or not self.path.exists():
            return empty
        if self.path.stat().st_size > 524288:
            raise ValueError('replanning sidecar too large')
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if isinstance(data,dict) and data.get('schema')==1 and set(data)==set(empty)-{'attempts'}:
            data.update(schema=2,attempts=[])  # Read-only upgrade; original archives stay untouched.
        if not isinstance(data,dict) or set(data) != set(empty) or data['schema'] != 2 or not integer(data['revision']):
            raise ValueError('invalid replanning sidecar')
        for field, limit in (('failures',64),('requests',3),('decisions',64),('attempts',3)):
            if not isinstance(data[field],list) or len(data[field]) > limit or any(not isinstance(x,dict) for x in data[field]):
                raise ValueError('invalid replanning history')
        ids = set()
        for request in data['requests']:
            if set(request) != {'request_id','snapshot','evidence','capabilities','ttl_s','consumed'}:
                raise ValueError('invalid saved request')
            validate_snapshot(request['snapshot'])
            if request['request_id'] in ids or type(request['consumed']) is not bool or request['ttl_s'] != 30.:
                raise ValueError('invalid saved request state')
            ids.add(request['request_id'])
        for failure in data['failures']:
            fields={'key','event_id','snapshot','skill_id','policy_ref'}
            if not fields<=set(failure) or set(failure)-fields-{'evidence','arguments'}:
                raise ValueError('invalid failure receipt')
            validate_snapshot(failure['snapshot'])
            cap=dict(skill_id=failure['skill_id'],policy_ref=failure['policy_ref'])
            if failure['key'] != failed_attempt_key(failure['snapshot'],cap):
                raise ValueError('failure receipt identity mismatch')
            if 'evidence' in failure:
                ReplanGuard._evidence(failure['snapshot'],failure['evidence'])
            if 'arguments' in failure:
                if failure['skill_id']=='explore_then_resume':
                    bound=goal_conditioning_capability(failure['policy_ref'],**failure['arguments'])
                else:
                    bound=collection_capability(failure['snapshot']['goal'],failure['policy_ref'])
                if bound['skill_id']!=failure['skill_id'] or bound['arguments']!=failure['arguments']:
                    raise ValueError('invalid saved failure method')
        consumed = set()
        for decision in data['decisions']:
            if (decision.get('request_id') not in ids or decision['request_id'] in consumed or
                    decision.get('execution_authorized') is not False or type(decision.get('accepted')) is not bool or
                    not isinstance(decision.get('reasons'),list) or decision['accepted'] != (not decision['reasons'])):
                raise ValueError('invalid saved decision')
            consumed.add(decision['request_id'])
            if 'method' in decision:
                request=next(r for r in data['requests'] if r['request_id']==decision['request_id'])
                method=decision['method']
                if (not decision['accepted'] or not isinstance(method,dict)
                        or request['capabilities'].get(method.get('skill_id'))!=method):
                    raise ValueError('saved method differs from requested capability')
        if any(request['consumed'] != (request['request_id'] in consumed) for request in data['requests']):
            raise ValueError('request consumption receipt mismatch')
        attempts=set()
        for row in data['attempts']:
            validate_attempt(row,data['requests'],data['decisions'])
            if row['request_id'] in attempts:raise ValueError('duplicate execution receipt')
            attempts.add(row['request_id'])
        return data

    def snapshot(self):
        return copy.deepcopy(self.data)

    def commit(self, data):
        value = copy.deepcopy(data)
        value['revision'] = self.data['revision'] + 1
        if any(len(value[k]) > limit for k,limit in (('failures',64),('requests',3),('decisions',64),('attempts',3))):
            raise ValueError('replanning history capacity exhausted')
        if len(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'))>524288:
            raise ValueError('replanning sidecar too large')
        if self.path is not None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with file_lock(self.path.with_suffix('.lock')):
                if self._read()['revision'] != self.data['revision']:
                    raise ValueError('stale history writer')
                atomic_json(self.path,value)
        self.data = value


class ReplanGuard:
    """Checks proposals only. Every verdict has execution_authorized=False."""
    def __init__(self, history, capabilities, *, allow_goal_conditioning=False, allow_subgoals=False):
        self.history = history
        self.capabilities = copy.deepcopy(capabilities)
        for name,cap in self.capabilities.items():
            from .subgoal_skills import SKILLS, validate_capability
            if allow_subgoals and name in SKILLS:
                expected=validate_capability(cap)
            elif allow_goal_conditioning and name=='explore_then_resume':
                expected=goal_conditioning_capability(cap['policy_ref'],**cap['arguments'])
            else:
                expected=collection_capability(cap['goal'],cap['policy_ref'])
            if cap != expected or name != cap['skill_id']:
                raise ValueError('unsupported capability binding')

    @staticmethod
    def _evidence(s, e):
        if (not isinstance(e,dict) or e.get('event') != 'stalled' or e.get('stalled') is not True or
                e.get('observation_valid') is not True or not number(e.get('tick')) or
                not 0 <= s['tick'] - e['tick'] <= 4 or
                e.get('reason') not in ('no_progress','no_goal_progress','stationary_repetition','repeated_cycle') or
                not isinstance(e.get('event_id'),str) or not 1 <= len(e['event_id']) <= 256):
            raise ValueError('unproven stagnation event')
        result={k:copy.deepcopy(e[k]) for k in ('event_id','event','stalled','observation_valid','tick','reason')}
        for key in ('no_progress_ticks','no_goal_progress_ticks','window_ticks','action_fraction'):
            if key in e:
                if not number(e[key]) or (key=='action_fraction' and e[key]>1):
                    raise ValueError('invalid stagnation measurement')
                result[key]=e[key]
        return result

    def record_failure(self, snapshot, event, skill_id):
        validate_snapshot(snapshot)
        evidence = self._evidence(snapshot,event)
        cap = self.capabilities[skill_id]
        if cap['goal'] != snapshot['goal']:
            raise ValueError('failed skill goal mismatch')
        row = dict(key=failed_attempt_key(snapshot,cap),event_id=evidence['event_id'],
                   snapshot=copy.deepcopy(snapshot),skill_id=skill_id,policy_ref=cap['policy_ref'],
                   arguments=copy.deepcopy(cap['arguments']),evidence=evidence)
        data = self.history.snapshot()
        existing = [x for x in data['failures'] if x['event_id'] == row['event_id']]
        if existing:
            if len(existing)!=1 or any(existing[0].get(k)!=v for k,v in row.items()
                                      if k in existing[0] or k not in ('evidence','arguments')):
                raise ValueError('failure event reused with different content')
            return
        data['failures'].append(row)
        self.history.commit(data)

    def request(self, snapshot, event):
        validate_snapshot(snapshot)
        evidence = self._evidence(snapshot,event)
        return self._request(snapshot,evidence)

    def request_after_attempt(self,snapshot,previous_request_id):
        """An observed failed skill boundary may replan without inventing a stall."""
        validate_snapshot(snapshot)
        row=next((a for a in self.history.data['attempts'] if a['request_id']==previous_request_id),None)
        if (row is None or row['outcome'] not in ('failed','partial') or row['after'] is None
                or not row['committed_steps'] or retry_scope(row['after'])!=retry_scope(snapshot)
                or not 0<=snapshot['tick']-row['after']['tick']<=4
                or snapshot['sequence']<row['after']['sequence']):
            raise ValueError('no fresh committed failed attempt for replanning')
        evidence=dict(event_id=previous_request_id+':attempt-ended',event='attempt_ended',
            observation_valid=True,tick=row['after']['tick'],reason=row['reason'],
            source_request_id=previous_request_id,outcome=row['outcome'])
        return self._request(snapshot,evidence)

    def _request(self,snapshot,evidence):
        if not active(snapshot) or goal_satisfied(snapshot) or not snapshot['observation_valid'] or snapshot['elapsed_s'] >= 180.:
            raise ValueError('inactive or expired planning context')
        data = self.history.snapshot()
        for previous in data['requests']:
            if previous['evidence']['event_id'] == evidence['event_id']:
                if previous['snapshot'] != snapshot:
                    raise ValueError('request event reused with different observation')
                return copy.deepcopy(previous)
        if len(data['requests']) >= 3:
            raise ValueError('planning request budget exhausted')
        request = dict(request_id=uuid.uuid4().hex,snapshot=copy.deepcopy(snapshot),evidence=evidence,
                       capabilities=copy.deepcopy(self.capabilities),ttl_s=30.,consumed=False)
        data['requests'].append(request)
        self.history.commit(data)
        return copy.deepcopy(request)

    def planner_input(self, request_id):
        """Bounded data for a provider; no dialogue or executable code is forwarded."""
        matches = [r for r in self.history.data['requests'] if r['request_id'] == request_id]
        if not matches:
            raise ValueError('unknown planning request')
        request = copy.deepcopy(matches[0])
        options = [cap for cap in request['capabilities'].values()
                   if cap['goal'] == request['snapshot']['goal'] and
                   not blocked_method(request['snapshot'],cap,self.history.data)]
        relevant = [row for row in self.history.data['failures']
                    if retry_scope(row['snapshot']) == retry_scope(request['snapshot'])]
        attempts=[row for row in self.history.data['attempts']
                  if retry_scope(row['before'])==retry_scope(request['snapshot'])]
        failures=[dict(event_id=r['event_id'],skill_id=r['skill_id'],policy_ref=r['policy_ref'],
                       arguments=copy.deepcopy(r.get('arguments')),
                       facts=memory_facts(r['snapshot']),evidence=copy.deepcopy(r.get('evidence')))
                  for r in relevant[-4:]]
        receipts=[dict(request_id=r['request_id'],method=copy.deepcopy(r['method']),outcome=r['outcome'],
                       reason=r['reason'],committed_steps=r['committed_steps'],
                       before=memory_facts(r['before']),after=memory_facts(r['after']) if r['after'] else None)
                  for r in attempts]
        return dict(request_id=request_id,snapshot=request['snapshot'],evidence=request['evidence'],
                    available_skills=options,failed_attempts=failures,attempts=receipts,
                    omitted_failures=max(0,len(relevant)-4),
                    blocked_methods=[copy.deepcopy(cap) for cap in request['capabilities'].values()
                                     if cap['goal']==request['snapshot']['goal'] and cap not in options],
                    call_model=bool(options) and not request['consumed'],
                    blocked_reason=None if options else 'no_available_alternative',
                    instruction='Return a proposal using only listed skills and evidence. Preserve the existing goal and RM. '
                                'Do not claim completion, set rewards, generate code or key inputs. '
                                'No available alternative means stay paused; a proposal is never execution permission.')

    def record_attempt(self,request_id,before,after,*,outcome,reason,committed_steps):
        data=self.history.snapshot()
        decision=next((d for d in data['decisions'] if d['request_id']==request_id),{})
        row=dict(request_id=request_id,method=copy.deepcopy(decision.get('method')),
                 before=copy.deepcopy(before),after=copy.deepcopy(after),outcome=outcome,
                 reason=reason,committed_steps=committed_steps)
        validate_attempt(row,data['requests'],data['decisions'])
        existing=[r for r in data['attempts'] if r['request_id']==request_id]
        if existing:
            if existing!=[row]:raise ValueError('execution receipt cannot be rewritten')
            return copy.deepcopy(row)
        data['attempts'].append(row);self.history.commit(data)
        return copy.deepcopy(row)

    def review(self, request_id, proposal, current, *, now_elapsed_s):
        data = self.history.snapshot()
        matches = [r for r in data['requests'] if r['request_id'] == request_id]
        reasons = []
        if not matches:
            return dict(accepted=False,execution_authorized=False,reasons=['unknown_request'])
        request = matches[0]
        if request['consumed']:
            return dict(accepted=False,execution_authorized=False,reasons=['request_consumed'])
        request['consumed'] = True
        try:
            validate_snapshot(current)
        except (ValueError,KeyError,TypeError):
            reasons.append('invalid_current_observation')
        else:
            old = request['snapshot']
            if not number(now_elapsed_s) or now_elapsed_s < current['elapsed_s']:
                reasons.append('invalid_review_clock')
            if not active(current):
                reasons.append('inactive_goal')
            if current['generation'] is None:
                reasons.append('unproven_generation')
            if context(current) != context(old):
                reasons.append('stale_context')
            if not current['observation_valid'] or any(current[k] < old[k] for k in ('sequence','tick','elapsed_s')):
                reasons.append('stale_observation')
            if current['facts'] != old['facts']:
                reasons.append('changed_preconditions')
            if number(now_elapsed_s) and now_elapsed_s - old['elapsed_s'] >= request['ttl_s']:
                reasons.append('request_expired')
            if (current['release_sequence'] is None or current['release_sequence'] < current['sequence'] or
                    current['held_key_count'] != 0):
                reasons.append('inputs_not_released')
            if goal_satisfied(current):
                reasons.append('goal_already_satisfied')
        fields = {'request_id','skill_id','arguments','budget_s','budget_ticks','evidence_ids','reason'}
        valid = (isinstance(proposal,dict) and set(proposal) == fields and
                 isinstance(proposal['skill_id'],str) and isinstance(proposal['reason'],str) and
                 1 <= len(proposal['reason']) <= 240 and number(proposal['budget_s']) and proposal['budget_s'] > 0 and
                 integer(proposal['budget_ticks'],1))
        if not valid:
            reasons.append('invalid_proposal')
        else:
            if proposal['request_id'] != request_id:
                reasons.append('wrong_request')
            if proposal['evidence_ids'] != [request['evidence']['event_id']]:
                reasons.append('unproven_evidence')
            cap = self.capabilities.get(proposal['skill_id'])
            if cap is None:
                reasons.append('unsupported_skill')
            else:
                if request['capabilities'].get(proposal['skill_id']) != cap:
                    reasons.append('capability_changed')
                if cap['goal'] != request['snapshot']['goal']:
                    reasons.append('goal_mismatch')
                if proposal['arguments'] != cap['arguments']:
                    reasons.append('unsupported_arguments')
                if proposal['budget_s'] > cap['max_budget_s'] or proposal['budget_ticks'] > cap['max_budget_ticks']:
                    reasons.append('skill_budget_exceeded')
                if blocked_method(request['snapshot'],cap,data):
                    reasons.append('same_failed_attempt')
            if number(now_elapsed_s) and proposal['budget_s'] > 180.-now_elapsed_s:
                reasons.append('budget_exceeds_remaining_time')
        verdict = dict(request_id=request_id,accepted=not reasons,execution_authorized=False,reasons=reasons,
                       rm_effect='preserve',goal_ref=copy.deepcopy(request['snapshot']['goal_ref']),
                       scope='proposal review only; no scheduler or action dispatch')
        if not reasons:verdict['method']=copy.deepcopy(cap)
        data['decisions'].append(copy.deepcopy(verdict))
        self.history.commit(data)
        return verdict
