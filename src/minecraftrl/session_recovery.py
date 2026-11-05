"""Reuse observed graph stalls and the bounded neural escape skill in task sessions."""
from copy import deepcopy
import math

from .rl_policy import state_fingerprint
from .stall_detection import StallDetector
from .stall_replay import wood_sample, log_count
from .subgoal_skills import navigation_state, completion

HAZARDS=('health_loss','hostile_proximity')
STABILIZED=('damage_stabilized','hazard_stabilized')
RULE_ORDER={'damage-v2':2,'hazard-v3':3,'hazard-v4':4,'pursuit-v5':5,'pursuit-v6':6,'pursuit-v7':7,'pursuit-v8':8}


class SessionRecovery:
    """Local rule, privileged observations, neural motor actions; no reward changes."""
    def __init__(self,policy,skill,history=None,*,threat_guard=False,hazard_skill=None,pursuit_guard=False):
        if hazard_skill is not None and not threat_guard:raise ValueError('hazard condition requires threat guard')
        if pursuit_guard and not threat_guard:raise ValueError('pursuit guard requires threat guard')
        self.pursuit_guard=pursuit_guard
        self.policy=policy;self.skill=skill;self.threat_guard=threat_guard;self.hazard_skill=hazard_skill or skill
        self.saved=deepcopy(history or dict(schema=1,history=[],high_water={}))
        s=self.saved
        if 'pending_damage' in s:
            if 'pending_hazard' in s:raise ValueError('ambiguous pending hazard')
            context=s.pop('pending_damage')
            s['pending_hazard']=dict(context=context,trigger='health_loss') if context is not None else None
        if (set(s)-{'pending_hazard'}!={'schema','history','high_water'} or s['schema']!=1
                or not isinstance(s['history'],list) or len(s['history'])>64
                or not isinstance(s['high_water'],dict) or len(s['high_water'])>64):
            raise ValueError('invalid recovery history')
        if any(not isinstance(k,str) or not k or len(k)>256 or type(v) is not int or v<0 for k,v in s['high_water'].items()):
            raise ValueError('invalid recovery high-water evidence')
        pending=s.get('pending_hazard')
        if pending is not None and (not isinstance(pending,dict) or set(pending)!={'context','trigger'}
                or not isinstance(pending['context'],str) or pending['context'] not in s['high_water'] or pending['trigger'] not in HAZARDS):
            raise ValueError('invalid pending hazard context')
        parents=set()
        for index,r in enumerate(s['history']):
            if (not isinstance(r,dict) or set(r)-{'continued_from','trigger','completion_rule','embedding_sha256','initial_embedding_sha256','cond_scale','initial_cond_scale'}!={'context','position','high_water','outcome','elapsed_ticks','elapsed_s'}
                    or any(type(r[k]) not in (int,float) or not math.isfinite(r[k]) or not 0<=r[k]<=7
                           for k in ('cond_scale','initial_cond_scale') if k in r)
                    or 'initial_cond_scale' in r and 'initial_embedding_sha256' not in r
                    or any(not isinstance(r[k],str) or len(r[k])!=64 or any(c not in '0123456789abcdef' for c in r[k])
                           for k in ('embedding_sha256','initial_embedding_sha256') if k in r)
                    or ('trigger' in r and r['trigger'] not in HAZARDS)
                    or ('completion_rule' in r and (r['completion_rule'] not in RULE_ORDER or r.get('trigger') not in HAZARDS))
                    or r['context'] not in s['high_water'] or not isinstance(r['position'],list) or len(r['position'])!=3
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in r['position'])
                    or type(r['high_water']) is not int or not 0<=r['high_water']<=s['high_water'][r['context']]
                    or r['outcome'] not in ('displaced',*STABILIZED,'inventory_progress','tick_budget','time_budget','control_changed','shutdown','terminal','recovery_blocked')
                    or type(r['elapsed_ticks']) is not int or r['elapsed_ticks']<0
                    or type(r['elapsed_s']) not in (int,float) or not math.isfinite(r['elapsed_s']) or r['elapsed_s']<0):
                raise ValueError('invalid recovery receipt')
            if RULE_ORDER.get(r.get('completion_rule'),0)>=5 and not pursuit_guard:
                raise ValueError('saved pursuit history requires pursuit guard')
            if 'continued_from' in r:
                parent=r['continued_from']
                if type(parent) is not int or not 0<=parent<index or parent in parents:
                    raise ValueError('invalid recovery continuation chain')
                old=s['history'][parent];parents.add(parent)
                upgraded=(old['outcome'] in ('tick_budget','time_budget') and
                    (old.get('completion_rule')=='hazard-v3' and RULE_ORDER.get(r.get('completion_rule'),0)>=4 or
                     old.get('completion_rule')=='pursuit-v5' and RULE_ORDER.get(r.get('completion_rule'),0)>=6 or
                     old.get('completion_rule')=='pursuit-v7' and r.get('completion_rule')=='pursuit-v8'))
                if (any(r[k]!=old[k] for k in ('context','position','high_water'))
                        or (r.get('initial_embedding_sha256',self.method_sha(r)),r.get('initial_cond_scale',r.get('cond_scale',6.)))!=self.method_id(old)
                        or old['outcome'] not in ('shutdown','control_changed') and not upgraded
                        or old.get('trigger') in HAZARDS and r.get('trigger') not in HAZARDS
                        or RULE_ORDER.get(r.get('completion_rule'),0)<RULE_ORDER.get(old.get('completion_rule'),0)):
                    raise ValueError('incompatible recovery continuation')
        self.mode='collection';self.pending=None;self.context=None;self.attempts=0;self.resume_from=None
        self.last_health=None;self.blocked=None
        # Legacy three-block damage receipts did not establish stable health.
        if 'pending_hazard' not in s:
            s['pending_hazard']=next((dict(context=r['context'],trigger=r['trigger']) for r in reversed(s['history'])
                if r.get('trigger') in HAZARDS and r['outcome'] not in STABILIZED),None)

    def totals(self,index):
        ticks=0;seconds=0.
        while True:
            r=self.saved['history'][index];ticks+=r['elapsed_ticks'];seconds+=r['elapsed_s']
            if 'continued_from' not in r:return ticks,seconds
            index=r['continued_from']

    def bind(self,world_id,task,server,now,emit):
        self.end('control_changed',now,emit)
        self.context=None;self.resume_from=None;self.last_health=None;self.blocked=None
        if task['goal']!='wood':return
        self.context='%s:%s:%s'%(world_id,task['task_id'],task['revision'])
        self.root_embed=self.policy.goal_embed.copy()
        self.root_cond_scale=getattr(self.policy.agent,'cond_scale',6.)
        self.detector=StallDetector();self.previous=None
        self.observe(task,server,{},now,emit)
        if self.blocked:return
        nav=navigation_state(server,0)
        for index in range(len(self.saved['history'])-1,-1,-1):
            r=self.saved['history'][index]
            if r['context']!=self.context:continue
            ticks,seconds=self.totals(index)
            max_ticks,max_seconds=self.limits(r)
            upgrade=(r['outcome'] in ('tick_budget','time_budget') and
                (self.threat_guard and r.get('completion_rule')=='hazard-v3' or self.pursuit_guard and r.get('completion_rule') in ('pursuit-v5','pursuit-v7')))
            distance=math.hypot(nav['position'][0]-r['position'][0],nav['position'][2]-r['position'][2])
            if ((r['outcome'] in ('shutdown','control_changed') or upgrade) and ticks<max_ticks and seconds<max_seconds
                    and self.method_id(r)==self.skill_id(self.selected_skill(r.get('trigger')))
                    and len(self.saved['history'])<64 and self.attempts<2
                    and (r.get('trigger') in HAZARDS or r['high_water']>=self.saved['high_water'][self.context])
                    and (r.get('trigger') in HAZARDS or math.dist(nav['position'],r['position'])<=6. and distance<3.)):
                self.resume_from=index;self.pending=('leave_stalled_area','resume_interrupted_skill')
                emit('recovery_resume_scheduled',continued_from=index,remaining_ticks=max_ticks-ticks,
                     remaining_seconds=max_seconds-seconds,process_policy_state_reset=True,budget_rule_upgraded=upgrade)
            break
        if self.pending_hazard_trigger() and self.resume_from is None:
            latest_index=next((i for i in range(len(self.saved['history'])-1,-1,-1)
                if self.saved['history'][i]['context']==self.context and self.saved['history'][i].get('trigger') in HAZARDS),None)
            latest=self.saved['history'][latest_index] if latest_index is not None else None
            if (latest and latest.get('completion_rule') in RULE_ORDER
                    and self.same_hazard_rule(latest)
                    and self.method_id(latest)==self.skill_id(self.selected_skill(latest.get('trigger')))
                    and latest['outcome'] not in STABILIZED
                    and (latest['outcome'] in ('tick_budget','time_budget','recovery_blocked')
                         or self.totals(latest_index)[0]>=self.limits(latest)[0] or self.totals(latest_index)[1]>=self.limits(latest)[1])):
                self.blocked='damage_recovery_incomplete'
            elif self.attempts>=2 or len(self.saved['history'])>=64:
                self.blocked='damage_recovery_budget_exhausted'
            elif self.duplicate_area(nav,self.pending_hazard_trigger()):
                self.blocked='duplicate_area_without_inventory_progress'
            else:self.pending=('leave_stalled_area',self.pending_hazard_trigger())

    def pending_hazard_trigger(self):
        pending=self.saved['pending_hazard']
        return pending['trigger'] if pending and pending['context']==self.context else None

    def hazard_pending(self):
        return bool(self.pending and (self.pending[1] in HAZARDS or
            self.resume_from is not None and self.saved['history'][self.resume_from].get('trigger') in HAZARDS))

    def hazard_rule(self):return 'pursuit-v8' if self.pursuit_guard else 'hazard-v4' if self.threat_guard else 'damage-v2'

    def same_hazard_rule(self,receipt):
        return not self.pursuit_guard or RULE_ORDER.get(receipt.get('completion_rule'),0)>=5

    def selected_skill(self,trigger):return self.hazard_skill if trigger in HAZARDS else self.skill

    def method_sha(self,receipt):return receipt.get('embedding_sha256',self.skill['sha256'])

    def method_id(self,receipt):return self.method_sha(receipt),receipt.get('cond_scale',6.)

    def skill_id(self,skill):return skill['sha256'],skill.get('cond_scale',6.)

    def limits(self,receipt=None):
        receipt=receipt if receipt is not None else self.saved['history'][-1]
        if self.pursuit_guard and receipt.get('trigger') in HAZARDS:return 900,180.
        return (400,90.) if self.threat_guard and receipt.get('trigger') in HAZARDS else (200,30.)

    def duplicate_area(self,nav,trigger=None):
        return any(r['context']==self.context and math.dist(r['position'],nav['position'])<=6. and
            self.method_id(r)==self.skill_id(self.selected_skill(trigger)) and
            (trigger not in HAZARDS or self.same_hazard_rule(r)) and
            (r.get('trigger') in HAZARDS or r['high_water']>=self.saved['high_water'][self.context]) and
            (r['outcome'] in ('displaced',*STABILIZED,'tick_budget','time_budget','recovery_blocked') or
             self.totals(i)[0]>=200 or self.totals(i)[1]>=30.) for i,r in enumerate(self.saved['history']))

    def observe(self,task,server,action,now,emit):
        if self.context is None:return
        if task['status']!='running' or task['goal']!='wood':
            if task['status']=='completed' and self.pending_hazard_trigger():self.saved['pending_hazard']=None
            self.end('terminal' if task['status']=='completed' else 'control_changed',now,emit);return
        nav=navigation_state(server,0)  # validates live telemetry, including GUI state
        health=server.get('server_health')
        if type(health) not in (int,float) or not math.isfinite(health) or not 0<health<=20:
            raise ValueError('invalid live server health')
        damage=self.last_health is not None and self.last_health-health>=2.
        from .threat_observation import active_threat,active_pursuit,grounded
        threat=active_pursuit(server) if self.pursuit_guard else active_threat(server) if self.threat_guard else False
        settled=grounded(server) if self.pursuit_guard else True
        trigger='health_loss' if damage else 'hostile_proximity' if threat else None
        if self.mode=='leave_stalled_area' and (threat or self.last_health is not None and health<self.last_health):
            self.last_damage_tick=nav['tick']
        self.last_health=health
        if trigger or health<=8.:
            self.saved['pending_hazard']=dict(context=self.context,trigger=trigger or 'health_loss')
        if health<=8.:
            self.blocked='low_health_requires_plan'
        elif trigger:
            if damage or self.mode=='collection' and not self.hazard_pending():
                emit('recovery_damage_observed' if damage else 'recovery_threat_observed',server_health=health,
                     threat_count=server.get('threat_count'),threat_distance=server.get('threat_distance'),
                     pursuit_count=server.get('pursuit_count'),pursuit_distance=server.get('pursuit_distance'),source='privileged_server')
            if self.mode=='leave_stalled_area':
                self.saved['history'][-1].setdefault('trigger',trigger)
                self.saved['history'][-1]['completion_rule']=self.hazard_rule()
                if self.method_id(self.saved['history'][-1])!=self.skill_id(self.hazard_skill):
                    self.pending=('retarget_hazard',trigger)
            elif self.duplicate_area(nav,trigger) or self.attempts>=2 or len(self.saved['history'])>=64:
                self.blocked='damage_recovery_budget_exhausted'
            else:self.pending=('leave_stalled_area',trigger)
        count=log_count(server)
        previous_best=self.saved['high_water'].get(self.context,count)
        self.saved['high_water'][self.context]=max(count,self.saved['high_water'].get(self.context,0))
        if self.mode=='collection' and count>previous_best and self.pending and not self.hazard_pending():
            self.pending=None;self.resume_from=None
            emit('recovery_cancelled',reason='objective_inventory_increased_before_switch',context=self.context)
        if self.mode=='leave_stalled_area':
            receipt=self.saved['history'][-1]
            receipt.update(elapsed_ticks=nav['tick']-self.enter_tick,elapsed_s=max(0.,now-self.enter_time))
            damaged=receipt.get('trigger') in HAZARDS
            max_ticks,max_seconds=self.limits(receipt)
            displacement=math.hypot(nav['position'][0]-self.enter['position'][0],nav['position'][2]-self.enter['position'][2])
            if count>receipt['high_water'] and not damaged:reason='inventory_progress'
            elif damaged and (self.pursuit_guard or displacement>=12.) and nav['tick']-self.last_damage_tick>=100 and server.get('screen')==0 and settled:
                reason='hazard_stabilized' if self.threat_guard else 'damage_stabilized'
            elif not damaged and completion('leave_stalled_area',self.enter,nav):reason='displaced'
            elif self.prior_ticks+receipt['elapsed_ticks']>=max_ticks:reason='tick_budget'
            elif self.prior_seconds+receipt['elapsed_s']>=max_seconds:reason='time_budget'
            else:reason=None
            if reason:
                if reason in STABILIZED and not self.pending:
                    emit('hazard_stability_observed' if self.threat_guard else 'damage_stability_observed',displacement_blocks=displacement,
                         quiet_server_ticks=nav['tick']-self.last_damage_tick,server_health=health,
                         required_blocks=0. if self.pursuit_guard else 12.,required_quiet_ticks=100,source='privileged_server')
                self.pending=('collection',reason)
        else:
            decision=self.detector.update(wood_sample(server,action,self.context,previous=self.previous))
            if decision.get('event'):
                emit('recovery_detector',decision=decision,source='privileged_server')
            if decision['event']=='stalled':
                duplicate=self.duplicate_area(nav)
                if duplicate or self.attempts>=2 or len(self.saved['history'])>=64:
                    emit('recovery_suppressed',reason='duplicate_area_without_inventory_progress' if duplicate else 'attempt_budget',context=self.context)
                elif not self.hazard_pending():self.pending=('leave_stalled_area',decision['reason'])
        self.previous=deepcopy(server)

    def needs_switch(self,server,now=None):
        if self.blocked:return False
        if (self.mode=='leave_stalled_area' and not self.pending and now is not None
                and self.prior_seconds+now-self.enter_time>=self.limits()[1]):
            self.pending=('collection','time_budget')
        return bool(self.pending and (self.pending[0]=='collection' or server.get('screen')==0))

    def switch(self,server,now,emit):
        if not self.needs_switch(server,now):return False
        nav=navigation_state(server,0)
        mode,reason=self.pending;self.pending=None
        memory=state_fingerprint(self.policy.agent.hidden_state)
        if mode=='retarget_hazard':
            receipt=self.saved['history'][-1]
            receipt.setdefault('initial_embedding_sha256',self.method_sha(receipt))
            receipt.setdefault('initial_cond_scale',receipt.get('cond_scale',6.))
            receipt['embedding_sha256']=self.hazard_skill['sha256']
            receipt['cond_scale']=self.hazard_skill.get('cond_scale',6.)
            self.policy.goal_embed=self.hazard_skill['embedding'].copy()
            self.policy.agent.cond_scale=receipt['cond_scale']
            mode='leave_stalled_area';reason='hazard_condition_handoff'
        elif mode=='leave_stalled_area':
            self.enter=nav;self.enter_tick=nav['tick'];self.enter_time=now;self.last_damage_tick=nav['tick']
            self.prior_ticks=0;self.prior_seconds=0.
            receipt=dict(context=self.context,position=list(nav['position']),
                high_water=self.saved['high_water'][self.context],outcome='in_progress',elapsed_ticks=0,elapsed_s=0.)
            if reason in HAZARDS:receipt.update(trigger=reason,completion_rule=self.hazard_rule())
            if self.resume_from is not None:
                self.prior_ticks,self.prior_seconds=self.totals(self.resume_from)
                receipt['continued_from']=self.resume_from
                parent=self.saved['history'][self.resume_from]
                receipt['position']=list(parent['position']);receipt['high_water']=parent['high_water']
                if parent.get('trigger'):receipt.update(trigger=parent['trigger'],completion_rule=self.hazard_rule())
                self.enter=dict(nav,position=list(receipt['position']))
            selected=self.selected_skill(receipt.get('trigger'));receipt['embedding_sha256']=selected['sha256']
            receipt['cond_scale']=selected.get('cond_scale',6.)
            self.saved['history'].append(receipt);self.resume_from=None
            self.policy.goal_embed=selected['embedding'].copy();self.attempts+=1
            self.policy.agent.cond_scale=receipt['cond_scale']
        else:
            if self.saved['history'][-1].get('trigger') in HAZARDS and reason in ('tick_budget','time_budget'):
                self.blocked='damage_recovery_incomplete'
            self._finish(reason,now)
            self.policy.goal_embed=self.root_embed.copy()
            self.policy.agent.cond_scale=self.root_cond_scale
            self.detector=StallDetector();self.previous=None
        self.mode=mode
        assert state_fingerprint(self.policy.agent.hidden_state)==memory
        emit('neural_skill_switched',skill=mode,reason=reason,context=self.context,
             source='local_rule_privileged_server',root_goal_preserved=True,memory_sha256=memory,
             embedding_sha256=self.embedding_hash(),cond_scale=self.policy.agent.cond_scale,
             max_ticks=self.limits()[0],max_seconds=self.limits()[1],direct_action_overrides=0)
        return True

    def embedding_hash(self):
        import hashlib
        return hashlib.sha256(self.policy.goal_embed.tobytes()).hexdigest()

    def _finish(self,reason,now):
        r=self.saved['history'][-1]
        r.update(outcome=reason,elapsed_s=max(0.,now-self.enter_time))
        if reason in STABILIZED:self.saved['pending_hazard']=None

    def end(self,reason,now,emit):
        if self.mode=='leave_stalled_area':
            outcome=self.pending[1] if self.pending and self.pending[0]=='collection' else reason
            self._finish(outcome,now)
            emit('neural_skill_interrupted',reason=reason,subgoal_outcome=outcome,context=self.context)
        self.pending=None;self.mode='collection'

    def state(self):return deepcopy(self.saved)
