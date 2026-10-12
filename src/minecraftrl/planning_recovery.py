"""Eval-only pause, review and bounded neural recovery. No paid API creation."""
from copy import deepcopy
import time

from .goal_recovery import GoalRecoveryController
from .recovery_draft import validate_draft, planning_context
from .recovery_live import checked_reply
from .recovery_evidence import assess_stall, assess_attempt
from .replanning import (PlanHistory, ReplanGuard, collection_capability,
                         goal_conditioning_capability, make_snapshot)
from .rl_policy import cpu_tree, state_fingerprint
from .runtime import jsonable, write_json
from .stall_detection import StallDetector
from .stall_replay import log_count, wood_sample


class FixedRecoveryProvider:
    """Deterministic delayed fixture, explicitly not an LLM response."""
    kind='fixed_local_fixture'
    live=False
    actual_llm_calls=0
    def __init__(self, choice, delay_s=5., *, clock=time.monotonic):
        import math
        if choice not in ('explore_then_resume','remain_paused'):
            raise ValueError('unsupported fixed recovery choice')
        if type(delay_s) not in (float,int) or not math.isfinite(delay_s) or not 0<=delay_s<=60:
            raise ValueError('invalid fixture response delay')
        self.choice,self.delay_s,self.clock=choice,delay_s,clock
        self.calls=0;self.closed=False;self.packet=None
    def start(self,packet):
        if self.calls or self.closed:raise RuntimeError('provider already used or closed')
        self.packet=deepcopy(packet);self.calls=1;self.ready=self.clock()+self.delay_s
    def poll(self):
        if self.closed or self.packet is None or self.clock()<self.ready:return None
        return validate_draft(dict(request_id=self.packet['request_id'],evidence_id=self.packet['evidence_id'],
            choice=self.choice,reason='Fixed local response for execution-contract validation; not LLM planning.'),self.packet)
    def close(self):self.closed=True


class PlanningRecoveryController(GoalRecoveryController):
    """Single pause and decision; STOP remains latched and owned by run_episode."""
    wait_limit_s=20.
    recovery_limit_s=30.

    def __init__(self,student,embedding_path,output,*,raw,world_id,provider,wait=time.sleep):
        super().__init__(student,embedding_path,output)
        self.raw,self.world_id,self.provider,self.wait=raw,world_id,provider,wait
        self.phase='collect';self.generation=0;self.planner_events=[]
        self.next_action_kind='policy';self.neutral_context=None;self.request=None
        self.elapsed=lambda:0.;self.emit=lambda *a,**k:None
        self.deadline=180.;self.release_sequence=None
        self.attempts=[];self.attempt_start=None
        self.resumed_detector=None;self.resumed_decisions=[]
        self.stop_reason=lambda:None

    def note(self,event,**fields):
        row=dict(event=event,step=len(self.rows),elapsed_s=self.elapsed(),generation=self.generation,**fields)
        self.planner_events.append(jsonable(row))
        self.emit('planner_'+event,**{k:v for k,v in row.items() if k!='event'})

    def reset(self,goal=None):
        super().reset(goal)
        self.baseline=log_count(self.environment.gui)
        cap=goal_conditioning_capability(str(self.student.checkpoint),self.embed_hashes['wood'],self.embed_hashes['intervention'])
        self.capabilities={'collect_wood':collection_capability('wood',str(self.student.checkpoint)),cap['skill_id']:cap}
        self.guard=ReplanGuard(PlanHistory(self.output/'planner-history.json'),self.capabilities,allow_goal_conditioning=True)
        self.condition_metadata['planning']=dict(provider=self.provider.kind,choice=self.provider.choice,
            delay_s=self.provider.delay_s,wait_limit_s=self.wait_limit_s,recovery_limit_s=self.recovery_limit_s,
            maximum_calls=1,actual_llm_calls=None if self.provider.live else 0,
            neutral_is_context_only=True,eligible_for_training=False,
            planner_memory_schema=2,duplicate_guard='observed_conditions_and_method',
            recovery_handoff='fresh_target_or_interaction_v2')

    def snapshot(self):
        return make_snapshot(self.environment.gui,world_id=self.world_id,episode_id=str(self.output),
            goal='wood',task=None,generation=self.generation,sequence=len(self.rows),elapsed_s=self.elapsed(),
            status='stalled',release_sequence=self.release_sequence,
            held_key_count=self.environment.gui.get('held_key_count'),baseline_count=self.baseline)

    def pause(self,reason):
        self.phase='paused';self.scheduled=None
        self.note('paused',reason=reason)

    def make_packet(self,s,evidence):
        memory=planning_context(self.guard,self.request['request_id'])
        return dict(request_id=self.request['request_id'],evidence_id=evidence['event_id'],
            root_goal=dict(goal='wood',target_count=1),root_reference=s['goal_ref'],
            assessment=assess_stall(self.trigger_observation,self.trigger,baseline_count=self.baseline),
            status='stalled',remaining_control_s=self.deadline-self.elapsed(),runtime_binding=s,
            planner_context=memory),memory['call_model']

    def create_request(self,s):
        evidence=dict(self.trigger,event_id=self.trigger.get('event_id',str(self.output)+':stall-'+str(self.trigger['step'])))
        if not any(f['event_id']==evidence['event_id'] for f in self.guard.history.data['failures']):
            self.guard.record_failure(s,evidence,'collect_wood')
        return self.guard.request(s,evidence),evidence

    def check_reply(self,reply):
        return checked_reply(reply,self.packet,live=self.provider.live)

    def resumed_stall(self,evidence):
        self.pause('resumed_collection_stalled_request_budget_exhausted')

    def remaining_requests(self):return 0

    def finish_recovery(self,reason):
        """Consume committed evidence before scheduling another neural action."""
        if self.attempt_start is None:return
        rows=self.rows[self.attempt_start:]
        result=assess_attempt(self.attempt_before,[r['server_after'] for r in rows],
            reason=reason,baseline_count=self.baseline,budget_ticks=self.max_intervention_ticks)
        # A server target remains locked after looking away. Only the acquisition
        # transition proves a fresh sighting without adding another vision sensor.
        may_resume=result['classification'] in ('interaction_ready','target_acquired')
        # This controller has one approved request/window. Exhaustion is evidence
        # for a future planner, never permission for another call or blind retry.
        if result['goal_completed']:next_phase='completed'
        elif reason in ('stop','control_budget','environment_done'):next_phase='ended'
        else:next_phase='resumed' if may_resume else 'paused'
        receipt=dict(request_id=self.request['request_id'],start_step=self.attempt_start+1,
            end_step=len(self.rows),committed_transitions=len(rows),assessment=result,next_phase=next_phase,
            root_reference=deepcopy(self.request['snapshot']['goal_ref']),
            remaining_calls=self.remaining_requests(),remaining_interventions=self.remaining_requests())
        outcome=('completed' if result['goal_completed'] else
                 'interrupted' if reason in ('stop','control_budget') else
                 'unverified' if result['classification']=='observation_unreliable' else
                 'partial' if may_resume or result['classification'].startswith('recovery_incomplete_with_') else 'failed')
        after=None if outcome=='unverified' else self.snapshot()
        self.guard.record_attempt(self.request['request_id'],self.attempt_snapshot,after,
            outcome=outcome,reason=result['classification'],committed_steps=len(rows))
        self.attempts.append(receipt);self.attempt_start=None
        write_json(self.output/'recovery-attempts.json',self.attempts)
        self.note('recovery_assessed',**receipt)
        self.phase=next_phase;self.scheduled=None
        if next_phase=='resumed':
            self.scheduled=('wood',reason)
            # Keep the episode's graph/history intact. A new collection attempt
            # needs its own observation window: the root stall may still be latched.
            self.resumed_detector=StallDetector()
            self.resumed_detector.update(wood_sample(rows[-1]['server_after'],{},str(self.output)+':resumed'))
        elif next_phase=='paused':self.pause('recovery_requires_new_plan')

    def prepare_control(self,*,elapsed,limit_s,should_stop,request_stop,emit):
        self.elapsed,self.deadline,self.emit=elapsed,limit_s,emit
        self.stop_reason=should_stop
        if should_stop():return
        if self.phase=='request':
            if limit_s-elapsed()<=self.wait_limit_s+self.recovery_limit_s:
                self.pause('insufficient_remaining_time')
            else:
                s=self.snapshot()
                self.request,evidence=self.create_request(s)
                self.packet,call_model=self.make_packet(s,evidence)
                write_json(self.output/'planner-input.json',self.packet)
                if not call_model:
                    self.pause('no_available_alternative');request_stop('planning_paused');return
                self.wait_started=elapsed();self.phase='waiting'
                self.note('wait_started',request_id=self.request['request_id'],snapshot=s,packet=self.packet,
                          provider=self.provider.kind,live=self.provider.live)
                try:self.provider.start(self.packet)
                except Exception as exc:self.pause('provider_start_failed:'+type(exc).__name__)
        while self.phase=='waiting':
            if should_stop():return
            if elapsed()-self.wait_started>=self.wait_limit_s:
                self.pause('planner_timeout');break
            try:reply=self.provider.poll()
            except Exception as exc:
                self.pause('provider_failed:'+type(exc).__name__);break
            if reply is not None:
                try:
                    validated=self.check_reply(reply)
                except (KeyError,TypeError,ValueError):
                    self.pause('invalid_response');break
                self.reply=validated;self.phase='refresh'
                self.note('response_received',reply=validated,wait_s=elapsed()-self.wait_started)
                break
            self.wait(.02)
        if self.phase=='recovery' and elapsed()>=self.recovery_expires:
            if self.attempt_start is None:self.pause('recovery_expired_before_dispatch')
            else:self.finish_recovery('planning_time_budget')
        if self.phase=='paused':request_stop('planning_paused')
        self.next_action_kind='context_only' if self.phase in ('release','refresh') else 'policy'

    def act(self,screen):
        if self.phase not in ('release','refresh'):
            if self.phase in ('waiting','request','paused'):raise RuntimeError('policy inference during planning pause')
            if self.scheduled is not None and self.scheduled[0]=='intervention':
                self.attempt_start=len(self.rows)
                self.attempt_before=deepcopy(self.environment.gui)
                self.attempt_snapshot=self.snapshot()
            return super().act(screen)
        if self.ended or self.is_stopped():raise InterruptedError('STOP before planning neutral')
        if self.pending is not None:raise RuntimeError('uncommitted planning neutral')
        self.student.goal_embed=self.wood_embed.copy()
        features,state=self.student.features(screen)  # Encode observed context; do not sample a motor action/RNG.
        action=self.raw.action_space.noop()
        self.controller_name='planner_neutral'
        self.pending=dict(mode='wood',owner='planner_neutral',action=jsonable(action),
            server_before=jsonable(deepcopy(self.environment.gui)),reward_context=list(self.context),
            frame=f'frames/{len(self.rows):06d}.png',embedding_sha256=self.embed_hashes['wood'],
            loss_eligible=False,context_only=True,neutral_phase=self.phase)
        self.neutral_context=dict(features=cpu_tree(features),state=state,
            memory_before=state_fingerprint(self.student.agent.hidden_state))
        return action

    def commit(self,action,reward,terminal):
        if self.pending is None or self.pending['owner']!='planner_neutral':
            previous=self.trigger
            super().commit(action,reward,terminal)
            if self.attempt_start is not None:
                if self.is_stopped():self.finish_recovery('stop')
                elif terminal:
                    reason='goal_completed' if log_count(self.environment.gui)>self.baseline else 'environment_done'
                    self.finish_recovery(reason)
                elif (self.rows[-1]['server_before']['wood_v3_kind']==0
                      and self.rows[-1]['server_after']['wood_v3_kind']==1):
                    self.finish_recovery('wood_target_acquired')
                elif self.scheduled is not None and self.scheduled[0]=='wood':
                    self.finish_recovery(self.scheduled[1])
            elif self.phase=='resumed' and not terminal and not self.is_stopped():
                row=self.rows[-1]
                decision=self.resumed_detector.update(wood_sample(row['server_after'],row['action'],
                    str(self.output)+':resumed',previous=row['server_before']))
                decision['step']=len(self.rows);self.resumed_decisions.append(decision)
                if decision['event']=='stalled':
                    evidence=dict(decision,event_id=str(self.output)+':restall-'+str(len(self.rows)))
                    self.guard.record_failure(self.snapshot(),evidence,'collect_wood')
                    self.note('resumed_collection_stalled',evidence=evidence)
                    self.resumed_stall(evidence)
            if previous is None and self.trigger is not None and not terminal and not self.is_stopped():
                self.trigger_observation=deepcopy(self.environment.gui)
                self.generation+=1;self.scheduled=None;self.phase='release'
                self.note('stall',trigger=self.trigger)
            return
        if jsonable(action)!=self.pending['action']:raise ValueError('planning neutral dispatch mismatch')
        row=self.pending;self.pending=None
        row.update(server_after=jsonable(self.environment.gui),reward=reward,terminal=terminal)
        self.rows.append(row)
        context=self.neutral_context;self.neutral_context=None
        self.student.rows.append(dict(features=context['features'],action=None,old_logp=None,value=None,
            first=False,env_action=jsonable(action),reward_context=list(row['reward_context']),
            reward=reward,terminal=terminal,context_only=True,loss_eligible=False,
            goal_conditioning=dict(mode='wood',embedding_sha256=self.embed_hashes['wood'])))
        self.student.agent.hidden_state=context['state'];self.student.first=False
        row['memory_before']=context['memory_before'];row['memory_after']=state_fingerprint(context['state'])
        decision=self.detector.update(wood_sample(self.environment.gui,jsonable(action),str(self.output),
            previous=row['server_before'],terminal=terminal))
        decision['step']=len(self.rows);self.decisions.append(decision)
        if terminal or self.is_stopped():
            self.ended=True;self.scheduled=None;self.detector.stop()
            self.note('terminal_during_neutral',terminal=terminal,stopped=self.is_stopped());return
        gui=self.environment.gui
        if gui.get('server_available')!=1 or gui.get('error')!=0 or gui.get('held_key_count')!=0 or gui.get('screen')!=0:
            self.pause('neutral_release_unverified');return
        self.release_sequence=len(self.rows)
        self.note('neutral_confirmed',phase=self.phase,server_tick=gui['server_tick'],held_key_count=0,
                  memory_before=row['memory_before'],memory_after=row['memory_after'])
        if self.phase=='release':self.phase='request';return
        # The refreshed server/RM state is checked before selecting a neural condition.
        current=self.snapshot()
        self.review_reply(current)

    def review_reply(self,current):
        if self.reply['choice']=='remain_paused':
            self.pause('provider_requested_pause');return
        cap=self.capabilities['explore_then_resume']
        proposal=dict(request_id=self.reply['request_id'],skill_id=cap['skill_id'],arguments=cap['arguments'],
            budget_s=self.recovery_limit_s,budget_ticks=200,evidence_ids=[self.reply['evidence_id']],
            reason=self.reply['reason'][:240])
        verdict=self.guard.review(self.request['request_id'],proposal,current,now_elapsed_s=self.elapsed())
        self.note('reviewed',verdict=verdict,proposal=proposal,current=current)
        if not verdict['accepted']:
            self.pause('proposal_rejected');return
        self.phase='recovery';self.scheduled=('intervention','planner_review_accepted')
        self.recovery_expires=self.elapsed()+self.recovery_limit_s

    def close(self):
        if self.attempt_start is not None:
            self.finish_recovery('control_budget' if self.stop_reason()=='time_limit' else 'stop')
        self.provider.close();self.generation+=1
        if self.resumed_detector is not None:self.resumed_detector.stop()
        self.note('closed',provider_calls=self.provider.calls,actual_llm_calls=self.provider.actual_llm_calls)
        super().close()
