"""Development-only navigation with motion memory and bounded root-policy probes."""
from copy import deepcopy
import math,time
from .waypoint_control import WaypointController
from .adaptive_route import AdaptiveRoute
from .exploration_route import pose_only
from .navigation_choice import CandidateMailbox,choice_packet,validate_choice
from .rl_policy import cpu_tree,state_fingerprint
from .rl_core import ShapedWoodRewardV3
from .runtime import jsonable,write_json
from .stall_replay import wood_sample


class AdaptiveWaypointController(WaypointController):
    def __init__(self,*args,planner_mode='adaptive_rule',**kwargs):
        if planner_mode not in ('adaptive_rule','adaptive_llm'):raise ValueError('invalid adaptive planner')
        super().__init__(*args,planner_mode='visit_memory',**kwargs)
        self.planner_mode=planner_mode
        self.provider=CandidateMailbox(self.output/'candidate-exchange') if planner_mode=='adaptive_llm' else None

    def reset(self,goal=None):
        super().reset(goal)
        self.nav_steps=0;self.recovery_steps=0;self.phase='ready';self.next_action_kind='policy'
        self.probes=[];self.probe_start=0;self.escape_since=None;self.recovery_origin=None
        self.plan_events=[];self.reply=None;self.second_requested=False
        self.condition_metadata.update(llm_scope='candidate choice only; max two requests then explicit rule choices',
            route_history_schema=2,
            observed_access_source='privileged server wood sensor',root_probe_budget_actions=100,maximum_root_probes=2)

    def note(self,event,**data):
        self.plan_events.append(dict(sequence=len(self.rows),event=event,**data))
        write_json(self.output/'candidate-events.json',self.plan_events)

    def switch(self,mode,reason):
        self.mode=mode;self.head.goal=None
        self.switches.append(dict(sequence=len(self.rows),mode=mode,reason=reason,
            memory_sha256=state_fingerprint(self.student.agent.hidden_state)))

    def pause(self,reason):
        self.failure=reason;self.phase='paused';self.note('paused',reason=reason)

    def prepare_control(self,*,elapsed,limit_s,should_stop,request_stop,emit):
        if should_stop() or self.ended:return
        if self.environment.reward is not self.root_reward:raise RuntimeError('root RM replaced')
        if self.trigger and self.route is None:
            history=[[r['server_before']['server_'+a] for a in 'xyz'] for r in self.rows]
            self.route=AdaptiveRoute(pose_only(self.environment.gui),history)
            self.recovery_origin=self.route._position(self.environment.gui)
            self.route_start=dict(sequence=len(self.rows),elapsed_s=elapsed());self.mode='waypoint'
            self.switches.append(dict(sequence=len(self.rows),mode='waypoint',targets=deepcopy(self.route.targets),
                planner=self.planner_mode,trigger=deepcopy(self.trigger),source=self.condition_metadata['goal_source'],
                memory_sha256=state_fingerprint(self.student.agent.hidden_state)))
            write_json(self.output/'waypoint-route.json',self.switches[-1])
            if self.provider:self.phase='release'
        if self.mode not in ('waypoint','root_probe'):return
        def over():return self.recovery_steps>=400 or elapsed()-self.route_start['elapsed_s']>=60
        if over():self.pause('waypoint_budget_exhausted')
        g=self.environment.gui
        if g.get('error')!=0 or g.get('server_available')!=1 or g.get('screen')!=0:self.pause('invalid_recovery_observation')
        if self.phase!='paused' and ShapedWoodRewardV3.state(g)[0] in ('mine','collect'):
            self.phase='ready';self.switch('resumed','wood_interaction_ready');self.next_action_kind='policy';return
        if self.mode=='root_probe' and self.recovery_steps-self.probe_start>=100 and self.phase!='paused':
            self.route.replan(g,sequence=self.nav_steps,reason='root_probe_no_access')
            self.switch('waypoint','root_probe_no_access');self.escape_since=None
        if self.route.exhausted:self.pause('exploration_candidates_exhausted')
        if self.mode=='waypoint' and self.phase=='ready' and not self.failure:
            p=self.route._position(g)
            displaced=math.hypot(p[0]-self.recovery_origin[0],p[2]-self.recovery_origin[2])>=3
            if not displaced:self.escape_since=None
            elif self.escape_since is None:self.escape_since=self.recovery_steps
            eligible=(self.route.receipts and self.escape_since is not None and self.recovery_steps-self.escape_since>=20
                and len(self.probes)<2 and (not self.probes or math.hypot(p[0]-self.probes[-1][0],p[2]-self.probes[-1][2])>3))
            if eligible:
                self.probes.append(p);self.probe_start=self.recovery_steps;self.switch('root_probe','sustained_local_escape_probe')
            elif self.provider and self.provider.calls==1 and self.route.failures and not self.second_requested:
                # Only choose before any action in the newly generated candidate.
                if self.nav_steps==self.route.attempt_start:self.second_requested=True;self.phase='release'
        if self.phase=='request':
            self.request=choice_packet(self.route,str(self.output)+':'+str(self.provider.calls+1),self.nav_steps)
            try:
                self.provider.start(self.request);self.wait_started=elapsed();self.phase='waiting'
                self.note('request',packet=self.request)
            except Exception as exc:self.pause('provider_start_failed:'+type(exc).__name__)
        while self.phase=='waiting':
            if should_stop():return
            if over():self.pause('waypoint_budget_exhausted');break
            if elapsed()-self.wait_started>=20:self.pause('planner_timeout');break
            try:reply=self.provider.poll()
            except Exception as exc:self.pause('provider_failed:'+type(exc).__name__);break
            if should_stop():return
            if reply is not None:
                try:self.reply=validate_choice(reply,self.request)
                except (ValueError,TypeError,KeyError):self.pause('invalid_response');break
                self.phase='refresh';self.note('reply',reply=self.reply,wait_s=elapsed()-self.wait_started);break
            time.sleep(.02)
        if self.phase=='review':
            current=self.route._position(g);old=self.route._position(self.request['pose']);i=self.reply['candidate_index']
            if i<0:self.pause('provider_requested_pause')
            elif math.dist(current,old)>.5 or self.nav_steps!=self.request['sequence']:self.pause('stale_choice_pose')
            elif self.route.blocked(g,self.request['candidates'][i]['target']):self.pause('choice_blocked_after_refresh')
            else:
                self.route.choose(i,sequence=self.nav_steps,source='llm');self.phase='ready'
                self.note('choice_applied',candidate_index=i,nav_sequence=self.nav_steps)
        if self.phase=='paused':request_stop('planning_paused')
        self.next_action_kind='context_only' if self.phase in ('release','refresh') else 'policy'

    def act(self,screen):
        if self.ended or self.is_stopped():raise InterruptedError('STOP before adaptive action')
        if self.phase not in ('release','refresh'):
            if self.phase!='ready':raise RuntimeError('policy inference during planning pause')
            return super().act(screen)
        if self.pending is not None or self.environment.reward is not self.root_reward:raise RuntimeError('invalid neutral state')
        features,state=self.student.features(screen)
        action=self.environment.env.raw.action_space.noop();self.controller_name='planner_neutral';self.head.goal=None
        self.pending=dict(mode='planner_neutral',goal=None,target=None,action=jsonable(action),context_only=True,
            neutral_phase=self.phase,server_before=deepcopy(self.environment.gui),frame=f'frames/{len(self.rows):06d}.png')
        self.neutral_context=dict(features=cpu_tree(features),state=state,memory_before=state_fingerprint(self.student.agent.hidden_state))
        return action

    def commit(self,action,reward,terminal):
        if self.pending is None or jsonable(action)!=self.pending['action']:raise ValueError('adaptive dispatch mismatch')
        row=self.pending;self.pending=None
        if row['mode']=='planner_neutral':
            c=self.neutral_context;self.neutral_context=None
            self.student.rows.append(dict(features=c['features'],action=None,old_logp=None,value=None,first=False,
                env_action=jsonable(action),reward=reward,terminal=terminal,context_only=True,loss_eligible=False,waypoint_goal=None))
            self.student.agent.hidden_state=c['state'];self.student.first=False
            row.update(memory_before=c['memory_before'],memory_after=state_fingerprint(c['state']))
        else:self.student.commit(action,reward,terminal)
        row.update(reward=reward,terminal=terminal,server_after=deepcopy(self.environment.gui));self.rows.append(row);self.observe_space()
        if self.is_stopped() or terminal:self.ended=True;return
        g=self.environment.gui
        if row['mode']=='planner_neutral':
            if g.get('held_key_count')!=0 or g.get('error')!=0 or g.get('server_available')!=1 or g.get('screen')!=0:
                self.pause('neutral_release_unverified');return
            self.note('neutral_confirmed',phase=self.phase,held_key_count=0)
            self.phase='request' if self.phase=='release' else 'review';return
        if self.mode=='root':
            decision=self.detector.update(wood_sample(g,jsonable(action),str(self.output),previous=row['server_before']))
            self.decisions.append(decision)
            if decision['event']=='stalled':self.trigger=deepcopy(decision)
        elif self.mode in ('waypoint','root_probe'):
            self.recovery_steps+=1
            self.route.observe_action(row['server_before'],g,jsonable(action),sequence=len(self.rows))
            if self.mode=='waypoint':
                self.nav_steps+=1;self.route.observe(g,sequence=self.nav_steps)

    def close(self):
        if self.provider:self.provider.close()
        super().close()
