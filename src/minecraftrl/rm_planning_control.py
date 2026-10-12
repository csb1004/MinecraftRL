"""Eval-only RM plan execution through existing neural/neutral/STOP boundaries."""
from copy import deepcopy
import time

from .planning_recovery import PlanningRecoveryController
from .replanning import goal_satisfied
from .rm_planner import RMPlanScheduler, fixture_plan, planning_packet, validate_plan
from .runtime import write_json
from .stall_detection import StallDetector
from .stall_replay import wood_sample


class FixedRMPlanProvider:
    kind='fixed_rm_plan_fixture'
    live=False
    actual_llm_calls=0
    choice='subgoal_plan_fixture'
    def __init__(self,delay_s=5.,*,unsupported=False,max_calls=3,clock=time.monotonic):
        import math
        if type(delay_s) not in (int,float) or not math.isfinite(delay_s) or not 0<=delay_s<=60:
            raise ValueError('invalid planner fixture delay')
        if type(max_calls) is not int or not 1<=max_calls<=3:raise ValueError('invalid request budget')
        self.delay_s,self.unsupported,self.maximum,self.clock=delay_s,unsupported,max_calls,clock
        self.calls=0;self.closed=False;self.packet=None;self.packets=[]
    def start(self,packet):
        if self.closed or self.calls>=self.maximum:raise RuntimeError('planner request budget exhausted')
        self.packet=deepcopy(packet);self.packets.append(self.packet);self.calls+=1
        self.ready=self.clock()+self.delay_s
    def poll(self):
        if self.closed or self.packet is None or self.clock()<self.ready:return None
        blocked=not any(c['skill_id']=='explore_then_resume' for c in self.packet['memory']['available_skills'])
        return validate_plan(fixture_plan(self.packet,unsupported=self.unsupported or blocked),self.packet)
    def close(self):self.closed=True


class RMPlanningController(PlanningRecoveryController):
    scheduler_type=RMPlanScheduler
    def __init__(self,*args,max_requests=3,**kwargs):
        super().__init__(*args,**kwargs)
        if type(max_requests) is not int or not 1<=max_requests<=3:raise ValueError('invalid planner call bound')
        self.max_requests=max_requests;self.pending_attempt_request=None;self.active_plan_node=None
        self.root_started=None

    def reset(self,goal=None):
        super().reset(goal)
        self.scheduler=self.scheduler_type(self.output/'rm-schedule.json',self.snapshot())
        self.root_reward_object=self.environment.reward
        self.condition_metadata['planning'].update(protocol='rm_plan_v1',maximum_calls=self.max_requests,
            schedule='dependency_ready_priority',unsupported_concepts='retained_without_execution')
        self.condition_metadata['intervention']['max_interventions']=self.max_requests

    def create_request(self,s):
        if self.pending_attempt_request is not None:
            request=self.guard.request_after_attempt(s,self.pending_attempt_request)
            self.pending_attempt_request=None
            return request,request['evidence']
        return super().create_request(s)

    def make_packet(self,s,evidence):
        packet=planning_packet(self.guard,self.request['request_id'],self.scheduler)
        write_json(self.output/f'planner-input-{self.provider.calls+1:02d}.json',packet)
        return packet,packet['proposal_allowed']

    def check_reply(self,reply):
        return validate_plan(reply,self.packet)

    def remaining_requests(self):return max(0,self.max_requests-self.provider.calls)

    def pause(self,reason):
        super().pause(reason)
        if hasattr(self,'scheduler'):self.scheduler.pause(reason)

    def choose_node(self,current):
        selected=self.scheduler.select(current,self.guard.history.snapshot())
        self.note('node_selected',selection=selected,schedule_revision=self.scheduler.state['revision'],current=current)
        node=selected['node']
        if node is None:self.pause(selected['reason']);return None
        self.active_plan_node=node
        return node

    def review_reply(self,current):
        try:self.scheduler.install(self.reply,self.packet,current)
        except ValueError as exc:self.pause('plan_rejected:'+str(exc));return
        self.note('plan_installed',plan=self.reply,current=current,schedule_revision=self.scheduler.state['revision'])
        node=self.choose_node(current)
        if node is None:return
        cap=self.capabilities[node['skill_id']]
        proposal=dict(request_id=self.request['request_id'],skill_id=cap['skill_id'],arguments=cap['arguments'],
            budget_s=min(node['budget_s'],self.deadline-self.elapsed()),budget_ticks=node['budget_ticks'],
            evidence_ids=[self.request['evidence']['event_id']],reason=self.reply['reason'][:240])
        verdict=self.guard.review(self.request['request_id'],proposal,current,now_elapsed_s=self.elapsed())
        self.note('reviewed',verdict=verdict,proposal=proposal,current=current)
        if not verdict['accepted']:self.pause('proposal_rejected');return
        self.start_node(node,'rm_plan_accepted')

    def start_node(self,node,reason):
        self.active_plan_node=node
        if node['skill_id']=='explore_then_resume':
            self.max_intervention_ticks=node['budget_ticks']
            self.phase='recovery';self.scheduled=('intervention',reason)
            self.recovery_expires=min(self.deadline,self.elapsed()+node['budget_s'])
            self.root_started=None
        else:
            self.phase='resumed';self.scheduled=('wood',reason)
            self.root_started=dict(tick=self.environment.gui['server_tick'],elapsed_s=self.elapsed())
            self.resumed_detector=StallDetector()
            self.resumed_detector.update(wood_sample(self.environment.gui,{},str(self.output)+':resumed'))

    def replan(self,*,event=None,attempt_request=None):
        if self.is_stopped() or goal_satisfied(self.snapshot()):return
        if self.provider.calls>=self.max_requests:
            self.pause('planning_request_budget_exhausted');return
        if self.deadline-self.elapsed()<=self.wait_limit_s+self.recovery_limit_s:
            self.pause('insufficient_remaining_time');return
        if event is not None:
            self.trigger=deepcopy(event);self.trigger_observation=deepcopy(self.environment.gui)
        self.pending_attempt_request=attempt_request
        self.generation+=1;self.phase='release';self.scheduled=None;self.root_started=None
        self.note('replan_scheduled',previous_request_id=self.request['request_id'])

    def resumed_stall(self,evidence):
        self.scheduler.observe_root(self.snapshot(),failure=evidence)
        self.scheduler.pause('root_stalled')
        if self.active_plan_node['on_failure']=='replan':self.replan(event=evidence)
        else:self.pause('root_stalled')

    def finish_recovery(self,reason):
        if self.attempt_start is None:return
        node=deepcopy(self.active_plan_node)
        super().finish_recovery(reason)
        receipt=self.attempts[-1];assessment=receipt['assessment']
        if (assessment.get('committed_ticks',0)>0 and assessment['classification']!='observation_unreliable'):
            # Base controller may have paused; the execution receipt still binds the node.
            self.scheduler.state['active_node']=node['id']
            self.scheduler.finish(node['id'],assessment,self.snapshot())
        if self.phase=='resumed':
            selected=self.choose_node(self.snapshot())
            if selected is not None:self.start_node(selected,reason)
        elif self.phase=='paused' and node['on_failure']=='replan':
            self.replan(attempt_request=self.request['request_id'])

    def prepare_control(self,**kwargs):
        if (self.phase=='resumed' and self.root_started is not None and not kwargs['should_stop']()
                and (self.environment.gui['server_tick']-self.root_started['tick']>=self.active_plan_node['budget_ticks']
                     or kwargs['elapsed']()-self.root_started['elapsed_s']>=self.active_plan_node['budget_s'])):
            self.pause('root_skill_budget_exhausted')
        super().prepare_control(**kwargs)

    def act(self,screen):
        if self.environment.reward is not self.root_reward_object:raise RuntimeError('original RM was replaced')
        action=super().act(screen)
        self.pending['rm_plan']=dict(revision=self.scheduler.state['revision'],
            node_id=self.scheduler.state['active_node'] or 'root',
            root_reference=deepcopy(self.scheduler.state['root']['reference']))
        return action

    def commit(self,action,reward,terminal):
        tag=deepcopy(self.pending.get('rm_plan'))
        super().commit(action,reward,terminal)
        self.student.rows[-1]['rm_plan']=tag
        if terminal and goal_satisfied(self.snapshot()):
            self.scheduler.observe_root(self.snapshot())
            self.scheduler.pause('goal_completed')

    def close(self):
        super().close()
        if hasattr(self,'scheduler'):
            current=self.snapshot()
            self.scheduler.observe_root(current)
            self.scheduler.pause('goal_completed' if goal_satisfied(current) else 'episode_closed')
