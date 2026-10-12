"""Eval-only LLM-selectable neural options with server-defined termination."""
from copy import deepcopy

import numpy as np

from .goal_recovery import GoalRecoveryController
from .replanning import ReplanGuard, goal_satisfied, validate_snapshot
from .rm_planner import validate_plan
from .rm_planning_control import RMPlanningController, FixedRMPlanProvider
from .runtime import write_json
from .subgoal_planner import SubgoalScheduler, blocked_subgoal, subgoal_fixture
from .subgoal_skills import SKILLS, capability, completion, navigation_state
from .spatial_memory import SpatialMemory


class FixedSubgoalProvider(FixedRMPlanProvider):
    kind='fixed_neural_subgoal_fixture'
    def poll(self):
        if self.closed or self.packet is None or self.clock()<self.ready:return None
        return validate_plan(subgoal_fixture(self.packet),self.packet)


class NeuralSubgoalController(RMPlanningController):
    scheduler_type=SubgoalScheduler

    def __init__(self,*args,catalogue,**kwargs):
        super().__init__(*args,**kwargs)
        if set(catalogue)!=set(SKILLS):raise ValueError('incomplete neural skill catalogue')
        self.catalogue=catalogue
        self.exploration_embed=self.intervention_embed.copy()
        self.subgoal_start=None;self.pending_subgoal_failure=False

    def reset(self,goal=None):
        super().reset(goal)
        # Current fixed development arenas are overworld-only; no cross-world map reuse.
        self.spatial=SpatialMemory(self.world_id,str(self.output),source='privileged_server',dimension='overworld')
        self.snapshot()
        self.condition_metadata['intervention']['embedding_sha256']=deepcopy(self.embed_hashes)
        manifest={}
        for name,item in self.catalogue.items():
            cap=capability(name,str(self.student.checkpoint),item['sha256'])
            self.capabilities[name]=cap
            np.save(self.output/(name+'-embed.npy'),item['embedding'])
            manifest[name]=dict(embedding_sha256=item['sha256'],provenance=item['provenance'],capability=cap)
        self.guard=ReplanGuard(self.guard.history,self.capabilities,allow_goal_conditioning=True,allow_subgoals=True)
        write_json(self.output/'neural-skills.json',manifest)
        self.condition_metadata['planning'].update(abstract_navigation=True,neural_skills=manifest,
            outcome_scope='experimental options; predicate satisfaction is not root inventory success')

    def snapshot(self):
        s=super().snapshot()
        # Dead/invalid observations remain representable for release/closure, never continuation.
        if s['facts']['alive']:
            s['navigation']=navigation_state(self.environment.gui,self.baseline)
            if hasattr(self,'spatial'):
                nav=s['navigation']
                if self.spatial.last is None or nav['tick']>self.spatial.last['tick']:
                    self.spatial.observe([float(v) for v in nav['position']],nav['tick'])
        validate_snapshot(s)
        return s

    def make_packet(self,s,evidence):
        packet,allowed=super().make_packet(s,evidence)
        memory=packet['memory'];registry=packet['execution_registry']
        blocked=[c for c in registry.values() if blocked_subgoal(s,c,self.guard.history.data,self.scheduler.state['plans'])]
        memory['blocked_methods']=deepcopy(blocked)
        memory['available_skills']=[deepcopy(c) for c in registry.values() if c not in blocked]
        packet['spatial_memory']=self.spatial.summary()
        self.spatial.save(self.output/'spatial-memory.json')
        write_json(self.output/f'planner-input-{self.provider.calls+1:02d}.json',packet)
        return packet,allowed

    def create_request(self,s):
        if self.pending_subgoal_failure:
            request=self.scheduler.request_after_failure(self.guard,s)
            self.pending_subgoal_failure=False
            return request,request['evidence']
        return super().create_request(s)

    def start_node(self,node,reason):
        if node['skill_id'] not in SKILLS:
            if node['skill_id']=='explore_then_resume':
                self.intervention_embed=self.exploration_embed.copy()
                self.embed_hashes['intervention']=self.capabilities['explore_then_resume']['arguments']['explore_sha256']
            return super().start_node(node,reason)
        item=self.catalogue[node['skill_id']]
        self.active_plan_node=node;self.phase='subgoal';self.root_started=None
        self.intervention_embed=item['embedding'].copy()
        self.embed_hashes['intervention']=item['sha256']
        self.subgoal_start=self.snapshot()
        self.max_intervention_ticks=node['budget_ticks']
        self.scheduled=('intervention',node['skill_id']+':'+reason)
        self.note('subgoal_started',node=node,abstract_state=self.subgoal_start['navigation'])

    def finish_subgoal(self,reason):
        if self.subgoal_start is None:return
        node=deepcopy(self.active_plan_node)
        current=self.snapshot()
        if 'navigation' not in current:
            self.note('subgoal_unverified',node=node,reason='invalid_final_navigation')
            self.subgoal_start=None;self.pause('invalid_final_navigation');return
        receipt=self.scheduler.finish_subgoal(node,self.subgoal_start,current,reason=reason)
        self.note('subgoal_finished',receipt=receipt)
        self.subgoal_start=None;self.scheduled=None
        if self.is_stopped() or reason in ('stop','environment_done') or goal_satisfied(current):
            self.phase='ended';return
        if receipt['outcome']=='unrun':
            self.pause('subgoal_expired_before_dispatch');return
        if receipt['outcome']=='completed':
            selected=self.choose_node(current)
            if selected is not None:self.start_node(selected,'observed_subgoal_completion')
        elif node['on_failure']=='replan':
            self.pending_subgoal_failure=True
            self.replan()
        else:self.pause('subgoal_failed')

    def prepare_control(self,**kwargs):
        if self.phase=='subgoal' and self.subgoal_start is not None and not kwargs['should_stop']():
            node=self.active_plan_node
            if (self.environment.gui['server_tick']-self.subgoal_start['tick']>=node['budget_ticks']
                    or kwargs['elapsed']()-self.subgoal_start['elapsed_s']>=node['budget_s']):
                self.finish_subgoal('budget_exhausted')
        super().prepare_control(**kwargs)

    def act(self,screen):
        if self.environment.reward is not self.root_reward_object:raise RuntimeError('original RM was replaced')
        if self.phase=='subgoal':
            action=GoalRecoveryController.act(self,screen)
            self.pending['rm_plan']=dict(revision=self.scheduler.state['revision'],
                node_id=self.scheduler.state['active_node'],root_reference=deepcopy(self.scheduler.state['root']['reference']))
        else:action=super().act(screen)
        self.pending['subgoal_context']=dict(skill_id=(self.active_plan_node or {}).get('skill_id','collect_wood'),
            phase=self.phase,abstract_state=navigation_state(self.environment.gui,self.baseline))
        return action

    def commit(self,action,reward,terminal):
        self.snapshot()  # Observe every committed server state, including neutral transitions.
        tag=deepcopy(self.pending.get('subgoal_context'));rm_tag=deepcopy(self.pending.get('rm_plan'))
        if self.phase!='subgoal':
            super().commit(action,reward,terminal)
        else:
            GoalRecoveryController.commit(self,action,reward,terminal)
            self.student.rows[-1]['rm_plan']=rm_tag
            self.scheduled=None  # This option's predicate, not the legacy exploration timer, owns termination.
            if self.is_stopped():self.finish_subgoal('stop')
            elif terminal:self.finish_subgoal('environment_done')
            elif completion(self.active_plan_node['skill_id'],self.subgoal_start['navigation'],
                            navigation_state(self.environment.gui,self.baseline)):
                self.finish_subgoal('predicate_satisfied')
            elif len(self.rows)-self.subgoal_start['sequence']>=self.active_plan_node['budget_ticks']:
                self.finish_subgoal('budget_exhausted')
        self.student.rows[-1]['subgoal_context']=tag

    def close(self):
        if self.subgoal_start is not None:self.finish_subgoal('stop')
        super().close()
        if hasattr(self,'spatial'):self.spatial.save(self.output/'spatial-memory.json')
