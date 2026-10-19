"""Experimental waypoint instances inside the existing root-preserving RM scheduler."""
from copy import deepcopy
import math
from pathlib import Path
from .adaptive_route import AdaptiveRoute
from .exploration_route import pose_only,POSE_KEYS
from .goal_recovery import GoalRecoveryController
from .replanning import ReplanGuard,context,goal_satisfied,validate_snapshot
from .rm_planner import fixture_plan,validate_plan
from .subgoal_control import NeuralSubgoalController,FixedSubgoalProvider
from .subgoal_planner import SubgoalScheduler
from .runtime import write_json
from .waypoint_skill import NAMES,COMPLETION,capability,assess_navigation,same_failed_route,distance
from .waypoints import arrived,goal_features
from .motion_blockage import target_blockage


def validate_waypoint_context(packet):
    data=packet.get('waypoint_context',{});pose=data.get('pose',{})
    if set(pose)!=set(POSE_KEYS):raise ValueError('missing bound waypoint pose')
    s=packet['runtime_binding'];nav=s.get('navigation',{})
    if ([pose['server_'+a] for a in 'xyz']!=nav.get('position') or pose['screen']!=0 or
            pose['server_available']!=1 or pose['error']!=0):raise ValueError('stale waypoint pose')
    candidates=data.get('candidates',[])
    if len(candidates)!=4 or [v.get('skill_id') for v in candidates]!=list(NAMES):raise ValueError('invalid waypoint candidates')
    for item in candidates:
        cap=packet['execution_registry'].get(item['skill_id'],{});args=cap.get('arguments',{})
        target=item.get('target');goal_features(pose,target)
        local=data.get('source')=='privileged_local_passability_v1'
        geometry=abs(distance(nav['position'],target)-3)<=1e-6 and target[1]==nav['position'][1]
        if local:
            from .local_navigation import checked_grid,validate_candidate
            observation=dict(pose,server_tick=s['tick'],local_navigation=data.get('local_navigation'))
            checked_grid(observation)
            geometry=validate_candidate(observation,target) if item.get('admissible') else target==nav['position']
        if (args.get('target')!=target or args.get('actor_sha256')!=data.get('actor_sha256') or not geometry or
                type(item.get('admissible')) is not bool or type(item.get('cost')) not in (int,float) or
                not math.isfinite(item['cost'])):raise ValueError('candidate differs from registered waypoint')


class NavigationScheduler(SubgoalScheduler):
    def finish_navigation(self,node,before,after,rows,*,reason,already_satisfied=False,exploration=None):
        validate_snapshot(before);validate_snapshot(after)
        plan=self.state['plans'][-1];original=next(n for n in plan['proposal']['steps'] if n['id']==node['id'])
        if (node!=original or node['skill_id'] not in NAMES or self.state['active_node']!=node['id'] or
                context(before)!=context(after) or after['sequence']-before['sequence']!=len(rows) or
                after['tick']-before['tick']!=len(rows) or len(rows)>node['budget_ticks']):
            raise ValueError('unbound waypoint outcome')
        cap=plan['registry'][node['skill_id']];assessment=assess_navigation(cap,rows,reason=reason)
        if already_satisfied:
            nav=after['navigation'];target=cap['arguments']['target']
            if rows or nav['gui_open'] or distance(nav['position'],target)>.35 or abs(nav['position'][1]-target[1])>1.5:
                raise ValueError('unproven already satisfied waypoint')
            assessment.update(outcome='completed',diagnosis='already_satisfied')
        outcome=assessment['outcome']
        self.update_timing(after)
        receipt=dict(type='neural_waypoint',request_id=plan['proposal']['request_id'],node_id=node['id'],
            method=deepcopy(cap),before=deepcopy(before),after=deepcopy(after),committed_steps=len(rows),
            outcome=outcome,reason=reason,assessment=assessment,predicate_satisfied=outcome=='completed',
            root_completed=goal_satisfied(after),already_satisfied=already_satisfied,classification=assessment['diagnosis'],
            end_sequence=after['sequence'],execution_authorized=False,rm_effect='preserve')
        if exploration is not None:receipt['exploration']=deepcopy(exploration)
        state=next(n for n in plan['nodes'] if n['id']==node['id']);state.update(status=outcome,receipt=receipt)
        self.state['active_node']=None;self.observe_root(after);self.save()
        return deepcopy(receipt)


class FixedNavigationProvider(FixedSubgoalProvider):
    kind='fixed_rm_navigation_fixture'
    def make_plan(self,packet):
        value=fixture_plan(packet);blocked={c['skill_id'] for c in packet['memory']['blocked_methods']}
        candidates=[c for c in packet['waypoint_context']['candidates'] if c['admissible'] and c['skill_id'] not in blocked]
        if not candidates:
            value.update(decision='pause',steps=[],reason='No untried executable navigation/approach chain in fixed fixture.')
            return value
        chosen=min(candidates,key=lambda c:c['cost'])
        move=dict(id='move',objective='Reach the bound candidate using the learned navigation head',
            skill_id=chosen['skill_id'],priority=80,depends_on=[],entry='active_root',completion=COMPLETION,
            budget_s=20.,budget_ticks=100,on_failure='replan')
        locate=dict(id='locate',objective='Observe a wood target after movement; skip if already known',
            skill_id='locate_wood',priority=75,depends_on=['move'],entry='active_root',completion='wood_target_known',
            budget_s=20.,budget_ticks=100,on_failure='replan')
        approach=dict(id='approach',objective='Regain observed wood interaction readiness',skill_id='approach_wood',
            priority=70,depends_on=['locate'],entry='active_root',completion='wood_interaction_ready',
            budget_s=20.,budget_ticks=100,on_failure='replan')
        if packet['waypoint_context'].get('source')=='privileged_local_passability_v1':
            approach.update(budget_s=30.,budget_ticks=200)
            locate.update(budget_s=30.,budget_ticks=200)
        root=value['steps'][-1];root['depends_on']=['approach'];value['steps']=[move,locate,approach,root]
        value['reason']='Fixed local integration fixture; no LLM inference.'
        return value
    def poll(self):
        if self.closed or self.packet is None or self.clock()<self.ready:return None
        return validate_plan(self.make_plan(self.packet),self.packet)


class RMNavigationController(NeuralSubgoalController):
    scheduler_type=NavigationScheduler

    def __init__(self,*args,navigation_checkpoint,navigation_strategy='visited',**kwargs):
        if navigation_strategy not in ('visited','coverage','coverage_motion','local_passability'):raise ValueError('invalid navigation strategy')
        self.navigation_strategy=navigation_strategy
        self.local_navigation=None;self.local_approach_target=None
        self.navigation_coverage_start=None
        super().__init__(*args,**kwargs)
        self.navigation_path=Path(navigation_checkpoint);self.navigation_start=None;self.route_memory=None
        self.navigation_refined=False
        self.waypoint_context=None;self.attach_navigation()

    def attach_navigation(self):
        from .navigation_model import load_navigation,NavigationSwitch
        from .world_store import digest
        saved=self.student.torch.load(self.navigation_path,map_location='cpu')
        if (saved.get('schema') not in ('waypoint_actor_v1','navigation_refinement_v1') or
                (saved['schema']=='navigation_refinement_v1' and saved.get('motion_dim')!=10) or
                digest(self.student.checkpoint)!=saved['parent_sha256'] or
                digest(self.student.embed_path)!=saved['embedding_sha256']):raise ValueError('incompatible navigation checkpoint')
        parent=self.student.agent.policy.pi_head;head,self.navigation_refined=load_navigation(parent,saved)
        head.eval().requires_grad_(False)
        self.nav_head=NavigationSwitch(parent,head,self.navigation_refined);self.student.agent.policy.pi_head=self.nav_head
        self.navigation_sha256=digest(self.navigation_path)

    def reset(self,goal=None):
        super().reset(goal);self.nav_head.goal=None
        if self.navigation_strategy=='local_passability':
            from .local_navigation import LocalNavigation
            from .subgoal_skills import capability as subgoal_capability
            self.local_navigation=LocalNavigation()
            self.local_navigation.observe(self.environment.gui)
            import numpy as np
            manifest=self.condition_metadata['planning']['neural_skills']
            for skill in ('locate_wood','approach_wood'):
                self.capabilities[skill]=subgoal_capability(skill,str(self.student.checkpoint),
                    self.embed_hashes['wood'],navigation_actor_sha256=self.navigation_sha256)
                entry=manifest[skill]
                entry['catalogue_capability']=deepcopy(entry['capability'])
                entry.update(capability=deepcopy(self.capabilities[skill]),embedding_sha256=self.embed_hashes['wood'],
                    executor='local passability planner + frozen navigation actor; root wood embedding')
                np.save(self.output/(skill+'-embed.npy'),self.wood_embed)
            write_json(self.output/'neural-skills.json',manifest)
            self.guard=ReplanGuard(self.guard.history,self.capabilities,allow_goal_conditioning=True,allow_subgoals=True)
            from .local_execution import configure_stall,LOCAL_STALL_WINDOWS,LOCAL_GOAL_WAIT,MINING_TEMPERATURE
            from .stall_replay import wood_sample
            configure_stall(self.detector,wood_sample(self.environment.gui,{},str(self.output)).stage)
            self.condition_metadata['planning']['local_execution']=dict(stall_windows=LOCAL_STALL_WINDOWS,
                goal_wait_ticks=LOCAL_GOAL_WAIT,mining_temperature=MINING_TEMPERATURE,
                decoding='tempered neural sampling only during current confirmed log mining',
                collection_budget_unchanged=True,episode_limit_unchanged=True)
        self.condition_metadata['planning'].update(waypoint_integration=True,navigation_actor=str(self.navigation_path),
            navigation_actor_sha256=self.navigation_sha256,experimental=True,
            navigation_strategy=self.navigation_strategy,
            navigation_input='motion_goal_10' if self.navigation_refined else 'relative_goal_6',
            diagnostics='observed hypotheses; no certified terrain',root_goal_and_reward_preserved=True)
        if self.local_navigation is not None:
            self.condition_metadata['planning'].update(diagnostics='bounded privileged local passability; conservative level walking only',
                observation_condition='server pose + local collision/support grid + camera surface rays + wood lock; not video-only',
                wood_detection='center ray plus occlusion-tested camera rays: yaw +/-40, pitch +/-20, spacing 5 degrees, range 16')

    def navigation_receipts(self):
        return [n['receipt'] for p in self.scheduler.state['plans'] for n in p['nodes']
                if n.get('receipt',{} ) and n['receipt'].get('type')=='neural_waypoint']

    def blocked_navigation(self,cap):
        current=self.environment.gui;s=self.snapshot()
        if self.local_navigation is not None and self.waypoint_context is not None:
            if any(c['skill_id']==cap['skill_id'] and not c['admissible'] for c in self.waypoint_context['candidates']):
                return True
        if self.route_memory.blocked(current,cap['arguments']['target']):return True
        return any(same_failed_route(cap,current,r,world_id=s['world_id'],episode_id=s['episode_id']) for r in self.navigation_receipts())

    def create_request(self,s):
        g=self.environment.gui;history=[[r['server_before']['server_'+a] for a in 'xyz'] for r in self.rows]
        self.route_memory=AdaptiveRoute(pose_only(g),history,
            blocked_directions=self.route_memory.blocked_directions if self.route_memory else None,
            windowed_motion=self.navigation_strategy in ('coverage_motion','local_passability'))
        choices=self.route_memory.candidates;candidates=[];exploration=None
        if self.navigation_strategy in ('coverage','coverage_motion'):
            from .exploration_coverage import score_candidates
            choices,exploration=score_candidates(self.spatial,choices)
        if self.local_navigation is not None:
            choices=self.local_navigation.choices(g)
            while len(choices)<4:
                choices.append(dict(target=[g['server_'+a] for a in 'xyz'],cost=1000.,admissible=False))
        for index,c in enumerate(choices):
            cap=capability(index,str(self.student.checkpoint),self.navigation_sha256,self.embed_hashes['wood'],c['target'])
            self.capabilities[cap['skill_id']]=cap
            candidates.append(dict(skill_id=cap['skill_id'],target=deepcopy(c['target']),cost=c['cost'],admissible=c['admissible']))
        self.waypoint_context=dict(pose=pose_only(g),candidates=candidates,actor_sha256=self.navigation_sha256,
            source='privileged_server_pose_and_observed_motion',terrain='unknown')
        if self.local_navigation is not None:
            self.waypoint_context.update(source='privileged_local_passability_v1',terrain='conservative_level_walking',
                local_navigation=deepcopy(g['local_navigation']))
        if exploration is not None:self.waypoint_context['exploration']=exploration
        if self.navigation_strategy=='coverage_motion':
            self.waypoint_context['motion_evidence']=deepcopy(self.route_memory.blocked_directions)
        self.guard=ReplanGuard(self.guard.history,self.capabilities,allow_goal_conditioning=True,allow_subgoals=True,allow_waypoints=True)
        return super().create_request(s)

    def make_packet(self,s,evidence):
        packet,allowed=super().make_packet(s,evidence)
        packet['waypoint_context']=deepcopy(self.waypoint_context)
        if self.local_navigation is not None:
            packet['observation_source']='privileged server pose + bounded local passability grid + observed wood lock'
        packet['spatial_memory']['waypoint_execution_supported']=True
        blocked=[cap for cap in self.capabilities.values() if cap['skill_id'] in NAMES and self.blocked_navigation(cap)]
        memory=packet['memory'];prior={c['skill_id'] for c in memory['blocked_methods']}
        memory['blocked_methods']+=deepcopy([c for c in blocked if c['skill_id'] not in prior])
        blocked_names={c['skill_id'] for c in memory['blocked_methods']}
        memory['available_skills']=[c for c in memory['available_skills'] if c['skill_id'] not in blocked_names]
        write_json(self.output/f'planner-input-{self.provider.calls+1:02d}.json',packet)
        return packet,allowed

    def choose_node(self,current):
        for _ in range(8):
            node=super().choose_node(current)
            if node is None or node['skill_id'] not in NAMES:return node
            cap=self.scheduler.state['plans'][-1]['registry'][node['skill_id']]
            if self.blocked_navigation(cap):self.pause('same_failed_navigation');return None
            if not arrived(self.environment.gui,cap['arguments']['target']):return node
            self.scheduler.finish_navigation(node,current,current,[],reason='predicate_satisfied',already_satisfied=True)
            write_json(self.output/'navigation-receipts.json',self.navigation_receipts())
        self.pause('no_executable_navigation');return None

    def review_reply(self,current):
        old=self.packet['runtime_binding'].get('navigation',{}).get('position')
        now=current.get('navigation',{}).get('position')
        if old is None or now is None or math.dist(old,now)>.5:
            self.pause('stale_waypoint_pose');return
        return super().review_reply(current)

    def start_node(self,node,reason):
        self.nav_head.goal=None
        if node['skill_id'] not in NAMES:
            result=super().start_node(node,reason)
            if self.local_navigation is not None and node['skill_id'] in ('locate_wood','approach_wood'):
                self.local_approach_target=None
                self.intervention_embed=self.wood_embed.copy();self.embed_hashes['intervention']=self.embed_hashes['wood']
            if self.local_navigation is not None and self.phase=='resumed':
                from .local_execution import configure_stall
                from .stall_replay import wood_sample
                configure_stall(self.resumed_detector,wood_sample(self.environment.gui,{},str(self.output)).stage)
            return result
        self.navigation_cap=deepcopy(self.scheduler.state['plans'][-1]['registry'][node['skill_id']])
        self.active_plan_node=node;self.phase='waypoint';self.root_started=None;self.navigation_start=self.snapshot()
        if self.navigation_strategy in ('coverage','coverage_motion'):self.navigation_coverage_start=self.spatial.summary()
        self.intervention_embed=self.wood_embed.copy();self.embed_hashes['intervention']=self.embed_hashes['wood']
        self.max_intervention_ticks=node['budget_ticks'];self.scheduled=('intervention','rm_waypoint:'+reason)
        self.note('navigation_started',node=node,capability=self.navigation_cap,entry=self.navigation_start)

    def finish_navigation(self,reason):
        if self.navigation_start is None:return
        before=self.navigation_start;node=deepcopy(self.active_plan_node)
        rows=self.rows[before['sequence']:];current=self.snapshot()
        exploration=None
        if self.navigation_coverage_start is not None:
            from .exploration_coverage import coverage_delta
            exploration=coverage_delta(self.navigation_coverage_start,self.spatial.summary())
        receipt=self.scheduler.finish_navigation(node,before,current,rows,reason=reason,exploration=exploration)
        self.note('navigation_finished',receipt=receipt)
        self.navigation_start=None;self.navigation_coverage_start=None;self.nav_head.goal=None;self.scheduled=None
        write_json(self.output/'navigation-receipts.json',self.navigation_receipts())
        if self.is_stopped() or reason in ('stop','environment_done','control_budget') or goal_satisfied(current):
            self.phase='ended';return
        if receipt['outcome']=='completed':
            next_node=self.choose_node(current)
            if next_node is not None:self.start_node(next_node,'observed_waypoint_arrival')
        elif receipt['outcome']=='failed' and node['on_failure']=='replan':
            self.pending_subgoal_failure=True;self.replan()
        else:self.pause('navigation_'+receipt['outcome'])

    def prepare_control(self,**kwargs):
        if self.phase=='waypoint' and self.navigation_start is not None and not kwargs['should_stop']():
            if kwargs['elapsed']()-self.navigation_start['elapsed_s']>=self.active_plan_node['budget_s']:
                self.finish_navigation('budget_exhausted')
        super().prepare_control(**kwargs)

    def act(self,screen):
        self.nav_head.goal=None
        from .local_execution import mining_temperature
        self.student.action_temperature=(mining_temperature(self.environment.gui,self.phase)
            if self.local_navigation is not None else 1.)
        if (self.local_navigation is not None and self.phase=='subgoal'
                and self.active_plan_node['skill_id'] in ('locate_wood','approach_wood')):
            from .navigation_refinement import motion_goal
            choice=self.local_navigation.waypoint(self.environment.gui)
            if choice is not None:self.local_approach_target=choice['target']
            elif abs(self.environment.gui['server_y']-(self.local_navigation.floor or 0))<=.15:
                self.local_approach_target=None
            target=self.local_approach_target
            if target is not None:
                previous=self.rows[-1]['server_before'] if self.rows else None
                self.nav_head.goal=(motion_goal(self.environment.gui,target,previous,
                    remaining=min(100,self.active_plan_node['budget_ticks']-(len(self.rows)-self.subgoal_start['sequence'])))
                    if self.navigation_refined else goal_features(self.environment.gui,target))
            action=NeuralSubgoalController.act(self,screen)
            self.pending.update(waypoint_goal=deepcopy(self.nav_head.goal),
                local_route=dict(target=deepcopy(target),choice=choice,stage=self.active_plan_node['skill_id'],source='privileged_local_passability_v1',
                    status='learned_route_following' if target is not None else 'no_route_collection_actor'))
            return action
        if self.phase!='waypoint':return super().act(screen)
        if self.environment.reward is not self.root_reward_object:raise RuntimeError('original RM replaced')
        target=self.navigation_cap['arguments']['target']
        if self.navigation_refined:
            from .navigation_refinement import motion_goal
            previous=self.rows[-1]['server_before'] if self.rows else None
            if self.rows and self.rows[-1]['server_after']!=self.environment.gui:
                raise ValueError('navigation motion requires the last committed observation')
            remaining=self.active_plan_node['budget_ticks']-(len(self.rows)-self.navigation_start['sequence'])
            self.nav_head.goal=motion_goal(self.environment.gui,target,previous,remaining=remaining)
        else:self.nav_head.goal=goal_features(self.environment.gui,target)
        action=GoalRecoveryController.act(self,screen)
        self.pending.update(waypoint_goal=list(self.nav_head.goal),waypoint_target=deepcopy(target),
            rm_plan=dict(revision=self.scheduler.state['revision'],node_id=self.scheduler.state['active_node'],
                root_reference=deepcopy(self.scheduler.state['root']['reference'])))
        return action

    def commit(self,action,reward,terminal):
        if self.phase!='waypoint':
            before=self.pending['server_before'];local=deepcopy(self.pending.get('local_route'))
            goal=deepcopy(self.pending.get('waypoint_goal'))
            super().commit(action,reward,terminal)
            if local is not None:
                self.student.rows[-1].update(waypoint_goal=goal,local_route=local)
                if local['target'] is not None and target_blockage(self.rows[-24:],local['target']):
                    self.local_navigation.reject(local['target'],self.environment.gui['server_tick'])
                    self.local_approach_target=None
            if (self.local_navigation is not None and self.environment.gui.get('screen')==0
                    and self.environment.gui.get('server_alive')==1 and self.environment.gui.get('error')==0):
                self.local_navigation.observe(self.environment.gui)
            if self.route_memory and self.environment.gui.get('error')==0:
                # Other skills can refute a blockage by crossing it; their mining actions
                # must not become evidence of navigation blockage.
                self.route_memory.observe_action(before,self.environment.gui,{},sequence=len(self.rows))
            return
        goal=deepcopy(self.pending['waypoint_goal']);tag=deepcopy(self.pending['rm_plan']);self.snapshot()
        GoalRecoveryController.commit(self,action,reward,terminal)
        self.student.rows[-1].update(waypoint_goal=goal,rm_plan=tag)
        self.scheduled=None;row=self.rows[-1]
        self.route_memory.observe_action(row['server_before'],row['server_after'],row['action'],sequence=len(self.rows))
        if self.is_stopped():self.finish_navigation('stop')
        elif terminal:self.finish_navigation('environment_done')
        elif arrived(self.environment.gui,self.navigation_cap['arguments']['target']):self.finish_navigation('predicate_satisfied')
        elif (self.navigation_strategy in ('coverage_motion','local_passability') and self.route_memory.latest_blockage and
                self.route_memory.blocked(self.environment.gui,self.navigation_cap['arguments']['target']) and
                target_blockage(self.rows[self.navigation_start['sequence']:],self.navigation_cap['arguments']['target'])):
            self.finish_navigation('observed_blockage')
        elif len(self.rows)-self.navigation_start['sequence']>=self.active_plan_node['budget_ticks']:
            self.finish_navigation('budget_exhausted')

    def close(self):
        if self.navigation_start is not None:self.finish_navigation('control_budget' if self.stop_reason()=='time_limit' else 'stop')
        self.nav_head.goal=None;super().close()
