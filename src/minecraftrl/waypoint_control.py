"""Development route or visit-history goals with learned actions; preserve the wood RM."""
from copy import deepcopy
import time

from .rl_policy import cpu_tree,state_fingerprint
from .runtime import jsonable,write_json,run_episode
from .spatial_memory import SpatialMemory
from .stall_detection import StallDetector
from .stall_replay import wood_sample
from .student_correction import recovery_waypoints
from .waypoints import goal_features,world_targets,RouteProgress
from .world_store import digest


class WaypointController:
    def __init__(self,student,checkpoint,world,output,*,start_mode='stall',planner_mode='fixture'):
        if start_mode not in ('stall','initial'):raise ValueError('invalid waypoint start mode')
        if planner_mode not in ('fixture','visit_memory'):raise ValueError('invalid waypoint planner')
        from .waypoint_actor import navigation_head,SwitchHead
        saved=student.torch.load(checkpoint,map_location='cpu')
        if (saved.get('schema')!='waypoint_actor_v1' or digest(student.checkpoint)!=saved['parent_sha256']
                or digest(student.embed_path)!=saved['embedding_sha256']):
            raise ValueError('navigation weights do not match parent/embedding')
        if world.get('experimental_split')!='development':raise ValueError('waypoint diagnostic requires development world')
        original=student.agent.policy.pi_head;navigation=navigation_head(original)
        navigation.load_state_dict(saved['actor']);navigation.eval().requires_grad_(False)
        self.head=SwitchHead(original,navigation);student.agent.policy.pi_head=self.head
        self.student,self.world,self.output=student,world,output;self.actor_path=str(checkpoint)
        self.actor_sha256=digest(checkpoint);self.rows=[];self.decisions=[];self.route=None
        self.is_stopped=lambda:False;self.environment=None;self.pending=None;self.ended=False
        self.mode='root';self.trigger=None;self.route_start=None;self.failure=None;self.switches=[]
        self.start_mode=start_mode
        self.planner_mode=planner_mode

    def set_reward_context(self,context):self.student.set_reward_context(context)

    def reset(self,goal=None):
        if self.rows:raise RuntimeError('waypoint controller is single-use')
        self.student.reset(goal);self.root_reward=self.environment.reward
        self.detector=StallDetector();self.detector.update(wood_sample(self.environment.gui,{},str(self.output)))
        self.spatial=SpatialMemory(self.world['sha256'],str(self.output),source='privileged_server',dimension='overworld')
        self.observe_space()
        if self.start_mode=='initial':self.trigger=dict(event='diagnostic_route_start',source='explicit initial-state evaluation')
        source='known development fixture; privileged_server' if self.planner_mode=='fixture' else 'own pose and visits; no wall geometry; privileged_server'
        self.condition_metadata=dict(self.student.condition_metadata,waypoint_actor=self.actor_path,
            waypoint_actor_sha256=self.actor_sha256,goal_source=source,waypoint_planner=self.planner_mode,
            direct_action_overrides=False,root_actor_preserved=True,route_budget_actions=400,route_budget_s=60,
            waypoint_start=self.start_mode)

    def observe_space(self):
        g=self.environment.gui
        if self.spatial.last is None or g['server_tick']>self.spatial.last['tick']:
            self.spatial.observe([float(g['server_'+a]) for a in 'xyz'],g['server_tick'])

    def prepare_control(self,*,elapsed,limit_s,should_stop,request_stop,emit):
        if should_stop() or self.ended:return
        if self.trigger and self.route is None:
            if self.planner_mode=='fixture':
                self.route=RouteProgress(world_targets(self.world,recovery_waypoints(self.world,self.environment.gui)))
            else:
                from .exploration_route import ExplorationRoute,pose_only
                history=[[r['server_before']['server_'+a] for a in 'xyz'] for r in self.rows]
                self.route=ExplorationRoute(pose_only(self.environment.gui),history)
            self.mode='waypoint';self.route_start=dict(sequence=len(self.rows),elapsed_s=elapsed())
            self.switches.append(dict(sequence=len(self.rows),mode=self.mode,targets=deepcopy(self.route.targets),
                planner=self.planner_mode,trigger=deepcopy(self.trigger),source=self.condition_metadata['goal_source'],
                memory_sha256=state_fingerprint(self.student.agent.hidden_state)))
            write_json(self.output/'waypoint-route.json',self.switches[-1])
        if self.mode=='waypoint':
            self.route.observe(self.environment.gui,sequence=len(self.rows)-self.route_start['sequence'])
            if self.route.completed:
                self.mode='resumed';self.head.goal=None
                self.switches.append(dict(sequence=len(self.rows),mode=self.mode,
                    memory_sha256=state_fingerprint(self.student.agent.hidden_state)))
            else:
                if self.environment.gui['screen']!=0:self.failure='gui_open'
                elif getattr(self.route,'exhausted',False):self.failure='exploration_candidates_exhausted'
                elif len(self.rows)-self.route_start['sequence']>=400 or elapsed()-self.route_start['elapsed_s']>=60:
                    self.failure='waypoint_budget_exhausted'
                if self.failure:request_stop('planning_paused')

    def act(self,screen):
        if self.ended or self.is_stopped():raise InterruptedError('STOP before waypoint inference')
        if self.environment.reward is not self.root_reward:raise RuntimeError('root RM replaced')
        if self.pending is not None:raise RuntimeError('uncommitted waypoint action')
        goal=None;target=None
        if self.mode=='waypoint':
            target=self.route.targets[self.route.index];goal=goal_features(self.environment.gui,target)
        self.head.goal=goal
        action=self.student.act(screen);self.controller_name='steve1_'+self.mode
        self.pending=dict(mode=self.mode,goal=goal,target=target,action=jsonable(action),
            server_before=deepcopy(self.environment.gui),frame=f'frames/{len(self.rows):06d}.png')
        self.student.pending['waypoint_goal']=goal
        return action

    def commit(self,action,reward,terminal):
        if self.pending is None or jsonable(action)!=self.pending['action']:raise ValueError('waypoint dispatch mismatch')
        self.student.commit(action,reward,terminal)
        self.rows.append(dict(self.pending,reward=reward,terminal=terminal,server_after=deepcopy(self.environment.gui)))
        self.pending=None;self.observe_space()
        if self.is_stopped() or terminal:self.ended=True;return
        if self.mode=='root':
            decision=self.detector.update(wood_sample(self.environment.gui,jsonable(action),str(self.output),
                previous=self.rows[-1]['server_before'],terminal=terminal));self.decisions.append(decision)
            if decision['event']=='stalled':self.trigger=deepcopy(decision)
        elif self.mode=='waypoint':
            self.route.observe(self.environment.gui,sequence=len(self.rows)-self.route_start['sequence'])

    def close(self):self.ended=True;self.head.goal=None


def run_waypoint(policy,env,output,*,world,gate,stop_file,**kwargs):
    from .rl_collect import RLEnvironment
    from .rl_core import LOGS
    stopped=lambda:gate.requested.is_set() or bool(stop_file and stop_file.exists())
    wrapper=RLEnvironment(env,policy,strict_ticks=True,reward_mode='wood-potential-v4',is_stopped=stopped)
    policy.environment,policy.is_stopped=wrapper,stopped
    def closed(report):write_json(output/'rl-control-finished.json',dict(control_closed=True,release=report['release'],
        cancelled_archive=report['status']=='cancelled',monotonic_s=time.monotonic()))
    try:
        report=run_episode(policy,wrapper,output,goal='collect wood',target_items=LOGS,
            completion_check=lambda obs:wrapper.reward.completed,gate=gate,stop_file=stop_file,
            on_control_closed=closed,**kwargs)
        receipts=list((output/'candidate-exchange').glob('request-*/api-receipt.json'))
        import json
        actual_calls=sum(json.loads(p.read_text(encoding='utf-8')).get('network_calls',0) for p in receipts)
        report.update(controller='steve1_waypoint_development',rl_updates=0,actual_llm_calls=actual_calls,
            rl_training_eligible=False,condition=getattr(policy,'condition_metadata',{}),initial_world=world,
            route_started=policy.route is not None,route_completed=bool(policy.route and policy.route.completed),
            route_receipts=policy.route.receipts if policy.route else [],route_failure=policy.failure,
            mode_steps={mode:sum(r['mode']==mode for r in policy.rows) for mode in ('root','waypoint','resumed','root_probe','planner_neutral')},
            root_inventory_success=bool(wrapper.reward and wrapper.reward.completed),direct_action_overrides=0,
            scope=getattr(policy,'condition_metadata',{}).get('goal_source','initialization incomplete')+'; learned motor actions; development only')
        write_json(output/'waypoint-rows.json',policy.rows);write_json(output/'waypoint-switches.json',policy.switches)
        write_json(output/'stall-decisions.json',policy.decisions);write_json(output/'initial-state.json',getattr(wrapper,'initial',{}))
        if policy.planner_mode in ('visit_memory','adaptive_rule','adaptive_llm'):
            write_json(output/'exploration-route.json',dict(
                decisions=getattr(policy.route,'decisions',[]),failures=getattr(policy.route,'failures',[]),
                targets=getattr(policy.route,'targets',[]),blocked_directions=getattr(policy.route,'blocked_directions',[]),
                interruptions=getattr(policy.route,'interruptions',[]),
                probes=getattr(policy,'probes',[]),scope='speculative goals; unknown traversability'))
        if hasattr(policy,'spatial'):policy.spatial.save(output/'spatial-memory.json')
        if hasattr(policy.student,'initial_state'):
            policy.student.torch.save(dict(rows=policy.student.rows,condition=getattr(policy,'condition_metadata',{}),eligible=False,
                initial_recurrent_state=policy.student.initial_state),output/'waypoint-trajectory.pt')
        write_json(output/'summary.json',report);return report
    finally:policy.student.dispose()
