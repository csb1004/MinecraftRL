"""Explicit navigation demonstrations or learned-only trials under the normal supervisor."""
from copy import deepcopy
import json,time,math
from pathlib import Path
from .navigation_refinement import TrialState,motion_goal,local_targets,navigation_action_selection
from .runtime import run_episode,jsonable,write_json
from .rl_policy import cpu_tree,guided,state_fingerprint
from .rl_collect import RLEnvironment
from .world_store import digest


class NavigationTrial:
    def __init__(self,student,raw,world,output,config,root):
        if world.get('kind')=='natural_navigation':
            from .natural_navigation import validate_natural_trial
            validate_natural_trial(world,config)
        if world.get('kind')=='step_navigation':
            from .step_navigation import validate_step_trial
            validate_step_trial(world,config)
        from .navigation_model import load_navigation,NavigationSwitch,value_head
        self.student,self.raw,self.world,self.output=student,raw,world,output
        self.config=config;self.mode=config['mode'];self.rows=[];self.pending=None;self.previous=None
        self.action_selection=navigation_action_selection(config)
        self.delivered=None
        if self.action_selection=='delivered_mode':
            from .navigation_action_likelihood import DeliveredActionLikelihood
            self.delivered=DeliveredActionLikelihood(student.agent.action_mapper).to(student.device)
        self.pose_source=config.get('pose_source','server')
        self.is_stopped=lambda:False;self.ended=False;self.environment=None
        expected='train' if world['experimental_split']=='train' else 'development'
        if config['split']!=expected or (self.mode=='ppo' and expected!='train'):raise ValueError('navigation split mismatch')
        self.state=TrialState(local_targets(world,config['points']),self.mode,
            student_ticks=config.get('student_ticks',32),arrival_hold_ticks=config.get('arrival_hold_ticks',1),
            pose_source=self.pose_source)
        path=(root/config['actor']).resolve()
        if not path.is_relative_to(root/'results') or digest(path)!=config['actor_sha256']:raise ValueError('unbound navigation actor')
        saved=student.torch.load(path,map_location='cpu')
        if saved['parent_sha256']!=digest(student.checkpoint) or saved['embedding_sha256']!=digest(student.embed_path):
            raise ValueError('navigation parent mismatch')
        self.actor_hash=digest(path);self.saved=saved
        parent=student.agent.policy.pi_head;head,refined=load_navigation(parent,saved)
        self.switch=NavigationSwitch(parent,head.eval().requires_grad_(False),refined)
        student.agent.policy.pi_head=self.switch
        self.value=value_head().to(student.device).eval().requires_grad_(False)
        if 'navigation_value' in saved:self.value.load_state_dict(saved['navigation_value'])
        else:
            for p in self.value.parameters():student.torch.nn.init.zeros_(p)

    def set_reward_context(self,context):self.student.set_reward_context(context)

    def reset(self,goal=None):
        self.student.reset('collect wood');self.root_reward=self.environment.reward
        self.student.condition_metadata['navigation_root_context_mode']=self.saved.get('root_context_mode','observed')
        if self.delivered is not None:
            self.student.condition_metadata.update(navigation_action_selection=self.action_selection,
                navigation_behavior_likelihood='point_mass',navigation_proposal_likelihood='delivered_command')
        write_json(self.output/'rl-condition.json',self.student.condition_metadata)
        self.root_fingerprint=state_fingerprint(self.switch.root.state_dict())
        self.state.start(self.environment.gui)

    def prepare_control(self,**kwargs):
        if self.state.finished:kwargs['request_stop']('planning_paused')

    def act(self,screen):
        if self.is_stopped() or self.ended or self.state.finished:raise InterruptedError('STOP before navigation action')
        if self.pending is not None:raise RuntimeError('uncommitted navigation action')
        if self.environment.reward is not self.root_reward:raise RuntimeError('root reward replaced')
        from .navigation_model import append_motion
        from .fixture_probe import navigate
        import numpy as np
        student=self.student;torch=student.torch;g=deepcopy(self.environment.gui)
        goal=motion_goal(g,self.state.target,self.previous,remaining=100-self.state.count,pose_source=self.pose_source)
        self.switch.goal=goal
        # All owners encode their actual pre-action screen, so teacher frames retain real student history.
        features,memory=student.features(screen)
        with torch.no_grad():
            pd=guided(self.switch,features)
            if any(not torch.isfinite(v).all() for v in pd.values()):raise ValueError('nonfinite navigation distribution')
            if self.state.owner=='student':
                action=self.delivered.mode(pd) if self.delivered is not None else self.switch.sample(pd,deterministic=False)
            else:
                env_action=self.raw.action_space.noop()
                if self.state.owner=='teacher':
                    if self.world.get('kind')=='step_navigation':
                        from .step_navigation import step_teacher
                        changes,_=step_teacher(g,self.state.target,self.world,pose_source=self.pose_source)
                    else:changes,_=navigate(g,self.state.target[0],self.state.target[2],pose_source=self.pose_source)
                    env_action.update(changes)
                factored=student.agent.action_transformer.env2policy({k:np.asarray(v) for k,v in env_action.items()})
                mapped=student.agent.action_mapper.from_factored({k:v[None] for k,v in factored.items()})
                action={k:torch.as_tensor(v,device=student.device).unsqueeze(1) for k,v in mapped.items()}
            value=self.value(append_motion(features[0:1],goal)).item()
            logp=0. if self.delivered is not None else self.switch.logprob(action,pd).item()
            if not math.isfinite(value) or not math.isfinite(logp):raise ValueError('nonfinite navigation value/logp')
            proposal_logp=self.delivered(pd,action).item() if self.delivered is not None else None
            if proposal_logp is not None and not math.isfinite(proposal_logp):raise ValueError('nonfinite proposal likelihood')
        env_action=student.agent._agent_action_to_env({k:v[:,0] for k,v in action.items()})
        self.controller_name='navigation_'+self.state.owner
        self.pending=dict(owner=self.state.owner,index=self.state.index,target=list(self.state.target),goal=goal,
            server_before=g,action=jsonable(env_action),frame=f'frames/{len(self.rows):06d}.png')
        if self.state.arrival.required_ticks>1:self.pending['arrival_hold_before']=self.state.arrival.count
        student.pending=dict(features=features.cpu(),action=cpu_tree(action),old_logp=logp,value=value,
            first=student.first,env_action=jsonable(env_action),motion_goal=goal,owner=self.state.owner,index=self.state.index)
        if proposal_logp is not None:student.pending['proposal_delivered_logp']=proposal_logp
        student.pending_state=memory
        return env_action

    def commit(self,action,reward,terminal):
        if self.pending is None or self.pending['action']!=jsonable(action):raise ValueError('navigation dispatch mismatch')
        row=self.pending;after=deepcopy(self.environment.gui)
        interrupted=bool(terminal or self.is_stopped())
        nav=None if interrupted else self.state.advance(row['server_before'],after)
        row.update(server_after=after,root_reward=reward,root_terminal=terminal,navigation_reward=nav,interrupted=interrupted)
        if self.state.arrival.required_ticks>1:row['arrival_hold_after']=self.state.arrival.count
        self.rows.append(row);self.pending=None
        self.student.commit(action,reward,terminal)
        self.student.rows[-1].update(navigation_reward=nav,interrupted=interrupted)
        self.previous=row['server_before']
        if interrupted:self.ended=True

    def close(self):self.ended=True;self.switch.goal=None


def run_navigation_trial(policy,env,output,*,world,gate,stop_file,**kwargs):
    stopped=lambda:gate.requested.is_set() or bool(stop_file and stop_file.exists())
    wrapper=RLEnvironment(env,policy,strict_ticks=True,reward_mode='wood-potential-v4',is_stopped=stopped)
    policy.environment,policy.is_stopped=wrapper,stopped
    def closed(report):write_json(output/'rl-control-finished.json',dict(control_closed=True,release=report['release'],monotonic_s=time.monotonic()))
    try:
        report=run_episode(policy,wrapper,output,goal='finish bounded navigation trial schedule',target_items=(),
            completion_check=lambda obs:policy.state.finished or (world.get('kind')=='natural_navigation' and wrapper.reward.completed),
            gate=gate,stop_file=stop_file,on_control_closed=closed,**kwargs)
        eligible=bool(report['execution_valid'] and policy.state.finished and not policy.pending and
            not any(r['interrupted'] for r in policy.rows))
        report.update(controller='explicit_navigation_'+policy.mode,rl_training_eligible=False,rl_updates=0,
            navigation_training_eligible=eligible and policy.config['split']=='train' and world.get('kind')!='natural_navigation',initial_world=world,
            navigation_receipts=policy.state.receipts,navigation_finished=policy.state.finished,
            learned_arrivals=sum(r['student_success'] for r in policy.state.receipts),
            teacher_arrivals=sum(r.get('owner')=='teacher' and r['outcome']=='completed' for r in policy.state.receipts),
            root_inventory_success=bool(wrapper.reward and wrapper.reward.completed),actual_llm_calls=0,
            actor_sha256=policy.actor_hash,mode_steps={o:sum(r['owner']==o for r in policy.rows) for o in ('student','neutral','teacher')},
            scope='privileged coordinates; known open local fixture; schedule completion is not learned success',
            root_actor_preserved=state_fingerprint(policy.switch.root.state_dict())==policy.root_fingerprint)
        report['arrival_hold_ticks']=policy.state.arrival.required_ticks
        if wrapper.startup_control:
            report['startup_control']=dict(wrapper.startup_evidence)
        report['pose_source']=policy.pose_source
        report['navigation_root_context_mode']=policy.saved.get('root_context_mode','observed')
        if policy.action_selection!='sample':
            report.update(action_selection=policy.action_selection,navigation_ppo_eligible=False,
                behavior_likelihood='point_mass',proposal_likelihood='delivered_command')
        report['trained_coordinate_source']=policy.saved.get('coordinate_source','privileged_server')
        if policy.state.arrival.required_ticks>1:
            report['scope']='privileged flat coordinates; consecutive settled arrival, not natural-terrain safety'
        if policy.pose_source=='client':
            report['scope']='privileged current client control pose; authoritative server completion plus current client settled check'
        if world.get('kind')=='step_navigation':
            observations=[wrapper.initial['server']]+[r['server_after'] for r in policy.rows]
            health=[g['server_health'] for g in observations]
            from .step_navigation import terrain_action_count
            mutation_count=terrain_action_count(policy.rows)
            qualified=bool(eligible and len(policy.state.receipts)==6 and
                all(r['outcome']=='completed' for r in policy.state.receipts) and
                min(health)==health[0]==20 and observations[-1]['server_alive']==1 and mutation_count==0)
            report['step_navigation']=dict(demonstration_qualified=qualified and policy.mode=='correction',
                initial_health=health[0],final_health=health[-1],minimum_health=min(health),
                observed_health_loss=sum(max(0.,a-b) for a,b in zip(health,health[1:])),
                terrain_action_count=mutation_count,
                completed_goals=sum(r['outcome']=='completed' for r in policy.state.receipts),
                learned_arrivals=report['learned_arrivals'],teacher_arrivals=report['teacher_arrivals'])
            report['navigation_training_eligible']=qualified and policy.mode=='correction' and policy.config['split']=='train'
            eligible=report['navigation_training_eligible']
            report['scope']=('Procedural step arena; privileged correction teacher; teacher arrivals are not learned-policy successes'
                if policy.mode=='correction' else 'Procedural step arena; learned-only evaluation; no correction teacher')
        if world.get('kind')=='natural_navigation':
            from .natural_navigation import attempted_goal_outcomes
            report['natural_navigation']=attempted_goal_outcomes(report,policy.rows,len(policy.state.targets))
            observations=[wrapper.initial['server']]+[r['server_after'] for r in policy.rows]
            health=[g['server_health'] for g in observations]
            report['natural_navigation'].update(initial_health=health[0],final_health=health[-1],
                minimum_health=min(health),observed_health_loss=sum(max(0.,a-b) for a,b in zip(health,health[1:])),
                final_food=observations[-1]['server_food'],final_alive=observations[-1]['server_alive'])
            report['natural_navigation']['termination_cause']=('goal_schedule_finished' if policy.state.finished else
                'root_reward_accepted' if wrapper.reward.completed else report['status'])
            report['scope']='Natural saved terrain; privileged XYZ targets; no route, teacher or training; unattempted goals excluded'
        write_json(output/'navigation-rows.json',policy.rows);write_json(output/'initial-state.json',getattr(wrapper,'initial',{}))
        if hasattr(policy.student,'initial_state'):
            policy.student.torch.save(dict(schema='navigation_trials_v1',split=policy.config['split'],eligible=eligible,
                rows=policy.student.rows,actor_sha256=policy.actor_hash,config=policy.config,world=world,
                initial_recurrent_state=policy.student.initial_state,receipts=policy.state.receipts),output/'navigation-trajectory.pt')
        write_json(output/'summary.json',report);return report
    finally:policy.student.dispose()
