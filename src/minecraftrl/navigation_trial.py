"""Explicit navigation demonstrations or learned-only trials under the normal supervisor."""
from copy import deepcopy
import json,time,math
from pathlib import Path
from .navigation_refinement import TrialState,motion_goal,local_targets
from .runtime import run_episode,jsonable,write_json
from .rl_policy import cpu_tree,guided,state_fingerprint
from .rl_collect import RLEnvironment
from .world_store import digest


class NavigationTrial:
    def __init__(self,student,raw,world,output,config,root):
        from .navigation_model import load_navigation,NavigationSwitch,value_head
        self.student,self.raw,self.world,self.output=student,raw,world,output
        self.config=config;self.mode=config['mode'];self.rows=[];self.pending=None;self.previous=None
        self.is_stopped=lambda:False;self.ended=False;self.environment=None
        expected='train' if world['experimental_split']=='train' else 'development'
        if config['split']!=expected or (self.mode=='ppo' and expected!='train'):raise ValueError('navigation split mismatch')
        self.state=TrialState(local_targets(world,config['points']),self.mode,student_ticks=config.get('student_ticks',32))
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
        goal=motion_goal(g,self.state.target,self.previous,remaining=100-self.state.count)
        self.switch.goal=goal
        # All owners encode their actual pre-action screen, so teacher frames retain real student history.
        features,memory=student.features(screen)
        with torch.no_grad():
            pd=guided(self.switch,features)
            if any(not torch.isfinite(v).all() for v in pd.values()):raise ValueError('nonfinite navigation distribution')
            if self.state.owner=='student':action=self.switch.sample(pd,deterministic=False)
            else:
                env_action=self.raw.action_space.noop()
                if self.state.owner=='teacher':
                    changes,_=navigate(g,self.state.target[0],self.state.target[2]);env_action.update(changes)
                factored=student.agent.action_transformer.env2policy({k:np.asarray(v) for k,v in env_action.items()})
                mapped=student.agent.action_mapper.from_factored({k:v[None] for k,v in factored.items()})
                action={k:torch.as_tensor(v,device=student.device).unsqueeze(1) for k,v in mapped.items()}
            value=self.value(append_motion(features[0:1],goal)).item()
            logp=self.switch.logprob(action,pd).item()
            if not math.isfinite(value) or not math.isfinite(logp):raise ValueError('nonfinite navigation value/logp')
        env_action=student.agent._agent_action_to_env({k:v[:,0] for k,v in action.items()})
        self.controller_name='navigation_'+self.state.owner
        self.pending=dict(owner=self.state.owner,index=self.state.index,target=list(self.state.target),goal=goal,
            server_before=g,action=jsonable(env_action),frame=f'frames/{len(self.rows):06d}.png')
        student.pending=dict(features=features.cpu(),action=cpu_tree(action),old_logp=logp,value=value,
            first=student.first,env_action=jsonable(env_action),motion_goal=goal,owner=self.state.owner,index=self.state.index)
        student.pending_state=memory
        return env_action

    def commit(self,action,reward,terminal):
        if self.pending is None or self.pending['action']!=jsonable(action):raise ValueError('navigation dispatch mismatch')
        row=self.pending;after=deepcopy(self.environment.gui)
        interrupted=bool(terminal or self.is_stopped())
        nav=None if interrupted else self.state.advance(row['server_before'],after)
        row.update(server_after=after,root_reward=reward,root_terminal=terminal,navigation_reward=nav,interrupted=interrupted)
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
            completion_check=lambda obs:policy.state.finished,gate=gate,stop_file=stop_file,on_control_closed=closed,**kwargs)
        eligible=bool(report['execution_valid'] and policy.state.finished and not policy.pending and
            not any(r['interrupted'] for r in policy.rows))
        report.update(controller='explicit_navigation_'+policy.mode,rl_training_eligible=False,rl_updates=0,
            navigation_training_eligible=eligible and policy.config['split']=='train',initial_world=world,
            navigation_receipts=policy.state.receipts,navigation_finished=policy.state.finished,
            learned_arrivals=sum(r['student_success'] for r in policy.state.receipts),
            teacher_arrivals=sum(r.get('owner')=='teacher' and r['outcome']=='completed' for r in policy.state.receipts),
            root_inventory_success=bool(wrapper.reward and wrapper.reward.completed),actual_llm_calls=0,
            actor_sha256=policy.actor_hash,mode_steps={o:sum(r['owner']==o for r in policy.rows) for o in ('student','neutral','teacher')},
            scope='privileged coordinates; known open local fixture; schedule completion is not learned success',
            root_actor_preserved=state_fingerprint(policy.switch.root.state_dict())==policy.root_fingerprint)
        write_json(output/'navigation-rows.json',policy.rows);write_json(output/'initial-state.json',getattr(wrapper,'initial',{}))
        if hasattr(policy.student,'initial_state'):
            policy.student.torch.save(dict(schema='navigation_trials_v1',split=policy.config['split'],eligible=eligible,
                rows=policy.student.rows,actor_sha256=policy.actor_hash,config=policy.config,world=world,
                initial_recurrent_state=policy.student.initial_state,receipts=policy.state.receipts),output/'navigation-trajectory.pt')
        write_json(output/'summary.json',report);return report
    finally:policy.student.dispose()
