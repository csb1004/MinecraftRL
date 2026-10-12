"""Real gameplay collector; cancellation/error trajectories cannot train."""
from pathlib import Path
import time
from .rl_core import WoodReward,SHAPED_REWARDS,LOGS,tick_delta
from .runtime import run_episode,write_json,inventory_counts
from .rl_policy import cpu_tree

class RLEnvironment:
    def __init__(self,env,policy,strict_ticks=False,is_stopped=lambda:False,reward_mode='sparse'):
        if reward_mode!='sparse' and reward_mode not in SHAPED_REWARDS:raise ValueError('unknown reward mode')
        self.reward_mode=reward_mode
        self.env=env;self.policy=policy;self.reward=None;self.last=None;self.gui=None;self.terminal=False;self.strict_ticks=strict_ticks;self.ticks=[];self.is_stopped=is_stopped
    def seed(self,seed):return self.env.seed(seed)
    def reset(self):
        self.env.reset()
        readiness=[];previous=None
        for _ in range(60 if self.strict_ticks else 1):
            if self.is_stopped():raise InterruptedError('STOP during neutral startup')
            obs,_,done,info=self.env.step(self.env.raw.action_space.noop())
            self.gui=info['workshop']
            if not self.gui['server_available'] or self.gui['error'] or done:raise ValueError('server baseline unavailable')
            if self.strict_ticks:
                if previous is not None:tick_delta(previous,self.gui)
                readiness.append(self.gui.get('render_ready',0));previous=self.gui
        if self.strict_ticks and getattr(self.policy,'output',None):
            from .runtime import save_png
            write_json(self.policy.output/'warmup.json',dict(render_readiness=readiness,last_server=self.gui))
            save_png(obs['pov'],self.policy.output/'warmup.png')
        # Global chunk queues include work outside this image and are not a
        # visible-frame completion barrier. The batch compares actual pixels.
        self.reward=SHAPED_REWARDS[self.reward_mode](self.gui) if self.reward_mode!='sparse' else WoodReward(self.gui['server_inventory'])
        if self.reward_mode!='sparse':self.policy.set_reward_context(self.reward.context())
        self.last=obs
        self.initial=dict(observation={k:v for k,v in obs.items() if k!='pov'},server=self.gui)
        return obs
    def step(self,action):
        if self.terminal or self.reward.completed:raise RuntimeError('action after accepting state')
        previous=self.gui
        obs,_,done,info=self.env.step(action);self.gui=info['workshop'];self.last=obs
        if not self.gui['server_available'] or self.gui['error']:raise ValueError('server reward unavailable')
        if self.strict_ticks:self.ticks.append(tick_delta(previous,self.gui))
        components=None
        if self.reward_mode!='sparse':
            components=self.reward.advance(self.gui,terminal=done);reward=components['total']
        else:reward=self.reward.observe(self.gui['server_inventory'])
        self.terminal=bool(done or self.reward.completed)
        self.policy.commit(action,reward,self.terminal)
        if components is not None:
            self.policy.rows[-1]['reward_components']=components
            self.policy.set_reward_context(self.reward.context())
            return obs,reward,done,dict(info,rl_reward=components)
        return obs,reward,done,dict(info,rl_reward=dict(skill=reward,rm_accepting=self.reward.completed,human_prior=0.))
    def release(self):
        status=self.env.release()
        if status=='neutral_command_sent':
            evidence=getattr(self.env,'release_evidence',{'verified':False})
            if getattr(self.policy,'output',None):write_json(Path(self.policy.output)/'input-release-state.json',evidence)
            if not evidence.get('verified'):raise RuntimeError('neutral release server keys not verified')
        return status
    def close(self):return self.env.close()

def run_rl(policy,env,output,*,split,world=None,reward_mode='sparse',**kwargs):
    def stopped():
        return kwargs['gate'].requested.is_set() or bool(kwargs.get('stop_file') and Path(kwargs['stop_file']).exists())
    wrapper=RLEnvironment(env,policy,strict_ticks=True,is_stopped=stopped,reward_mode=reward_mode)
    report=run_episode(policy,wrapper,output,goal='collect wood',target_items=LOGS,
                       completion_check=lambda obs:wrapper.reward.completed,**kwargs)
    # run_episode has latched the gate, released input and closed the game.
    # Bootstrap, tensor serialization and interpreter teardown are not control.
    if report['execution_valid'] and report['release']=='neutral_command_sent':
        write_json(Path(output)/'rl-control-finished.json',dict(
            control_closed=True,release=report['release'],monotonic_s=time.monotonic()))
    torch=policy.torch
    try:
        if not hasattr(policy,'initial_state'):
            report.update(rl_split=split,rl_training_eligible=False,trajectory_steps=0,rl_updates=0)
            write_json(output/'summary.json',report)
            return report
        eligible=report['execution_valid'] and report['status'] in ('target_reached','time_limit','environment_done') and bool(policy.rows)
        bootstrap=0. if wrapper.terminal or not eligible else policy.bootstrap(wrapper.last['pov'])
        dataset=dict(schema=1,split=split,scope='head_only_frozen_features',rows=policy.rows,bootstrap=bootstrap,
                     terminal=wrapper.terminal,eligible=eligible,status=report['status'],
                     initial_heads=policy.initial_heads,initial_recurrent_state=policy.initial_state,
                     final_recurrent_state=cpu_tree(policy.agent.hidden_state),condition=policy.condition_metadata)
        dataset['seed']=kwargs['seed'];dataset['initial_world']=world
        dataset['reward_config']=dict(wrapper.reward.metadata) if reward_mode!='sparse' else dict(name='sparse',gamma=.99)
        if reward_mode!='sparse':dataset['final_reward_context']=wrapper.reward.context()
        write_json(output/'initial-state.json',dict(getattr(wrapper,'initial',{}),initial_world=world,seed=kwargs['seed']))
        torch.save(dataset,output/'trajectory.pt')
        report.update(controller='steve1_head_rl',rl_split=split,rl_training_eligible=eligible and split=='train',
                      server_target_reached=bool(wrapper.reward and wrapper.reward.completed),
                      reward_sum=sum(row['reward'] for row in policy.rows),trajectory_steps=len(policy.rows),
                      rl_updates=0,condition=policy.condition_metadata)
        report['reward_config']=dataset['reward_config']
        if reward_mode!='sparse':
            report['reward_component_sums']={k:sum(r['reward_components'][k] for r in policy.rows) for k in ('task','shaping','time_cost')}
            if 'terminal_failure_cost' in dataset['reward_config']:report['reward_component_sums']['failure']=sum(r['reward_components']['failure'] for r in policy.rows)
        report.update(server_tick_deltas=wrapper.ticks,simulation_ticks=sum(wrapper.ticks),
                      neutral_startup_steps=60,render_ready_at_start=getattr(wrapper,'initial',{}).get('server',{}).get('render_ready'),
                      nominal_simulation_seconds=sum(wrapper.ticks)/20.,initial_world=world)
        write_json(output/'summary.json',report)
    finally:policy.dispose()
    return report
