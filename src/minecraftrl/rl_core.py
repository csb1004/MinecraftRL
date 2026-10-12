"""Reward and return contracts independent of the game and tensor runtime."""
import math
LOGS=tuple(k+'_log' for k in ('oak','birch','spruce','jungle','acacia','dark_oak'))

def tick_delta(before,after):
    values=[x.get('server_tick') for x in (before,after)]
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or int(v)!=v for v in values):
        raise ValueError('server tick unavailable')
    delta=int(values[1]-values[0])
    if delta!=1:raise ValueError('expected one server tick, observed '+str(delta))
    return delta

def episode_returns(episodes):
    advantages=[];returns=[]
    for data in episodes:
        a,r=gae([x['reward'] for x in data['rows']],[x['value'] for x in data['rows']],
                data['bootstrap'],terminal=data['terminal'])
        advantages.extend(a);returns.extend(r)
    return advantages,returns

def validate_training_data(data):
    if any(k in data.get('initial_heads',{}) for k in ('backbone_tail','backbone_temporal')):
        raise ValueError('head-only PPO cannot resume a finetuned backbone')
    if not data['eligible'] or data['split']!='train' or data['scope']!='head_only_frozen_features' or not data['rows']:
        raise ValueError('trajectory cannot train')
    if data.get('status')=='time_limit' and data.get('condition',{}).get('recurrent_commit_mode')!='dispatch_commit':
        raise ValueError('time-limit bootstrap requires committed recurrent memory')
    for row in data['rows']:
        if any(not math.isfinite(row[key]) for key in ('old_logp','value','reward')):
            raise ValueError('nonfinite trajectory')
    config=data.get('reward_config',{'name':'sparse','gamma':.99})
    if config.get('name')!='sparse':
        reward_class=SHAPED_REWARDS.get(config.get('name'))
        if reward_class is None or config!=reward_class.metadata or data.get('condition',{}).get('reward_mode')!=config['name']:
            raise ValueError('mismatched shaping/return contract')
        for row in data['rows']:
            part=row.get('reward_components',{})
            keys=('task','shaping','time_cost','total','potential_before','potential_after')
            if any(k not in part or not math.isfinite(part[k]) for k in keys):raise ValueError('missing reward audit')
            failure=part.get('failure',0.)
            expected_failure=-config.get('terminal_failure_cost',0.) if row['terminal'] and not part['rm_accepting'] else 0.
            if failure!=expected_failure:raise ValueError('invalid terminal failure cost')
            if part.get('ticks')!=1 or not math.isclose(row['reward'],part['task']+part['shaping']+part['time_cost']+failure,abs_tol=1e-10):
                raise ValueError('reward components mismatch')
            expected=ShapedWoodReward.alpha*(ShapedWoodReward.gamma*part['potential_after']-part['potential_before'])
            if not math.isclose(expected,part['shaping'],abs_tol=1e-10) or part['time_cost']!=-ShapedWoodReward.tick_cost:
                raise ValueError('invalid shaping calculation')

def validate_collector_result(summary):
    """Final supervisor verdict takes precedence over an earlier tensor save."""
    if not summary['execution_valid'] or summary.get('worker_exit_code')!=0:
        raise ValueError('collector did not finish cleanly; trajectory cannot train')

class WoodReward:
    def __init__(self,inventory):
        if self.count(inventory):raise ValueError('wood episode must start without logs')
        self.completed=False
    @staticmethod
    def count(inventory):
        if not isinstance(inventory,dict):raise ValueError('server inventory required')
        values=[inventory.get(k,0) for k in LOGS]
        if any(not isinstance(n,(int,float)) or not math.isfinite(n) or n<0 or int(n)!=n for n in values):
            raise ValueError('invalid server inventory')
        return sum(values)
    def observe(self,inventory):
        count=self.count(inventory)
        if not self.completed and count>=1:self.completed=True;return 1.
        return 0.


class ShapedWoodReward:
    """Opt-in first-log RM. Server ray telemetry is privileged training context.

    True terminals use zero potential; time-limit truncations retain potential
    and require the value bootstrap at the same final RM state.
    """
    gamma=.99
    alpha=.2
    tick_cost=.0001
    stages=('search','approach','mine','collect','complete')
    metadata=dict(name='wood-potential-v1',gamma=gamma,alpha=alpha,tick_cost=tick_cost,
                  distance_cap=16.,sensor='server_center_ray_16_blocks_locked_target',
                  collection_distance='last_broken_block_center_not_item_entity',
                  context='stage_one_hot_and_capped_distance')

    def __init__(self,observation):
        self.task=WoodReward(observation.get('server_inventory'))
        self.completed=False;self.ended=False;self.last=observation
        self.stage,self.distance,self.potential=self.state(observation)

    @staticmethod
    def state(obs):
        keys=('server_tick','wood_kind','wood_distance','wood_reachable')
        if any(k not in obs or not isinstance(obs[k],(int,float)) or not math.isfinite(obs[k]) for k in keys):
            raise ValueError('wood telemetry unavailable')
        kind=obs['wood_kind'];distance=obs['wood_distance'];reachable=obs['wood_reachable']
        if kind not in (0,1,2) or distance<0 or reachable not in (0,1) or obs['server_tick']<0 or int(obs['server_tick'])!=obs['server_tick']:
            raise ValueError('invalid wood telemetry')
        distance=min(distance,16.)
        if kind==0:return 'search',0.,0.
        if kind==2:return 'collect',distance,.6+.3*(1-distance/16.)
        if reachable:return 'mine',distance,.5
        return 'approach',distance,.1+.3*(1-distance/16.)

    def context(self):
        return [float(self.stage==s) for s in self.stages]+[self.distance/16.]

    def advance(self,observation,*,terminal=False):
        if self.ended:raise RuntimeError('action after terminal reward state')
        ticks=tick_delta(self.last,observation)
        stage,distance,potential=self.state(observation)
        task=self.task.observe(observation['server_inventory'])
        self.completed=self.task.completed;self.ended=bool(terminal or self.completed)
        if self.completed:stage='complete'
        if self.ended:potential=0.
        shaping=self.alpha*(self.gamma**ticks*potential-self.potential)
        cost=-self.tick_cost*ticks
        result=dict(task=task,shaping=shaping,time_cost=cost,total=task+shaping+cost,
                    stage_before=self.stage,stage=stage,potential_before=self.potential,
                    potential_after=potential,distance=distance,ticks=ticks,
                    rm_accepting=self.completed,human_prior=0.)
        self.stage=stage;self.distance=distance;self.potential=potential;self.last=observation
        return result


class ShapedWoodRewardV2(ShapedWoodReward):
    """Failure terminal cost removes the incentive to end early to save ticks.

    User cancellation/errors are ineligible data; wall-clock truncation is not
    terminal failure and continues to use bootstrap.
    """
    metadata=dict(ShapedWoodReward.metadata,name='wood-potential-v2',terminal_failure_cost=1.)

    def advance(self,observation,*,terminal=False):
        result=super().advance(observation,terminal=terminal)
        result['failure']=-1. if terminal and not self.completed else 0.
        result['total']+=result['failure']
        return result


class ShapedWoodRewardV3(ShapedWoodRewardV2):
    """Same objective, versioned target sensor following accepted server mining.

    'mine' includes readiness or an ongoing accepted server interaction; it is
    not proof that the block broke. Collection still uses the broken position.
    """
    metadata=dict(ShapedWoodRewardV2.metadata,name='wood-potential-v3',
                  sensor='server_locked_ray_target_with_accepted_mining_retarget_v1')

    @staticmethod
    def state(obs):
        translated={'server_tick':obs.get('server_tick')}
        for key in ('kind','distance','reachable'):
            translated['wood_'+key]=obs.get('wood_v3_'+key)
        return ShapedWoodReward.state(translated)


class ShapedWoodRewardV4(ShapedWoodRewardV3):
    """Opt-in stall feedback using the existing search context; no action override.

    This is a recovery candidate detector, not a proof of unreachable terrain.
    Old actor weights are transferable, not evidence of a learned recovery skill.
    """
    metadata=dict(ShapedWoodRewardV3.metadata,name='wood-potential-v4',
                  recovery=dict(window_ticks=200,axis_span_max=2.,progress_min=.5,exit_progress=1.,
                                feedback='search_context_until_progress_new_target_or_interaction'))

    def __init__(self,observation):
        from .approach_recovery import ApproachRecovery
        super().__init__(observation)
        self.recovery=ApproachRecovery()
        self.recovery.update(observation,self.stage)

    def advance(self,observation,*,terminal=False):
        if self.ended:raise RuntimeError('action after terminal reward state')
        tick_delta(self.last,observation)
        raw_stage,_,_=self.state(observation)
        accepting=self.task.count(observation['server_inventory'])>=1
        recovery=self.recovery.update(observation,raw_stage,terminal=terminal or accepting)
        selected=observation
        if recovery['active']:
            selected=dict(observation,wood_v3_kind=0,wood_v3_distance=0.,wood_v3_reachable=0)
        result=super().advance(selected,terminal=terminal)
        result['recovery']=recovery
        return result


SHAPED_REWARDS={'wood-potential-v1':ShapedWoodReward,'wood-potential-v2':ShapedWoodRewardV2,
                'wood-potential-v3':ShapedWoodRewardV3,'wood-potential-v4':ShapedWoodRewardV4}

def gae(rewards,values,bootstrap,*,terminal,gamma=.99,lam=.95):
    if not rewards or len(rewards)!=len(values):raise ValueError('inconsistent trajectory')
    advantage=[0.]*len(rewards);carry=0.;following=0. if terminal else bootstrap
    for i in reversed(range(len(rewards))):
        delta=rewards[i]+gamma*following-values[i]
        carry=delta+gamma*lam*carry;advantage[i]=carry;following=values[i]
    return advantage,[a+v for a,v in zip(advantage,values)]
