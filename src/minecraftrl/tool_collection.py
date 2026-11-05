"""Neural tool-use feasibility with server-verified, persistent collection goals.

The privileged monitor only adjudicates/halts; it never chooses motor actions.
The GUI workshop is a separate, explicitly assisted stage.
"""
import copy
import math
import time

from .runtime import run_episode,write_json
from .world_store import validate_restored_observation,observation_evidence
from .startup_pickups import saved_drop_budget

LOGS=('oak_log','birch_log','spruce_log','jungle_log','acacia_log','dark_oak_log')
HEALTH_PROFILES={
    'standard':dict(minimum_health=4,maximum_damage=2,minimum_start_food=1,require_no_pursuit=False),
    'fed_low_health':dict(minimum_health=2,maximum_damage=1,minimum_start_food=4,require_no_pursuit=True),
}
PHASES={
    'cobblestone':dict(tool='wooden_pickaxe',items=('cobblestone',),count=3,counter='server_pick_stone',prompt='dig'),
    'axe_wood':dict(tool='stone_axe',items=LOGS,count=64,counter='server_axe_logs',prompt='collect wood'),
    'workstation_access':dict(tool='wooden_pickaxe',items=(),count=1,counter=None,prompt='return to a usable surface workstation'),
    'recover_axe':dict(tool='stone_axe',items=('stone_axe',),count=1,counter=None,prompt='recover the dropped stone axe'),
    'evade_hostile':dict(tool='stone_axe',items=(),count=1,counter=None,prompt='stabilize after hostile pursuit'),
}


def health_limits_for_phase(phase,profile):
    if profile not in HEALTH_PROFILES:raise ValueError('unknown tool health profile')
    limits=dict(HEALTH_PROFILES[profile])
    if phase=='evade_hostile':
        if profile!='standard':raise ValueError('low-health collection profile cannot authorize hostile evasion')
        # Match the existing session recovery entry/stop floor. A post-hit guard
        # cannot prevent a single lethal hit or fall; this is not a safety proof.
        limits['minimum_health']=8
    return limits


def validate_condition_phase(phase,path):
    if phase not in PHASES:raise ValueError('unknown tool phase')
    if phase in ('workstation_access','recover_axe','evade_hostile') and path is None:
        raise ValueError('workstation recovery requires an explicit local condition')
    if path is not None and phase not in ('workstation_access','axe_wood','recover_axe','evade_hostile'):
        raise ValueError('custom condition is unsupported for this tool phase')


def observed_count(inventory,items):return sum(int(inventory.get(item,0)) for item in items)


def make_ledger(bundle,phase):
    spec=PHASES[phase];inventory=bundle['observation']['inventory']
    ledger=copy.deepcopy(bundle.get('controller_state',{}).get('tool_mission',{}))
    if ledger and ledger.get('schema')!=1:raise ValueError('unsupported tool mission ledger')
    if not ledger:
        ledger=dict(schema=1,world_id=bundle['world_id'],objective='stone axe then additional 64 logs',stages={})
    if ledger['world_id']!=bundle['world_id']:raise ValueError('tool mission world mismatch')
    if phase=='axe_wood' and ledger['stages'].get(phase,{}).get('last_halt_reason')=='tool_collection_pursuit_guard':
        raise ValueError('recorded pursuit requires recovery before collection')
    recovery_evidence=None
    if phase=='recover_axe':
        parent=ledger['stages'].get('axe_wood',{})
        if (parent.get('status')!='active' or parent.get('last_halt_reason')!='required_tool_lost'
                or inventory.get(spec['tool'],0)!=0):
            raise ValueError('tool recovery requires a recorded loss and an absent tool')
        recovery_evidence=saved_drop_budget(bundle['archive'],inventory,include_falling=True)
        if recovery_evidence['inventory_budget'].get(spec['tool'],0)<1:
            raise ValueError('no recent nearby saved tool supports recovery')
        previous=ledger['stages'].get(phase)
        if previous and previous['status']=='completed':
            ledger.setdefault('recovery_history',[]).append(copy.deepcopy(previous))
            del ledger['stages'][phase]
    elif inventory.get(spec['tool'],0)<1:raise ValueError('required crafted tool missing: '+spec['tool'])
    if phase=='evade_hostile':
        health=bundle['observation'].get('life_stats',{}).get('life')
        if type(health) not in (int,float) or not math.isfinite(health) or health<=8:
            raise ValueError('hostile recovery requires saved health above8')
        parent=ledger['stages'].get('axe_wood',{})
        if parent.get('status')!='active' or parent.get('last_halt_reason')!='tool_collection_pursuit_guard':
            raise ValueError('hazard recovery requires recorded pursuit during collection')
        previous=ledger['stages'].get(phase)
        if previous and previous['status']=='completed':
            ledger.setdefault('hazard_recovery_history',[]).append(copy.deepcopy(previous))
            del ledger['stages'][phase]
        elif previous and previous.get('episodes',0)>=1:
            raise ValueError('hazard recovery episode budget exhausted')
    if phase not in ledger['stages']:
        if phase in ('axe_wood','workstation_access') and ledger['stages'].get('cobblestone',{}).get('status')!='completed':
            raise ValueError('verified cobblestone prerequisite missing')
        ledger['stages'][phase]=dict(status='active',baseline=observed_count(inventory,spec['items']),
            target=spec['count'],verified_tool_harvests=0,episodes=0,source_generation=bundle['generation'])
        if phase=='workstation_access':
            tables=[m['position'] for m in bundle.get('controller_state',{}).get('last_workshop',{}).get('milestones',[])
                    if m['event']=='workstation_opened']
            if not tables:raise ValueError('observed surface workstation prerequisite missing')
            ledger['stages'][phase]['minimum_surface_y']=tables[-1][1]-1
    stage=ledger['stages'][phase]
    if stage['target']!=spec['count'] or stage['status']=='completed':raise ValueError('tool phase already completed or contract changed')
    if recovery_evidence is not None:stage['entry_evidence']=recovery_evidence
    ledger['active_phase']=phase
    return ledger


def criteria(stage,inventory,spec):
    return (observed_count(inventory,spec['items'])-stage['baseline']>=stage['target']
            and stage['verified_tool_harvests']>=stage['target'])


def workstation_ready(gui,minimum_y):
    if (gui.get('server_available')!=1 or gui.get('error') or gui.get('server_alive')!=1
            or gui.get('screen')!=0 or gui.get('server_on_ground')!=1 or gui.get('server_fall_distance')!=0
            or gui.get('server_y',-1e9)<minimum_y or gui.get('pursuit_valid')!=1 or gui.get('pursuit_count')!=0
            or gui.get('server_inventory',{}).get('cobblestone',0)<3):return False
    table=(gui.get('table_count',0)>0 and sum((gui['table_'+a]+.5-gui['eye_'+a])**2 for a in 'xyz')<=3.7**2)
    return bool(table or gui.get('candidate')==1)


class ToolCollectionEnvironment:
    def __init__(self,env,bundle,store,phase,gate,health_profile='standard'):
        self.health_profile=health_profile;self.health_limits=health_limits_for_phase(phase,health_profile)
        self.env=env;self.bundle=bundle;self.store=store;self.phase=phase;self.gate=gate
        self.spec=PHASES[phase];self.ledger=make_ledger(bundle,phase)
        self.stage=self.ledger['stages'][phase];self.prior_harvests=self.stage['verified_tool_harvests']
        self.counter=0;self.gui=None;self.restore=None;self.saved=None;self.completed=False;self.steps=0
        self.equipped_steps=0;self.minimum_health=None;self.minimum_food=None;self.halt_reason=None
        self.policy_memory=None;self.output=None
        self.final_tool_evidence=None;self.completion_source=None
        self.stability=None;self.axe_counter=0
        if phase=='evade_hostile':
            from .threat_observation import PursuitStability
            self.stability=PursuitStability()
            self.prior_axe_harvests=self.ledger['stages']['axe_wood']['verified_tool_harvests']
    def seed(self,seed):return self.env.seed(seed)
    def reset(self):
        obs=self.env.reset()
        if self.output is not None:
            write_json(self.output/'restore-observation.json',dict(expected=self.bundle['observation'],actual=observation_evidence(obs)))
        self.restore=validate_restored_observation(self.bundle,obs,allow_startup_pickups=True,allow_falling_startup_pickups=True)
        self.initial_health=float(obs['life_stats']['life']);self.minimum_health=self.initial_health
        if (not math.isfinite(self.initial_health) or self.initial_health<=0
                or obs['life_stats'].get('is_alive') is False):
            raise ValueError('insufficient or invalid starting health for tool experiment')
        self.minimum_food=float(obs['life_stats']['food'])
        if not math.isfinite(self.minimum_food) or not 0<=self.minimum_food<=20:
            raise ValueError('insufficient or invalid starting food for tool experiment')
        self.stage['retained_gain']=observed_count(obs['inventory'],self.spec['items'])-self.stage['baseline']
        collected=(self.phase in ('cobblestone','axe_wood') and criteria(self.stage,obs['inventory'],self.spec))
        recovered=(self.phase=='recover_axe' and obs['inventory'].get(self.spec['tool'],0)>=1)
        if collected or recovered:
            self.completed=True;self.stage['status']='completed';self.completion_source='restore_pickup'
        else:
            if self.initial_health<=self.health_limits['minimum_health']+1e-5:
                raise ValueError('insufficient starting health for tool experiment')
            if self.minimum_food<self.health_limits['minimum_start_food']:
                raise ValueError('insufficient starting food for tool experiment')
        return obs
    def step(self,action):
        if self.completed:raise RuntimeError('tool phase action after completion')
        obs,reward,done,info=self.env.step(action);self.steps+=1;g=info['workshop'];self.gui=g
        if g['server_available']!=1 or g['error']:raise ValueError('server tool evidence unavailable')
        self.update_counter(g)
        self.equipped_steps+=int(g['server_selected_item']==self.spec['tool'])
        health=float(g['server_health'])
        if not math.isfinite(health):raise ValueError('invalid observed health level')
        self.minimum_health=min(self.minimum_health,health)
        food=float(obs['life_stats']['food'])
        if not math.isfinite(food) or not 0<=food<=20:raise ValueError('invalid observed food level')
        self.minimum_food=food if self.minimum_food is None else min(self.minimum_food,food)
        self.stage['retained_gain']=observed_count(g['server_inventory'],self.spec['items'])-self.stage['baseline']
        if self.stability:self.stability.observe(g)
        self.completed=self.requirements(g)
        if self.stability and self.ledger['stages']['axe_wood']['status']=='completed':
            self.halt_reason='root_tool_goal_reached_during_recovery';self.gate.stop(self.halt_reason)
        if self.completed:self.stage['status']='completed';self.completion_source='control'
        # Client JSON rounds health; the server channel retains float32 precision.
        elif (health<=self.health_limits['minimum_health']+1e-5
                or self.initial_health-health>=self.health_limits['maximum_damage']-1e-5):
            self.halt_reason='tool_collection_health_guard';self.gate.stop(self.halt_reason)
        elif food==0:
            self.halt_reason='tool_collection_food_guard';self.gate.stop(self.halt_reason)
        elif self.phase!='evade_hostile' and self.health_limits['require_no_pursuit'] and (g.get('pursuit_valid')!=1 or g.get('pursuit_count')!=0):
            self.halt_reason='tool_collection_pursuit_guard';self.gate.stop(self.halt_reason)
        elif self.phase!='recover_axe' and g['server_inventory'].get(self.spec['tool'],0)<1:
            self.halt_reason='required_tool_lost';self.gate.stop(self.halt_reason)
        # Adjudication uses the server inventory; original client observation remains in env.
        authoritative=dict(obs,inventory=g['server_inventory'])
        return authoritative,reward,done,dict(info,tool_phase=self.phase,tool_phase_state=copy.deepcopy(self.stage))
    def update_counter(self,gui):
        if gui.get('server_available')!=1 or gui.get('error'):raise ValueError('server tool evidence unavailable')
        counter=gui[self.spec['counter']] if self.spec['counter'] else 0
        if not math.isfinite(counter) or counter!=int(counter) or counter<self.counter:
            raise ValueError('invalid or reset tool harvest counter')
        self.counter=int(counter);self.stage['verified_tool_harvests']=self.prior_harvests+self.counter
        if self.phase=='evade_hostile':
            count=gui['server_axe_logs']
            if not math.isfinite(count) or int(count)!=count or count<self.axe_counter:
                raise ValueError('invalid recovery axe harvest counter')
            self.axe_counter=int(count);parent=self.ledger['stages']['axe_wood']
            parent['verified_tool_harvests']=self.prior_axe_harvests+self.axe_counter
            parent['retained_gain']=observed_count(gui['server_inventory'],LOGS)-parent['baseline']
            complete=gui.get('server_alive')==1 and criteria(parent,gui['server_inventory'],PHASES['axe_wood'])
            parent['status']='completed' if complete else 'active'
    def requirements(self,gui):
        if self.phase=='evade_hostile':
            return (self.stability.ready(gui) and gui['server_inventory'].get('stone_axe',0)>=1
                    and gui.get('server_food',self.minimum_food)>0
                    and gui['server_health']>self.health_limits['minimum_health']+1e-5
                    and self.initial_health-gui['server_health']<self.health_limits['maximum_damage']-1e-5)
        if self.phase=='workstation_access':return workstation_ready(gui,self.stage['minimum_surface_y'])
        if self.phase=='recover_axe':
            return gui['server_alive']==1 and gui['server_inventory'].get(self.spec['tool'],0)>=1
        return criteria(self.stage,gui['server_inventory'],self.spec) and gui['server_alive']==1
    def release(self):
        status=self.env.release()
        if (self.output is not None and self.gate.requested.is_set() and status=='neutral_command_sent'
                and getattr(self.env,'release_evidence',{}).get('verified')):
            write_json(self.output/'tool-control-finished.json',dict(control_closed=True,release=status,
                monotonic_s=time.monotonic(),steps=self.steps,scope='saving only; policy gate already stopped'))
        return status
    def close(self):
        try:
            if self.restore:
                world=self.env.save_and_quit()
                state=copy.deepcopy(self.bundle.get('controller_state',{}))
                self.stage['episodes']+=1
                final_gui=self.env.last_info.get('workshop',{})
                inventory=final_gui.get('server_inventory')
                if inventory is None:raise ValueError('missing final server inventory')
                self.update_counter(final_gui)
                self.final_tool_evidence={k:copy.deepcopy(final_gui.get(k)) for k in (
                    'server_available','error','server_tick','held_key_count','server_alive','server_selected_item',
                    'server_inventory','server_pick_stone','server_axe_logs','server_log_harvests',
                    'server_health','server_recent_damage_type')}
                self.stage['retained_gain']=observed_count(inventory,self.spec['items'])-self.stage['baseline']
                completed=self.requirements(final_gui)
                if completed and not self.completed:self.completion_source='cleanup'
                if not completed:self.completion_source=None
                self.completed=completed;self.stage['status']='completed' if completed else 'active'
                self.stage['last_halt_reason']=self.halt_reason
                if self.stability:
                    self.stage['stability_evidence']=dict(quiet_observed_ticks=self.stability.quiet_ticks,
                        observed_pursuit_steps=self.stability.observed_pursuit_steps,
                        completion_rule='100_consecutive_observed_ticks_no_pursuit_or_damage_and_grounded',
                        scope='privileged local observation; not guaranteed future safety')
                    if completed:self.ledger['stages']['axe_wood']['last_halt_reason']=None
                state['tool_mission']=self.ledger
                observation=dict(self.env.last_observation,inventory=inventory)
                memory=None
                if self.policy_memory is not None and getattr(self.env,'release_evidence',{}).get('verified'):
                    memory=self.policy_memory.snapshot(self.output)
                destination=self.store.commit(world,copy.deepcopy(self.bundle['task_checkpoint']),observation,controller_state=state,policy_memory=memory)
                self.saved=dict(status='saved',generation=destination.name)
        finally:self.env.close()


class RememberedToolPolicy:
    def __init__(self,policy,memory,world_id,phase,output,condition=None):
        self.policy=policy;self.memory=memory;self.world_id=world_id;self.phase=phase;self.output=output
        self.condition=condition
    def reset(self,goal):
        if self.condition is None:self.policy.reset(goal)
        else:
            condition=self.condition;self.policy.agent.reset(cond_scale=condition['cond_scale'])
            self.policy.goal_embed=condition['embedding'].copy()
            self.policy.condition_metadata=dict(profile='experimental-tool-condition',requested_goal=goal,
                effective_prompt=condition['provenance']['prompt'],cond_scale=condition['cond_scale'],
                embedding_sha256=condition['sha256'],provenance=condition['provenance'])
        receipt=self.memory.on_reset(dict(task_id='stone-axe-mission-'+self.phase,revision=1,goal=goal),self.world_id)
        write_json(self.output/'policy-memory-restore.json',receipt)
    def act(self,screen):
        action=self.policy.act(screen);self.memory.inferred();return action
    def close(self):return self.policy.close()


def run_tool_collection(policy,env,output,*,bundle,store,phase,gate,condition_path=None,health_profile='standard',**kwargs):
    validate_condition_phase(phase,condition_path)
    wrapped=ToolCollectionEnvironment(env,bundle,store,phase,gate,health_profile=health_profile)
    condition=None
    if condition_path is not None:
        from .hazard_condition import load_condition
        kind='experimental_hazard_condition' if phase=='evade_hostile' else 'experimental_tool_condition'
        condition=load_condition(condition_path,policy.root,kind=kind)
        write_json(output/'tool-condition.json',condition['provenance'])
    from .cuda_startup import prime_convolution
    write_json(output/'backend-preparation-started.json',dict(phase='tool_collection_policy_initialization'))
    preparation=prime_convolution(policy.agent.policy.net.img_process.cnn.stacks[0].firstconv.layer)
    write_json(output/'backend-preparation.json',dict(enabled=True,**preparation))
    from .policy_memory import PolicyMemory,memory_identity,initialized_template
    memory=PolicyMemory(policy,memory_identity(policy,policy.root),initialized_template(policy),bundle)
    wrapped.policy_memory=memory;wrapped.output=output
    remembered=RememberedToolPolicy(policy,memory,bundle['world_id'],phase,output,condition)
    report=run_episode(remembered,wrapped,output,goal=wrapped.spec['prompt'],target_items=wrapped.spec['items'],
        target_count=wrapped.spec['count'],completion_check=lambda obs:wrapped.completed,
        check_completion_on_reset=phase not in ('workstation_access','evade_hostile'),gate=gate,**kwargs)
    report.update(tool_phase=phase,tool_mission=wrapped.ledger,world_restore=wrapped.restore,world_save=wrapped.saved,
        tool_health_profile=wrapped.health_profile,tool_health_limits=wrapped.health_limits,
        tool_phase_success=bool(wrapped.completed and wrapped.saved and report['execution_valid']),
        equipped_steps=wrapped.equipped_steps,minimum_server_health=wrapped.minimum_health,minimum_observed_food=wrapped.minimum_food,
        halt_reason=wrapped.halt_reason,release_evidence=getattr(env,'release_evidence',None),
        final_tool_evidence=wrapped.final_tool_evidence,completion_source=wrapped.completion_source,
        condition_metadata=policy.condition_metadata,
        controller_scope='Unmodified STEVE-1 motor actions; privileged server tool-at-harvest adjudication and health guard',
        policy_memory=copy.deepcopy(memory.receipt),api_calls=0)
    write_json(output/'summary.json',report)
    return report
