"""Isolated fixed-recipe probes; not supported dialogue tasks."""
from .runtime import inventory_counts, jsonable, run_episode, save_png, write_json
from .world_store import observation_evidence, validate_restored_observation
from copy import deepcopy
import math

GOAL='craft wooden planks'
RECIPES = {
    'planks': dict(goal=GOAL, ingredient='oak_log', initial=1, consumed=1, target='oak_planks', count=4),
    'sticks': dict(goal='craft sticks', ingredient='oak_planks', initial=4, consumed=2, target='stick', count=4),
    'table': dict(goal='craft a crafting table', ingredient='oak_planks', initial=4, consumed=4, target='crafting_table', count=1),
}


def _scalar(value):
    value=jsonable(value)
    while isinstance(value,list) and len(value)==1: value=value[0]
    if not isinstance(value,(int,float)) or not math.isfinite(value): raise ValueError('invalid craft action scalar')
    return value


def guard_result_slot(action,gui):
    if 'sneak' not in action: raise ValueError('sneak action required')
    forwarded=deepcopy(action)
    audit=dict(reason='world',predicted_mouse=None,geometry=jsonable(gui))
    if _scalar(action.get('inventory',0)):
        audit['reason']='gui_transition';forwarded['sneak']=0
    elif gui is None:
        audit['reason']='bootstrap';forwarded['sneak']=0
    elif gui.get('other_screen_open'):
        raise ValueError('unsupported GUI for result-slot guard')
    elif gui.get('inventory_screen')==1:
        required=('mouse_x','mouse_y','result_x','result_y','gui_width','gui_height')
        if any(k not in gui or not math.isfinite(gui[k]) for k in required): raise ValueError('result slot geometry missing')
        if not (0<=gui['result_x']<=gui['gui_width'] and 0<=gui['result_y']<=gui['gui_height']):
            raise ValueError('invalid result slot geometry')
        camera=jsonable(action.get('camera'))
        while isinstance(camera,list) and len(camera)==1:camera=camera[0]
        if not isinstance(camera,list) or len(camera)!=2:raise ValueError('camera action missing')
        # ReplaySender applies keyboard, then clipped cursor movement, then mouse edges.
        x=max(0,min(gui['gui_width'],gui['mouse_x']+_scalar(camera[1])/0.15))
        y=max(0,min(gui['gui_height'],gui['mouse_y']+_scalar(camera[0])/0.15))
        audit['predicted_mouse']=[x,y]
        result=(gui['result_x']-9<=x<gui['result_x']+9 and gui['result_y']-9<=y<gui['result_y']+9)
        audit['reason']='result_slot' if result else 'other_inventory_region'
        if result:forwarded['sneak']=0
    return forwarded,audit


class CraftEvidence:
    def __init__(self,observation,recipe='planks'):
        self.name=recipe; self.recipe=RECIPES[recipe]
        self.target=self.recipe['target']; self.count=self.recipe['count']
        self.baseline=inventory_counts(observation)
        if (self.baseline.get(self.recipe['ingredient'])!=self.recipe['initial'] or self.baseline.get(self.target,0)
                or (recipe=='planks' and any(v for k,v in self.baseline.items() if k.endswith('_planks')))):
            raise ValueError('craft probe requires the exact recipe material budget and no target items')
        self.initial_crafted=self.crafted(observation)
        self.witness=None; self.cleanup_witness=None

    def crafted(self,observation):
        if self.target not in observation.get('craft_item',{}): raise ValueError('craft statistics unavailable')
        return inventory_counts({'inventory':observation['craft_item']})[self.target]

    def ready_for_recovery(self,observation,gui):
        cursor,grid='cursor_'+self.name,'grid_'+self.name
        if not isinstance(gui,dict) or not {cursor,grid} <= gui.keys():
            raise ValueError('craft GUI telemetry missing')
        inventory=inventory_counts(observation)
        # Input slots and carried stack are real items. The recipe preview is excluded.
        return inventory.get(self.target,0)+gui[cursor]+gui[grid]>=self.count

    def server_confirmed(self,gui):
        return bool(gui and gui.get('server_available')==1 and gui.get('server_inventory_'+self.name,0)>=self.count
                    and gui.get('server_cursor_count')==0
                    and gui.get('server_crafted_'+self.name,-1)-self.initial_crafted>=self.count)

    def final_server_confirmed(self,gui):
        # Closing the inventory returns unused grid ingredients asynchronously.
        return self.server_confirmed(gui) and (self.name=='planks' or
            gui.get('server_inventory_planks')==self.recipe['initial']-self.recipe['consumed'])

    def observe(self,observation,step,*,cleanup=False):
        inventory=inventory_counts(observation)
        ingredient=self.recipe['ingredient']
        receipt={'step':step,ingredient+'_delta':inventory.get(ingredient,0)-self.baseline[ingredient],
                 self.target+'_delta':inventory.get(self.target,0)-self.baseline.get(self.target,0),
                 'crafted_'+self.target+'_delta':self.crafted(observation)-self.initial_crafted}
        if (receipt[ingredient+'_delta']==-self.recipe['consumed'] and receipt[self.target+'_delta']==self.count
                and receipt['crafted_'+self.target+'_delta']==self.count):
            if cleanup:
                if self.cleanup_witness is None: self.cleanup_witness=receipt
            elif self.witness is None: self.witness=receipt
        return receipt


class CraftingEnvironment:
    def __init__(self,env,bundle,store,output,*,recover_cursor=False,recipe='planks',no_quick_move=False,result_slot_guard=False):
        if (no_quick_move or result_slot_guard) and not recover_cursor: raise ValueError('action guard requires cursor recovery')
        if no_quick_move and result_slot_guard: raise ValueError('select one action guard')
        self.result_slot_guard=result_slot_guard
        self.no_quick_move=no_quick_move; self.guarded_steps=0
        self.recipe=recipe
        self.env=env; self.bundle=bundle; self.store=store; self.output=output
        self.evidence=None; self.steps=0; self.gui_open=False; self.gui_open_steps=0
        self.restore=None; self.saved=None; self.cleanup=None
        self.recover_cursor=recover_cursor; self.gui=None; self.recovery_trigger=None
        self.phase='crafting'

    def seed(self,seed): return self.env.seed(seed)

    def reset(self):
        obs=self.env.reset()
        self.restore=validate_restored_observation(self.bundle,obs)
        self.evidence=CraftEvidence(obs,self.recipe)
        write_json(self.output/'craft-baseline.json',dict(observation=observation_evidence(obs),
                    initial_crafted=self.evidence.initial_crafted,world_restore=self.restore))
        return obs

    def step(self,action):
        if self.recover_cursor and self.phase!='crafting':
            raise RuntimeError('craft policy dispatch forbidden after crafting phase')
        forwarded=action
        audit=None
        if self.no_quick_move:
            if 'sneak' not in action: raise ValueError('sneak action is required for explicit Shift release')
            forwarded=deepcopy(action); forwarded['sneak']=0
            proposed=jsonable(action['sneak'])
            while isinstance(proposed,list) and len(proposed)==1: proposed=proposed[0]
            self.guarded_steps+=int(bool(proposed))
        elif self.result_slot_guard:
            forwarded,audit=guard_result_slot(action,self.gui)
            self.guarded_steps+=int(_scalar(action['sneak'])!=_scalar(forwarded['sneak']))
        obs,reward,done,info=self.env.step(forwarded)
        if self.no_quick_move:
            info=dict(info,craft_action_guard=dict(mode='sneak_disabled_entire_craft_probe',
                proposed_action=jsonable(action),forwarded_action=jsonable(forwarded)))
        elif self.result_slot_guard:
            info=dict(info,craft_action_guard=dict(mode='result_slot_only',**audit,
                proposed_action=jsonable(action),forwarded_action=jsonable(forwarded)))
            observed_gui=info.get('craft_gui') or {}
            if audit['predicted_mouse'] is not None and observed_gui.get('inventory_screen')==1:
                actual=[observed_gui['mouse_x'],observed_gui['mouse_y']]
                if any(abs(a-b)>1 for a,b in zip(actual,audit['predicted_mouse'])):
                    raise RuntimeError('cursor prediction mismatch; refusing further guarded policy actions')
        self.steps+=1
        if 'isGuiOpen' not in info: raise ValueError('GUI monitor missing')
        self.gui_open=bool(info['isGuiOpen']); self.gui_open_steps+=int(self.gui_open)
        self.gui=info.get('craft_gui')
        if self.result_slot_guard and self.gui is not None:
            self.gui=dict(self.gui,other_screen_open=self.gui_open and self.gui.get('inventory_screen')!=1)
        if self.recover_cursor and self.evidence.ready_for_recovery(obs,self.gui) and self.recovery_trigger is None:
            self.recovery_trigger=dict(step=self.steps,gui=jsonable(self.gui),observation=observation_evidence(obs))
            self.phase='recovering'
        if self.recover_cursor:
            info=dict(info,craft_state=dict(phase=self.phase,policy_allowed=self.phase=='crafting',accepting=False))
        self.evidence.observe(obs,self.steps)
        return obs,reward,done,info

    def release(self):
        if self.evidence is None: return self.env.release()
        before=self.env.last_observation
        save_png(before['pov'],self.output/'craft-before-cleanup.png')
        self.cleanup=dict(before=observation_evidence(before),gui_open_before=self.gui_open,gui_close_attempted=False)
        if self.recover_cursor:
            result=self._release_with_cursor_check()
        else: result=self.env.release()
        if self.gui_open and not self.env.terminal:
            action=self.env.raw.action_space.noop(); action['inventory']=1
            self.cleanup['gui_close_attempted']=True
            if self.recover_cursor:
                self._cleanup_step(action,'close_inventory')
                self._cleanup_step(self.env.raw.action_space.noop(),'release_inventory_key')
                self.cleanup['gui_open_after']=self.gui_open
            else:
                _,_,_,info=self.env.step(action)
                self.cleanup['gui_open_after']=bool(info['isGuiOpen'])
                self.env.release()
        else: self.cleanup['gui_open_after']=self.gui_open
        if self.recover_cursor and not self.env.terminal:
            # Client inventory and server statistics can arrive on different ticks.
            for _ in range(8):
                if self.evidence.crafted(self.env.last_observation)-self.evidence.initial_crafted>=self.evidence.count and self.evidence.final_server_confirmed(self.gui): break
                self._cleanup_step(self.env.raw.action_space.noop(),'settle_server_statistics')
            self.cleanup['cursor_verified_empty']=self.gui['cursor_count']==0
        after=self.env.last_observation
        self.evidence.observe(after,self.steps,cleanup=True)
        if self.recover_cursor:
            self.phase='completed' if (self.evidence.cleanup_witness and self.cleanup.get('cursor_verified_empty') and
                not self.cleanup['gui_open_after'] and self.evidence.final_server_confirmed(self.gui)) else 'failed'
            self.cleanup['craft_state']=dict(phase=self.phase,policy_allowed=False,accepting=self.phase=='completed')
        self.cleanup['after']=observation_evidence(after)
        self.cleanup['cursor_item_verification']=('read-only carried-stack monitor' if self.recover_cursor else
            'not exposed by current observation; inspect before/after frames')
        save_png(after['pov'],self.output/'craft-after-cleanup.png')
        write_json(self.output/'craft-cleanup.json',self.cleanup)
        return result

    def _cleanup_step(self,action,label):
        obs,_,_,info=self.env.step(action)
        self.gui=info.get('craft_gui'); self.gui_open=bool(info['isGuiOpen'])
        self.cleanup.setdefault('actions',[]).append(dict(controller='cleanup',label=label,
            action=jsonable(action),gui=jsonable(self.gui),observation=observation_evidence(obs)))
        if self.gui is None: raise ValueError('craft GUI telemetry missing during cleanup')
        return obs

    def _release_with_cursor_check(self):
        if self.env.terminal: return 'environment_done_no_neutral_step'
        noop=self.env.raw.action_space.noop
        self._cleanup_step(noop(),'neutral_release')
        # Return a carried stack using ordinary mouse input BEFORE closing the screen.
        if self.gui['cursor_count']:
            if not self.gui['inventory_screen'] or self.gui['empty_slot']<0:
                raise RuntimeError('cannot safely return cursor stack: no empty inventory slot')
            x,y=self.gui['target_x'],self.gui['target_y']; slot=self.gui['empty_slot']
            action=noop(); action['camera']=[(y-self.gui['mouse_y'])*.15,(x-self.gui['mouse_x'])*.15]
            if any(abs(v)>180 for v in action['camera']): raise RuntimeError('cursor displacement outside action range')
            self._cleanup_step(action,'move_to_empty_inventory_slot')
            if (not self.gui['inventory_screen'] or self.gui['empty_slot']!=slot or
                abs(self.gui['mouse_x']-x)>1 or abs(self.gui['mouse_y']-y)>1):
                raise RuntimeError('cursor movement did not reach verified empty inventory slot')
            action=noop(); action['attack']=1
            self._cleanup_step(action,'place_carried_stack')
            self._cleanup_step(noop(),'release_mouse')
        if self.gui['cursor_count']: raise RuntimeError('cursor stack was not returned')
        if self.recovery_trigger:
            for _ in range(12):
                if self.evidence.server_confirmed(self.gui):break
                self._cleanup_step(noop(),'await_server_inventory_commit')
            if not self.evidence.server_confirmed(self.gui):
                raise RuntimeError('client inventory was not committed on server; refusing early GUI close')
        self.cleanup['cursor_verified_empty']=True
        return 'neutral_command_sent'

    def close(self):
        try:
            if self.evidence is not None:
                world=self.env.save_and_quit()
                checkpoint=dict(schema=1,world_id=self.bundle['world_id'],task=None,last_observation=self.steps)
                destination=self.store.commit(world,checkpoint,self.env.last_observation)
                self.saved=dict(status='saved',generation=destination.name)
        finally: self.env.close()


def run_crafting_probe(policy,env,output,*,bundle,store,stop_on_success=False,recover_cursor=False,recipe='planks',no_quick_move=False,result_slot_guard=False,**kwargs):
    if bundle is None: raise ValueError('craft probe requires an existing saved world')
    specification=RECIPES[recipe]
    wrapped=CraftingEnvironment(env,bundle,store,output,recover_cursor=recover_cursor,recipe=recipe,no_quick_move=no_quick_move,result_slot_guard=result_slot_guard)
    if getattr(policy,'controller_scope',None):policy.environment=wrapped
    complete=((lambda obs: wrapped.recovery_trigger is not None) if recover_cursor else
              (lambda obs: wrapped.evidence.witness is not None) if stop_on_success else None)
    report=run_episode(policy,wrapped,output,goal=specification['goal'],target_items=(specification['target'],),target_count=specification['count'],
                       completion_check=complete,completion_status='craft_ready_for_recovery' if recover_cursor else 'target_reached',**kwargs)
    evidence=wrapped.evidence
    report.update(craft_probe=True,recipe=recipe,recipe_specification=specification,world_restore=wrapped.restore,world_save=wrapped.saved,
                  no_quick_move=no_quick_move,result_slot_guard=result_slot_guard,guarded_policy_steps=wrapped.guarded_steps,
                  stop_on_success=stop_on_success,
                  planks_retained_after_cleanup=bool(evidence and evidence.witness and wrapped.cleanup and
                      wrapped.cleanup.get('after',{}).get('inventory',{}).get('oak_planks',0)>=4),
                  condition=getattr(policy,'condition_metadata',None),
                  craft_success=bool(evidence and evidence.witness),
                  craft_witness=evidence.witness if evidence else None,
                  cleanup_witness=evidence.cleanup_witness if evidence else None,
                  gui_open_steps=wrapped.gui_open_steps,craft_cleanup=jsonable(wrapped.cleanup),
                  cursor_item_verified=bool(wrapped.cleanup and wrapped.cleanup.get('cursor_verified_empty')),
                  recover_cursor=recover_cursor,recovery_trigger=wrapped.recovery_trigger,
                  craft_state=dict(phase=wrapped.phase,policy_allowed=False,
                      accepting=bool(wrapped.phase=='completed' and report['execution_valid'] and wrapped.saved)),
                  task_success_after_cleanup=bool(evidence and evidence.cleanup_witness and
                      report['execution_valid'] and wrapped.saved and wrapped.saved['status']=='saved' and
                      (not recover_cursor or wrapped.phase=='completed')),
                  controller_scope='STEVE-1 crafting + explicit GUI cleanup assistance' if recover_cursor else 'STEVE-1 policy')
    if no_quick_move: report['controller_scope']+=' + explicit sneak/Shift action constraint for entire probe'
    if result_slot_guard: report['controller_scope']+=' + explicit result-slot Shift constraint (bootstrap/GUI-transition fallback)'
    if getattr(policy,'controller_scope',None):
        report['controller_scope']=policy.controller_scope+' + explicit cursor recovery'
        report['controller']='recipe_book_gui';report['learned_policy_loaded']=False
        report['gui_decision_ms']=report.pop('inference_ms')
    write_json(output/'summary.json',report)
    return report
