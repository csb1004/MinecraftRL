"""Recipe-driven ordinary GUI actions, with server-confirmed commits between skills.

This is explicit assistance, not pretrained inference or reinforcement learning.
"""
from collections import Counter
import json
import math
from .runtime import run_episode, write_json, save_png, jsonable
from .world_store import validate_restored_observation
from .workshop_recipes import short


def nonzero(values):return {k:int(v) for k,v in values.items() if v}


class WorkshopController:
    def __init__(self,catalog,inventory,goals,output):
        self.catalog=catalog;self.expected=Counter(inventory);self.goals=goals;self.output=output
        self.plan,self.final_inventory=catalog.plan(inventory,goals)
        self.environment=None;self.stage='initializing';self.milestones=[];self.table=None

    @property
    def gui(self):
        value=self.environment.gui
        if not value or value['error'] or not value['server_available']:raise ValueError('workshop telemetry unavailable')
        return value

    def noop(self):return self.environment.env.raw.action_space.noop()
    def reset(self,goal=None):self.sequence=self.actions()
    def close(self):pass
    def act(self,screen):
        action=next(self.sequence)
        save_png(screen,self.output/('workshop-'+self.stage+'.png'))
        return action

    def wait(self,predicate,label,limit=30):
        for _ in range(limit):
            if predicate():return
            yield self.noop()
        if not predicate():raise ValueError('workshop timeout: '+label)

    def key(self,key):
        action=self.noop();action[key]=1;yield action;yield self.noop()

    def click(self,x,y,check=None):
        if min(x,y)<0:raise ValueError('GUI target unavailable')
        action=self.noop();action['camera']=[(y-self.gui['mouse_y'])*.15,(x-self.gui['mouse_x'])*.15]
        if any(abs(v)>180 for v in action['camera']):raise ValueError('mouse movement outside action space')
        yield action
        if abs(self.gui['mouse_x']-x)>1 or abs(self.gui['mouse_y']-y)>1:raise ValueError('mouse missed target')
        if check:check()
        yield from self.key('attack')

    def slot_click(self,index):
        slot=dict(self.gui['slots'][index]);screen=self.gui['screen']
        def check():
            if self.gui['screen']!=screen or self.gui['slots'][index]!=slot:raise ValueError('slot changed before click')
        yield from self.click(slot['x'],slot['y'],check)

    def return_cursor(self):
        if not self.gui['cursor_count']:return
        slots=[(i,s) for i,s in enumerate(self.gui['slots']) if 0<=s['inventory_index']<36 and not s['count']]
        if not slots:raise ValueError('no empty inventory slot for cursor recovery')
        # Keeping outputs in the hotbar also makes newly crafted workstations usable.
        slots.sort(key=lambda pair:(pair[1]['inventory_index']>=9,pair[1]['inventory_index']))
        yield from self.slot_click(slots[0][0])
        yield from self.wait(lambda:not self.gui['cursor_count'] and not self.gui['server_cursor'],'cursor return')

    def committed(self):
        return (not self.gui['server_cursor'] and not self.gui['cursor_count'] and
                nonzero(self.gui['server_inventory'])==nonzero(self.expected))

    def grid(self):return self.gui['slots'][1:1+int(self.gui['width'])**2]

    def valid_grid(self,key):
        cells=self.grid();r=self.catalog.recipes[key];preview=self.gui['slots'][0]
        return (self.catalog.matches(key,int(self.gui['width']),[s['item'] for s in cells],[s['count'] for s in cells]) and
                preview['item']==short(r['result']['item']) and preview['count']==r['result'].get('count',1) and
                not self.gui['cursor_count'])

    def mark(self,event,**fields):
        self.milestones.append(dict(event=event,step=self.environment.steps,inventory=nonzero(self.gui['server_inventory']),**fields))

    def find_recipe(self,key):
        yield from self.wait(lambda:all(self.gui['recipes'][key][k]==1 for k in ('known','server_known')),
                             'recipe unlock synchronization: '+key)
        if not self.gui['book_open']:yield from self.click(self.gui['book_x'],self.gui['book_y'])
        if not self.gui['search_tab']:yield from self.click(self.gui['search_x'],self.gui['search_y'])
        for _ in range(30):
            if self.gui['page']<=0:break
            yield from self.click(self.gui['previous_x'],self.gui['previous_y'])
        for _ in range(30):
            recipe=self.gui['recipes'][key]
            if recipe['x']>=0:
                if not all(recipe[k]==1 for k in ('known','server_known','single','craftable')):
                    raise ValueError('recipe locked, ambiguous, or not craftable: '+key)
                def check():
                    if self.gui['recipes'][key]!=recipe or any(s['count'] for s in self.grid()):
                        raise ValueError('recipe selection changed or input grid nonempty')
                yield from self.click(recipe['x'],recipe['y'],check);return
            if self.gui['next_x']<0:break
            yield from self.click(self.gui['next_x'],self.gui['next_y'])
        raise ValueError('recipe absent from book: '+key)

    def aim(self,x,y,z):
        dx=x-self.gui['eye_x'];dy=y-self.gui['eye_y'];dz=z-self.gui['eye_z']
        yaw=math.degrees(math.atan2(-dx,dz));pitch=-math.degrees(math.atan2(dy,math.hypot(dx,dz)))
        action=self.noop();action['camera']=[pitch-self.gui['pitch'],(yaw-self.gui['yaw']+180)%360-180]
        if any(abs(v)>180 for v in action['camera']):raise ValueError('aim outside action space')
        yield action;yield self.noop()

    def place_table(self):
        self.stage='equip_workstation'
        table_slot=next((i for i,s in enumerate(self.gui['slots']) if s['item']=='crafting_table' and s['count']==1 and 0<=s['inventory_index']<36),None)
        if table_slot is None:raise ValueError('crafted workstation missing')
        index=self.gui['slots'][table_slot]['inventory_index']
        if index>=9:
            yield from self.slot_click(table_slot);yield from self.return_cursor()
            index=next(s['inventory_index'] for s in self.gui['slots'] if s['item']=='crafting_table' and s['count']==1)
        if not 0<=index<9:raise ValueError('no empty hotbar slot for workstation')
        yield from self.key('inventory')
        yield from self.wait(lambda:self.gui['screen']==0,'close inventory')
        yield from self.key('hotbar.'+str(int(index)+1))
        yield from self.wait(lambda:self.gui['selected_item']=='crafting_table','equip table')
        if not self.gui['candidate']:raise ValueError('no reachable safe placement surface')
        pos=tuple(int(self.gui['candidate_'+axis]) for axis in 'xyz');before=self.gui['table_count']
        self.stage='place_workstation'
        yield from self.aim(pos[0]+.5,pos[1]-.001,pos[2]+.5)
        if tuple(int(self.gui['hit_'+a]) for a in 'xyz')!=(pos[0],pos[1]-1,pos[2]) or self.gui['hit_face']!=1:
            raise ValueError('aim did not hit verified support top')
        if self.gui['selected_item']!='crafting_table':raise ValueError('held item changed before placement')
        yield from self.key('use');self.expected['crafting_table']-=1
        yield from self.wait(lambda:self.committed() and self.gui['table_count']==before+1 and
                             tuple(int(self.gui['table_'+a]) for a in 'xyz')==pos,'server workstation placement')
        self.table=pos;self.mark('workstation_placed',position=pos)
        self.stage='open_workstation'
        yield from self.aim(pos[0]+.5,pos[1]+.999,pos[2]+.5)
        if not self.gui['hit_table'] or tuple(int(self.gui['hit_'+a]) for a in 'xyz')!=pos:
            raise ValueError('aim did not hit placed workstation')
        yield from self.key('use')
        yield from self.wait(lambda:self.gui['screen']==2 and self.gui['width']==3,'open 3x3 workstation')
        self.mark('workstation_opened',position=pos)

    def actions(self):
        yield self.noop()
        yield from self.wait(self.committed,'restored server inventory')
        yield from self.key('inventory')
        yield from self.wait(lambda:self.gui['screen']==1,'open 2x2 inventory')
        for key in self.plan:
            if max(self.catalog.dimensions(key))>self.gui['width']:yield from self.place_table()
            if any(s['count'] for s in self.grid()):raise ValueError('nonempty grid before recipe placement')
            self.stage='select_'+key
            yield from self.find_recipe(key)
            yield from self.wait(lambda:self.valid_grid(key),'exact grid and preview: '+key)
            ingredients=Counter()
            for slot in self.grid():
                if slot['count']:ingredients[slot['item']]+=int(slot['count'])
            if self.gui['book_open']:yield from self.click(self.gui['book_x'],self.gui['book_y'])
            self.stage='take_'+key
            def check():
                if not self.valid_grid(key):raise ValueError('recipe changed before result click')
            preview=self.gui['slots'][0]
            yield from self.click(preview['x'],preview['y'],check)
            result=self.catalog.recipes[key]['result'];count=result.get('count',1)
            yield from self.wait(lambda:self.gui['cursor_item']==short(result['item']) and self.gui['cursor_count']==count,'result on cursor')
            self.expected.subtract(ingredients);self.expected[short(result['item'])]+=count
            if any(n<0 for n in self.expected.values()):raise ValueError('recipe exceeded material budget')
            self.stage='store_'+key
            yield from self.return_cursor()
            yield from self.wait(lambda:self.committed() and not any(s['count'] for s in self.grid()),'server recipe commit: '+key)
            self.mark('recipe_committed',recipe=key,ingredients=dict(ingredients),result=result)
        self.stage='finish'
        yield from self.key('inventory')
        for _ in range(30):yield self.noop()
        raise RuntimeError('completion boundary was not accepted')


class WorkshopEnvironment:
    def __init__(self,env,controller,bundle,store,output):
        self.env=env;self.controller=controller;self.bundle=bundle;self.store=store;self.output=output
        self.gui=None;self.steps=0;self.completed=False;self.restore=None;self.saved=None;self.cleanup=None
    def seed(self,seed):return self.env.seed(seed)
    def reset(self):
        obs=self.env.reset();self.restore=validate_restored_observation(self.bundle,obs);return obs
    def observe(self,transition):
        obs,reward,done,info=transition;self.gui=info['workshop']
        return obs,reward,done,dict(info,workshop_stage=self.controller.stage)
    def step(self,action):
        if self.completed:raise RuntimeError('workshop actions forbidden after completion')
        self.steps+=1;transition=self.observe(self.env.step(action))
        if self.controller.stage=='finish' and self.gui['screen']==0 and self.controller.committed():
            if self.controller.table is None or not all(self.gui['server_inventory'].get(g,0)>=1 for g in self.controller.goals):
                raise ValueError('workshop final requirements not satisfied')
            if tuple(int(self.gui['table_'+a]) for a in 'xyz')!=self.controller.table:
                raise ValueError('placed workstation no longer observed on server')
            self.completed=True;self.controller.mark('completed')
        return transition
    def release(self):
        if not self.env.initialized:return self.env.release()
        self.cleanup={'steps':0}
        def dispatch(action):
            self.observe(self.env.step(action));self.cleanup['steps']+=1
            with (self.output/'workshop-cleanup.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(jsonable(dict(controller='cleanup',action=action,screen=self.gui['screen'],
                    cursor_item=self.gui['cursor_item'],cursor_count=self.gui['cursor_count'],
                    server_cursor=self.gui['server_cursor'],inventory=nonzero(self.gui['server_inventory']))))+'\n')
        dispatch(self.controller.noop())
        for action in self.controller.return_cursor():dispatch(action)
        if self.gui['screen'] in (1,2):
            for action in self.controller.key('inventory'):dispatch(action)
        dispatch(self.controller.noop())
        self.cleanup.update(cursor_empty=self.gui['cursor_count']==0 and self.gui['server_cursor']==0,
                            screen=self.gui['screen'],inventory=nonzero(self.gui['server_inventory']))
        if not self.cleanup['cursor_empty'] or self.gui['screen']!=0:raise RuntimeError('workshop cleanup incomplete')
        if self.completed and self.cleanup['inventory']!=nonzero(self.controller.expected):
            raise RuntimeError('completed inventory changed during cleanup')
        return 'neutral_command_sent'
    def close(self):
        try:
            if self.restore:
                world=self.env.save_and_quit()
                checkpoint=dict(schema=1,world_id=self.bundle['world_id'],task=None,last_observation=self.steps)
                destination=self.store.commit(world,checkpoint,self.env.last_observation)
                self.saved=dict(status='saved',generation=destination.name)
        finally:self.env.close()


def run_workshop(policy,env,output,*,bundle,store,**kwargs):
    wrapped=WorkshopEnvironment(env,policy,bundle,store,output);policy.environment=wrapped
    report=run_episode(policy,wrapped,output,goal='craft and place workstation; craft '+', '.join(policy.goals),
                       target_items=policy.goals,target_count=len(policy.goals),
                       completion_check=lambda obs:wrapped.completed,**kwargs)
    report.update(controller='workshop_gui',learned_policy_loaded=False,api_calls=0,goals=policy.goals,
                  recipe_plan=policy.plan,milestones=policy.milestones,table_position=policy.table,
                  expected_inventory=nonzero(policy.expected),world_restore=wrapped.restore,world_save=wrapped.saved,
                  workshop_cleanup=wrapped.cleanup,workshop_success=bool(wrapped.completed and report['execution_valid'] and wrapped.saved),
                  controller_scope='Explicit ordinary mouse/key GUI assistance with read-only server telemetry; no learned policy or RL')
    report['gui_decision_ms']=report.pop('inference_ms');write_json(output/'summary.json',report);return report
