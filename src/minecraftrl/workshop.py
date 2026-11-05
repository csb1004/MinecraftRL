"""Recipe-driven ordinary GUI actions, with server-confirmed commits between skills.

This is explicit assistance, not pretrained inference or reinforcement learning.
"""
from collections import Counter
import copy
import json
import math
from .runtime import run_episode, write_json, save_png, jsonable
from .world_store import validate_restored_observation
from .workshop_recipes import short


def nonzero(values):return {k:int(v) for k,v in values.items() if v}


def recipe_selection(recipe):
    if not all(recipe[k]==1 for k in ('known','server_known','craftable')):
        raise ValueError('recipe locked or not craftable')
    if recipe['x']>=0 and (recipe['single']==1 or recipe.get('overlay')==1):
        return 'attack',recipe['x'],recipe['y']
    if recipe.get('group_x',-1)>=0 and not recipe['single']:
        return 'use',recipe['group_x'],recipe['group_y']
    raise ValueError('recipe has no unambiguous visible selection')


def table_aim_points(pos,eye):
    points=[tuple(v+.5 for v in pos)]
    for axis in range(3):
        for offset in (.001,.999):
            point=[v+.5 for v in pos];point[axis]=pos[axis]+offset;points.append(tuple(point))
    return sorted(points,key=lambda point:sum((p-e)**2 for p,e in zip(point,eye)))


class WorkshopController:
    def __init__(self,catalog,inventory,goals,output,*,equip=None,equip_only=False,equip_slot=None,consume_food=None):
        self.catalog=catalog;self.expected=Counter(inventory);self.goals=goals;self.output=output
        self.plan=[];self.final_inventory=None  # Account for a reachable existing table after observing it.
        if equip is not None and equip not in goals:raise ValueError('equipped tool must be a workshop goal')
        self.equip=equip
        self.equip_only=equip_only
        if equip_slot is not None and (equip is None or type(equip_slot) is not int or not 1<=equip_slot<=9):
            raise ValueError('tool hotbar slot requires an explicit tool and index1..9')
        self.equip_slot=equip_slot
        if equip_only and (equip is None or goals!=[equip] or inventory.get(equip,0)<1):
            raise ValueError('equipment-only assistance requires an already owned tool')
        if consume_food is not None and (consume_food!='apple' or not equip_only or inventory.get(consume_food,0)<1):
            raise ValueError('food assistance requires an owned apple and explicit equipment-only mode')
        self.consume_food=consume_food;self.consumption=None
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

    def click(self,x,y,check=None,*,button='attack'):
        if min(x,y)<0:raise ValueError('GUI target unavailable')
        action=self.noop();action['camera']=[(y-self.gui['mouse_y'])*.15,(x-self.gui['mouse_x'])*.15]
        if any(abs(v)>180 for v in action['camera']):raise ValueError('mouse movement outside action space')
        yield action
        if abs(self.gui['mouse_x']-x)>1 or abs(self.gui['mouse_y']-y)>1:raise ValueError('mouse missed target')
        if check:check()
        yield from self.key(button)

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
            if recipe['x']>=0 or recipe.get('group_x',-1)>=0:
                button,x,y=recipe_selection(recipe)
                def check():
                    if self.gui['recipes'][key]!=recipe or any(s['count'] for s in self.grid()):
                        raise ValueError('recipe selection changed or input grid nonempty')
                yield from self.click(x,y,check,button=button)
                if button=='use':
                    yield from self.wait(lambda:self.gui['overlay_open']==1,'open recipe variants: '+key)
                    self.mark('recipe_variants_opened',recipe=key)
                    continue
                return
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
        yield from self.open_table(pos)

    def open_table(self,pos):
        self.stage='open_workstation'
        def hit():return self.gui['hit_table'] and tuple(int(self.gui['hit_'+a]) for a in 'xyz')==pos
        if not hit():
            for point in table_aim_points(pos,tuple(self.gui['eye_'+a] for a in 'xyz')):
                yield from self.aim(*point)
                if hit():break
        if not hit():raise ValueError('no visible reachable workstation face')
        yield from self.key('use')
        yield from self.wait(lambda:self.gui['screen']==2 and self.gui['width']==3,'open 3x3 workstation')
        self.table=pos;self.mark('workstation_opened',position=pos)

    def relocate_tool(self,slot,target_index):
        destination=next(i for i,s in enumerate(self.gui['slots']) if s['inventory_index']==target_index)
        if slot==destination:return slot
        if self.gui['cursor_count'] or self.gui['server_cursor']:raise ValueError('tool relocation requires an empty cursor')
        source=dict(self.gui['slots'][slot]);displaced=dict(self.gui['slots'][destination])
        yield from self.slot_click(slot)
        yield from self.wait(lambda:self.gui['cursor_item']==self.equip and self.gui['cursor_count']==source['count']
                            and self.gui['slots'][slot]['count']==0,'pick up owned tool')
        yield from self.slot_click(destination)
        yield from self.wait(lambda:self.gui['slots'][destination]['item']==self.equip
                            and self.gui['slots'][destination]['count']==source['count']
                            and self.gui['cursor_count']==displaced['count']
                            and (not displaced['count'] or self.gui['cursor_item']==displaced['item']),'swap tool into preferred slot')
        if displaced['count']:
            if self.gui['slots'][slot]['count']:raise ValueError('original tool slot became occupied')
            yield from self.slot_click(slot)
        yield from self.wait(self.committed,'tool relocation preserves inventory')
        self.mark('tool_relocated',item=self.equip,from_index=source['inventory_index'],to_index=target_index)
        return destination

    def equip_tool(self):
        self.stage='equip_tool'
        if self.gui['screen']!=1:
            if self.gui['screen']!=0:yield from self.key('inventory')
            yield from self.key('inventory')
            yield from self.wait(lambda:self.gui['screen']==1,'open inventory for tool')
        slot=next((i for i,s in enumerate(self.gui['slots']) if s['item']==self.equip and s['count']>=1
                   and 0<=s['inventory_index']<36),None)
        if slot is None:raise ValueError('crafted tool unavailable for equip')
        if self.equip_slot is not None:slot=yield from self.relocate_tool(slot,self.equip_slot-1)
        index=self.gui['slots'][slot]['inventory_index']
        if index>=9:
            yield from self.slot_click(slot);yield from self.return_cursor()
            index=next(s['inventory_index'] for s in self.gui['slots'] if s['item']==self.equip and s['count']==1)
        if not 0<=index<9:raise ValueError('no empty hotbar slot for tool')
        yield from self.key('inventory')
        yield from self.wait(lambda:self.gui['screen']==0,'close inventory before equip')
        yield from self.key('hotbar.'+str(int(index)+1))
        yield from self.wait(lambda:self.gui['server_selected_item']==self.equip,'server equipped tool')
        self.mark('tool_equipped',item=self.equip)

    def consume_owned_food(self):
        initial_food=self.gui['server_food']
        if not 0<=initial_food<20 or self.gui['server_alive']!=1:
            raise ValueError('food assistance requires a living hungry player')
        food=self.consume_food;expected=copy.deepcopy(self.expected);expected[food]-=1
        target_food=min(20,initial_food+4)  # Pinned 1.16.5 Foods.APPLE nutrition.
        tool,slot=self.equip,self.equip_slot
        try:
            self.equip=food;self.equip_slot=9
            yield from self.equip_tool()
            self.stage='consume_owned_food'
            for _ in range(64):
                if (nonzero(self.gui['server_inventory'])==nonzero(expected)
                        and self.gui['server_food']>=target_food):
                    self.expected=expected
                    self.consumption=dict(item=food,quantity=1,initial_food=initial_food,
                        final_food=self.gui['server_food'],target_food=target_food)
                    self.mark('food_consumed',**self.consumption)
                    yield self.noop()
                    return
                if self.gui['screen']!=0 or self.gui['server_alive']!=1:
                    raise ValueError('food use requires a living player outside GUI')
                action=self.noop();action['use']=1;yield action
            raise ValueError('food consumption not confirmed by inventory and hunger')
        finally:self.equip,self.equip_slot=tool,slot

    def actions(self):
        yield self.noop()
        yield from self.wait(self.committed,'restored server inventory')
        if self.equip_only:
            if self.consume_food:yield from self.consume_owned_food()
            self.final_inventory=copy.deepcopy(self.expected)
            yield from self.equip_tool()
            self.stage='finish'
            for _ in range(30):yield self.noop()
            raise RuntimeError('equipment completion boundary was not accepted')
        existing=None
        if self.gui['table_count']>0:
            pos=tuple(int(self.gui['table_'+a]) for a in 'xyz')
            if sum((pos[i]+.5-self.gui['eye_'+a])**2 for i,a in enumerate('xyz'))<=3.7**2:
                existing=pos
        self.plan,self.final_inventory=self.catalog.plan(self.expected,self.goals,workstation_available=existing is not None)
        yield from self.key('inventory')
        yield from self.wait(lambda:self.gui['screen']==1,'open 2x2 inventory')
        for key in self.plan:
            if max(self.catalog.dimensions(key))>self.gui['width']:
                if existing is not None:
                    yield from self.key('inventory')
                    yield from self.wait(lambda:self.gui['screen']==0,'close inventory to reuse table')
                    yield from self.open_table(existing)
                else:yield from self.place_table()
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
        if self.table is None:
            if existing is not None:
                yield from self.key('inventory')
                yield from self.wait(lambda:self.gui['screen']==0,'close inventory to reuse table')
                yield from self.open_table(existing)
            else:yield from self.place_table()
        if self.equip:yield from self.equip_tool()
        else:yield from self.key('inventory')
        self.stage='finish'
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
            if not all(self.gui['server_inventory'].get(g,0)>=1 for g in self.controller.goals):
                raise ValueError('workshop final requirements not satisfied')
            if not self.controller.equip_only:
                if self.controller.table is None or tuple(int(self.gui['table_'+a]) for a in 'xyz')!=self.controller.table:
                    raise ValueError('placed workstation no longer observed on server')
            if self.controller.equip and self.gui['server_selected_item']!=self.controller.equip:
                raise ValueError('server tool equip requirement not satisfied')
            if self.controller.consume_food and (not self.controller.consumption
                    or self.gui['server_food']<self.controller.consumption['target_food']):
                raise ValueError('food recovery was not retained')
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
                            screen=self.gui['screen'],inventory=nonzero(self.gui['server_inventory']),
                            held_key_count=self.gui['held_key_count'],server_selected_item=self.gui['server_selected_item'])
        self.cleanup.update(server_food=self.gui.get('server_food'),server_health=self.gui.get('server_health'))
        if not self.cleanup['cursor_empty'] or self.gui['screen']!=0:raise RuntimeError('workshop cleanup incomplete')
        if self.completed and self.cleanup['inventory']!=nonzero(self.controller.expected):
            raise RuntimeError('completed inventory changed during cleanup')
        return 'neutral_command_sent'
    def close(self):
        try:
            if self.restore:
                world=self.env.save_and_quit()
                # Crafting consumes materials; it must not erase previously achieved goals or their history.
                checkpoint=copy.deepcopy(self.bundle['task_checkpoint'])
                state=copy.deepcopy(self.bundle.get('controller_state',{}))
                state['last_workshop']=dict(steps=self.steps,completed=self.completed,goals=self.controller.goals,
                    mode='food_then_equipment' if getattr(self.controller,'consume_food',None) else 'equipment_only' if getattr(self.controller,'equip_only',False) else 'crafting',
                    food_consumption=getattr(self.controller,'consumption',None),
                    milestones=self.controller.milestones,policy_memory_invalidated='GUI actions were not observed by neural policy')
                destination=self.store.commit(world,checkpoint,self.env.last_observation,controller_state=state)
                self.saved=dict(status='saved',generation=destination.name)
        finally:self.env.close()


def run_workshop(policy,env,output,*,bundle,store,**kwargs):
    wrapped=WorkshopEnvironment(env,policy,bundle,store,output);policy.environment=wrapped
    goal=('equip existing '+policy.equip if policy.equip_only else 'craft and place workstation; craft '+', '.join(policy.goals))
    report=run_episode(policy,wrapped,output,goal=goal,
                       target_items=policy.goals,target_count=len(policy.goals),
                       completion_check=lambda obs:wrapped.completed,**kwargs)
    report.update(controller='workshop_gui',learned_policy_loaded=False,api_calls=0,goals=policy.goals,
                  workshop_mode='food_then_equipment' if policy.consume_food else 'equipment_only' if policy.equip_only else 'crafting',
                  food_consumption=policy.consumption,
                  preferred_hotbar_slot=policy.equip_slot,
                  recipe_plan=policy.plan,milestones=policy.milestones,table_position=policy.table,equipped_tool=policy.equip,
                  expected_inventory=nonzero(policy.expected),world_restore=wrapped.restore,world_save=wrapped.saved,
                  workshop_cleanup=wrapped.cleanup,workshop_success=bool(wrapped.completed and report['execution_valid'] and wrapped.saved),
                  controller_scope='Explicit ordinary mouse/key GUI assistance with read-only server telemetry; no learned policy or RL')
    report['gui_decision_ms']=report.pop('inference_ms');write_json(output/'summary.json',report);return report
