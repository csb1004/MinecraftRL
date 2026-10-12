"""Explicit GUI assistance for a fixed 2x2 recipe-book experiment, not STEVE-1."""
from .crafting_probe import RECIPES
from .runtime import save_png


def require_recipe_button(gui,item):
    if (gui.get('book_error')!=0 or gui.get('book_open')!=1 or
        gui.get('known_'+item)!=1 or gui.get('server_known_'+item)!=1 or
        gui.get('book_'+item+'_single')!=1 or gui.get('book_'+item+'_craftable')!=1 or
        gui.get('book_'+item+'_x',-1)<0 or gui.get('book_'+item+'_y',-1)<0):
        raise ValueError('recipe unavailable, ambiguous, locked, or not craftable: '+item)
    return gui['book_'+item+'_x'],gui['book_'+item+'_y']


def verify_grid(gui,recipe):
    spec=RECIPES[recipe]
    if recipe not in ('sticks','table'):raise ValueError('book probe supports sticks/table only')
    counts=tuple(gui['input_'+str(i)+'_count'] for i in range(1,5))
    allowed=((1,0,1,0),(0,1,0,1)) if recipe=='sticks' else ((1,1,1,1),)
    if counts not in allowed or any(gui['input_'+str(i)+'_item']!=gui['oak_planks_id']
                                   for i,n in enumerate(counts,1) if n):
        raise ValueError('wrong recipe shape, ingredient, or material budget')
    if (gui['preview_item']!=gui[spec['target']+'_id'] or gui['preview_count']!=spec['count'] or
        gui['cursor_count']!=0):
        raise ValueError('wrong result preview or nonempty cursor')


class RecipeBookController:
    controller_scope='Explicit recipe-book GUI assistance; no learned policy inference'
    def __init__(self,recipe,output):
        self.recipe=recipe;self.output=output;self.environment=None;self.stage='initializing'
        self.condition_metadata={'controller':'recipe_book_gui','learned_policy_loaded':False,'recipe':recipe}

    def reset(self,goal=None):self.sequence=self.actions()
    def close(self):pass
    def noop(self):return self.environment.env.raw.action_space.noop()
    @property
    def gui(self):
        gui=self.environment.gui
        if not gui or gui.get('book_error')!=0:raise ValueError('recipe book telemetry unavailable')
        return gui

    def click(self,x,y,check=None):
        action=self.noop();action['camera']=[(y-self.gui['mouse_y'])*.15,(x-self.gui['mouse_x'])*.15]
        if any(abs(v)>180 for v in action['camera']):raise ValueError('cursor displacement exceeds action range')
        yield action
        if abs(self.gui['mouse_x']-x)>1 or abs(self.gui['mouse_y']-y)>1:raise ValueError('cursor missed GUI target')
        if check:check()
        action=self.noop();action['attack']=1;yield action
        yield self.noop()

    def actions(self):
        self.stage='open_inventory'
        action=self.noop();action['inventory']=1;yield action
        yield self.noop()
        if self.gui['inventory_screen']!=1:raise ValueError('inventory screen did not open')
        if not self.gui['book_open']:
            self.stage='open_book'
            yield from self.click(self.gui['book_toggle_x'],self.gui['book_toggle_y'])
        self.stage='select_recipe'
        item=RECIPES[self.recipe]['target']
        x,y=require_recipe_button(self.gui,item)
        def check_button():
            if require_recipe_button(self.gui,item)!=(x,y):raise ValueError('recipe button moved')
        yield from self.click(x,y,check_button)
        self.stage='await_recipe_placement'
        for _ in range(30):
            try:verify_grid(self.gui,self.recipe);break
            except ValueError:yield self.noop()
        verify_grid(self.gui,self.recipe)
        # Closing the book exposes the result even on narrow GUI layouts.
        if self.gui['book_open']:
            self.stage='close_book'
            yield from self.click(self.gui['book_toggle_x'],self.gui['book_toggle_y'])
        self.stage='take_verified_result'
        verify_grid(self.gui,self.recipe)
        yield from self.click(self.gui['result_x'],self.gui['result_y'],lambda:verify_grid(self.gui,self.recipe))
        self.stage='await_result_observation'
        for _ in range(20):yield self.noop()
        raise RuntimeError('result did not reach the crafting completion boundary')

    def act(self,screen):
        action=next(self.sequence)
        save_png(screen,self.output/('book-'+self.stage+'.png'))
        return action
