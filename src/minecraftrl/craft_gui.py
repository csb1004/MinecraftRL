"""Optional read-only monitor for the isolated crafting experiment."""
def monitored_survival(**kwargs):
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    from minerl.herobraine.hero.handlers.translation import TranslationHandler
    from minerl.herobraine.hero import spaces
    import numpy as np
    class Monitor(TranslationHandler):
        def __init__(self):
            super().__init__(spaces.Dict({key:spaces.Box(low=-1,high=100000,shape=(),dtype=np.float64)
                for key in ('inventory_screen','cursor_count','cursor_planks','grid_planks',
                            'gui_width','gui_height','result_x','result_y',
                            'empty_slot','mouse_x','mouse_y','target_x','target_y','server_available',
                            'server_inventory_planks','server_cursor_count','server_crafted_planks',
                            'cursor_sticks','grid_sticks','server_inventory_sticks','server_crafted_sticks',
                            'cursor_table','grid_table','server_inventory_table','server_crafted_table')+
                ('book_error','book_open','book_toggle_x','book_toggle_y','preview_item','preview_count',
                 'oak_planks_id','stick_id','crafting_table_id')+
                tuple('input_'+str(i)+'_'+field for i in range(1,5) for field in ('item','count'))+
                tuple(prefix+item+suffix for item in ('oak_planks','stick','crafting_table','wooden_pickaxe','oak_slab')
                      for prefix,suffix in (('known_',''),('server_known_',''),('book_','_x'),('book_','_y'),
                                            ('book_','_single'),('book_','_craftable')))}))
        def to_string(self): return 'craft_gui'
        def xml_template(self): return ''
        def from_hero(self,value): return value['craft_gui']
        def from_universal(self,value): return self.from_hero(value)
    class Survival(HumanSurvival):
        def create_monitors(self): return super().create_monitors()+[Monitor()]
    return Survival(**kwargs)
