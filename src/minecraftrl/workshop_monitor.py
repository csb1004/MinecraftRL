"""Typed, read-only GUI and local placement telemetry for assisted workshop probes."""
def workshop_survival(*,terminate_on_death=False,**kwargs):
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    from minerl.herobraine.hero.handlers.translation import TranslationHandler
    from minerl.herobraine.hero import spaces,mc
    import numpy as np
    number=lambda:spaces.Box(low=-1e12,high=1e12,shape=(),dtype=np.float64)
    item=lambda:spaces.Enum(*mc.ALL_ITEMS)
    fields=('screen','width','error','book_open','mouse_x','mouse_y','book_x','book_y','cursor_count',
            'server_available','server_cursor','next_x','next_y','previous_x','previous_y','search_x','search_y',
            'page','pages','search_tab','yaw','pitch','eye_x','eye_y','eye_z','hit_x','hit_y','hit_z','hit_face',
            'candidate_x','candidate_y','candidate_z','table_x','table_y','table_z','hit_table','table_count','candidate',
            'server_tick','world_time','server_x','server_y','server_z','render_ready',
            'server_health','server_alive','held_key_count',
            'wood_kind','wood_distance','wood_reachable','wood_target_x','wood_target_y','wood_target_z',
            'wood_server_mining','wood_mining_x','wood_mining_y','wood_mining_z',
            'wood_v3_kind','wood_v3_distance','wood_v3_reachable','wood_v3_target_x','wood_v3_target_y','wood_v3_target_z')
    recipe_ids=['oak_planks','birch_planks','spruce_planks','jungle_planks','acacia_planks','dark_oak_planks',
                'stick','crafting_table','wooden_pickaxe','wooden_axe','oak_slab']
    schema={key:number() for key in fields}
    schema.update(cursor_item=item(),selected_item=item(),server_inventory=spaces.Dict({key:number() for key in mc.ALL_ITEMS}),
        slots=spaces.Tuple([spaces.Dict(dict(item=item(),**{k:number() for k in ('count','x','y','inventory_index')})) for _ in range(46)]),
        recipes=spaces.Dict({key:spaces.Dict({k:number() for k in ('known','server_known','x','y','single','craftable')}) for key in recipe_ids}))
    class Monitor(TranslationHandler):
        def __init__(self):super().__init__(spaces.Dict(schema))
        def to_string(self):return 'workshop'
        def xml_template(self):return ''
        def from_hero(self,value):return value['workshop']
        def from_universal(self,value):return self.from_hero(value)
    class Survival(HumanSurvival):
        def create_monitors(self):return super().create_monitors()+[Monitor()]
        def create_agent_start(self):
            handlers=super().create_agent_start()
            if terminate_on_death:
                from minerl.herobraine.hero.handlers import DoneOnDeath
                handlers.append(DoneOnDeath())
            return handlers
    return Survival(**kwargs)
