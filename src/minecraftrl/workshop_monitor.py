"""Typed, read-only GUI and local placement telemetry for assisted workshop probes."""
DEMO_NUMBERS=('demo_available','demo_destroys','demo_last_destroy_tick','demo_last_destroy_removed',
              'demo_last_destroy_harvestable','demo_last_destroy_x','demo_last_destroy_y','demo_last_destroy_z',
              'demo_mining_x','demo_mining_y','demo_mining_z','demo_fixture_state','demo_fixture_blocks','demo_fixture_sha52')


def workshop_survival(*,terminate_on_death=False,demo=False,**kwargs):
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    from minerl.herobraine.hero.handlers.translation import TranslationHandler
    from minerl.herobraine.hero import spaces,mc
    import numpy as np
    number=lambda:spaces.Box(low=-1e12,high=1e12,shape=(),dtype=np.float64)
    item=lambda:spaces.Enum(*mc.ALL_ITEMS)
    fields=('screen','width','error','book_open','mouse_x','mouse_y','book_x','book_y','cursor_count',
            'server_available','server_cursor','next_x','next_y','previous_x','previous_y','search_x','search_y',
            'page','pages','search_tab','overlay_open','yaw','pitch','eye_x','eye_y','eye_z','hit_x','hit_y','hit_z','hit_face',
            'candidate_x','candidate_y','candidate_z','table_x','table_y','table_z','hit_table','table_count','candidate',
            'server_tick','world_time','server_x','server_y','server_z','render_ready',
            'client_available','client_tick','client_x','client_y','client_z','client_on_ground',
            'server_health','server_food','server_alive','held_key_count','server_harvests','server_axe_logs','server_pick_stone',
            'server_log_harvests','server_tool_damage',
            'server_on_ground','server_fall_distance','server_inside_opaque_block','server_eye_y',
            'threat_valid','threat_scan_radius','threat_count','threat_distance',
            'pursuit_valid','pursuit_scan_radius','pursuit_count','pursuit_distance',
            'wood_kind','wood_distance','wood_reachable','wood_target_x','wood_target_y','wood_target_z',
            'wood_server_mining','wood_mining_x','wood_mining_y','wood_mining_z',
            'wood_v3_kind','wood_v3_distance','wood_v3_reachable','wood_v3_target_x','wood_v3_target_y','wood_v3_target_z')
    recipe_ids=['oak_planks','birch_planks','spruce_planks','jungle_planks','acacia_planks','dark_oak_planks',
                'stick','crafting_table','wooden_pickaxe','wooden_axe','stone_axe','oak_slab']
    if demo:recipe_ids.insert(recipe_ids.index('stone_axe')+1,'stone_pickaxe')
    schema={key:number() for key in fields}
    schema.update(cursor_item=item(),selected_item=item(),server_selected_item=item(),
        server_recent_damage_type=spaces.Enum('none','inWall','drown','fall','starve','other'),
        server_last_harvest_tool=item(),server_inventory=spaces.Dict({key:number() for key in mc.ALL_ITEMS}),
        slots=spaces.Tuple([spaces.Dict(dict(item=item(),**{k:number() for k in ('count','x','y','inventory_index')})) for _ in range(46)]),
        recipes=spaces.Dict({key:spaces.Dict({k:number() for k in ('known','server_known','x','y','single','craftable','group_x','group_y','overlay')}) for key in recipe_ids}))
    block=lambda:spaces.Enum(*mc.ALL_ITEMS,'other')
    demo_schema={key:number() for key in DEMO_NUMBERS}
    demo_schema.update(demo_last_destroyed_block=block(),demo_last_destroy_tool=item(),
                       demo_mining_block=block(),demo_hit_block=block())
    class Monitor(TranslationHandler):
        def __init__(self,name='workshop',fields=schema):
            super().__init__(spaces.Dict(fields));self.name=name
        def to_string(self):return self.name
        def xml_template(self):return ''
        def from_hero(self,value):return value[self.name]
        def from_universal(self,value):return self.from_hero(value)
    class Survival(HumanSurvival):
        def create_monitors(self):
            return super().create_monitors()+[Monitor()]+([Monitor('demo',demo_schema)] if demo else [])
        def create_agent_start(self):
            handlers=super().create_agent_start()
            if terminate_on_death:
                from minerl.herobraine.hero.handlers import DoneOnDeath
                handlers.append(DoneOnDeath())
            return handlers
    return Survival(**kwargs)
