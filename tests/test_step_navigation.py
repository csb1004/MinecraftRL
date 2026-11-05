import copy
import math
import unittest

from minecraftrl.step_navigation import (geometry, rotate, step_contract, step_targets,
    step_teacher, validate_clearance, validate_layout, validate_step_trial,terrain_action_count)
from minecraftrl.navigation_refinement import TrialState, local_targets, validate_dataset


def fixture(rotation=0, material='minecraft:stone', edge=8,*,goal_profile=None,start_side=None,start_distance=.35):
    layout=dict(material=material, rotation=rotation, edge=edge, start_distance=start_distance, yaw_offset=0.)
    if goal_profile is not None:layout['goal_profile']=goal_profile
    if start_side is not None:layout['start_side']=start_side
    contract=step_contract(layout, [0, 0], 'a'*64)
    world=dict(kind='step_navigation', experimental_split='train', sha256='a'*64,
        chunk=[0,0], step_layout=layout, navigation_goals=contract, position=contract['initial_position'])
    config=dict(schema='navigation_trial_config_v1', goal_contract=contract['schema'],
        world_sha256=world['sha256'], split='train', mode='correction', pose_source='client',
        arrival_hold_ticks=12, student_ticks=8, points=contract['points'])
    return world,config


def pose(x=7.65,y=201.,z=3.5,rotation=0,tick=1,ground=1):
    x,z=rotate(x,z,rotation)
    return dict(client_x=x,client_y=y,client_z=z,server_x=x,server_y=y,server_z=z,
        server_available=1,client_available=1,server_tick=tick,client_tick=tick,
        server_on_ground=ground,client_on_ground=ground,screen=0,error=0,
        yaw=(-90.+rotation+180)%360-180,pitch=0.,held_key_count=0)


class StepNavigationTests(unittest.TestCase):
    def test_long_prefix_contract_preserves_terrain_goals_and_legacy_contract(self):
        world,config=fixture()
        old=copy.deepcopy(world['navigation_goals'])
        new=step_contract(world['step_layout'],world['chunk'],world['sha256'],student_prefix_actions=32)
        self.assertEqual(new['schema'],'step_navigation_goals_v2')
        self.assertEqual(new['student_prefix_actions'],32)
        self.assertEqual(new['maximum_correction_actions_per_goal'],133)
        for key in old:
            if key not in ('schema','student_prefix_actions','maximum_correction_actions_per_goal'):
                self.assertEqual(new[key],old[key])
        world['navigation_goals']=new
        config.update(goal_contract=new['schema'],student_ticks=32)
        self.assertEqual(validate_step_trial(world,config),old['points'])
        self.assertEqual(step_contract(world['step_layout'],world['chunk'],world['sha256']),old)

    def test_long_prefix_cannot_override_bound_window_or_change_legacy_schema(self):
        world,config=fixture()
        with self.assertRaises(ValueError):validate_step_trial(world,dict(config,student_ticks=32))
        new=step_contract(world['step_layout'],world['chunk'],world['sha256'],student_prefix_actions=32)
        world['navigation_goals']=new
        config.update(goal_contract=new['schema'],student_ticks=32)
        for key,value in (('student_ticks',8),('student_ticks',True),('student_ticks',32.0),
                          ('goal_contract','step_navigation_goals_v1')):
            with self.assertRaises(ValueError):validate_step_trial(world,dict(config,**{key:value}))
        for changes in (dict(schema='step_navigation_goals_v1'),dict(maximum_correction_actions_per_goal=109)):
            changed=copy.deepcopy(world);changed['navigation_goals'].update(changes)
            with self.assertRaises(ValueError):validate_step_trial(changed,config)

    def test_prefix_bounds_and_boolean_values_rejected(self):
        world,_=fixture()
        for prefix in (0,61,-1,True,32.0,'32'):
            with self.assertRaises(ValueError):
                step_contract(world['step_layout'],world['chunk'],world['sha256'],student_prefix_actions=prefix)

    def test_long_prefix_student_survives_landing_until_explicit_handoff(self):
        world,config=fixture()
        state=TrialState(config['points'],'correction',student_ticks=32,arrival_hold_ticks=12,pose_source='client')
        state.start(pose())
        for tick in range(1,32):
            state.advance(pose(tick=tick,ground=0 if tick==12 else 1),pose(tick=tick+1))
            self.assertEqual(state.owner,'student')
        state.advance(pose(tick=32),pose(tick=33))
        self.assertEqual(state.owner,'neutral')
        with self.assertRaises(ValueError):state.advance(pose(tick=33),dict(pose(tick=34),held_key_count=1))
        state.advance(pose(tick=33),pose(tick=34))
        self.assertEqual(state.owner,'teacher')
        self.assertFalse(state.receipts)

    def test_braking_goals_and_both_start_sides_are_supported_in_all_rotations(self):
        for material in ('minecraft:stone','minecraft:dirt','minecraft:oak_leaves'):
            for rotation in (0,90,180,270):
                legacy,_=fixture(rotation,material)
                original_blocks,_,_,_=geometry(legacy['step_layout'])
                for side in ('lower','upper'):
                    for distance in (.35,2.):
                        world,config=fixture(rotation,material,goal_profile='stop-before-step',
                                             start_side=side,start_distance=distance)
                        blocks,start,yaw,points=geometry(world['step_layout'])
                        self.assertEqual(blocks,original_blocks)
                        self.assertEqual(start[1],201 if side=='lower' else 202)
                        self.assertEqual([p[1] for p in points],[201,202,201,201,202,201])
                        for point in [start]+points:validate_clearance(blocks,point)
                        expected_yaw=((-90 if side=='lower' else 90)+rotation+180)%360-180
                        self.assertEqual(yaw,expected_yaw)
                        self.assertEqual(validate_step_trial(world,config),points)
                        self.assertAlmostEqual(math.hypot(points[0][0]-points[1][0],points[0][2]-points[1][2]),1.)

    def test_grounded_wrong_height_keeps_correcting_inside_old_horizontal_radius(self):
        from minecraftrl.fixture_probe import navigate
        for rotation in (0,90,180,270):
            world,config=fixture(rotation,goal_profile='stop-before-step',start_side='upper')
            target=config['points'][0]
            g=pose(x=7.71,y=202,rotation=rotation)
            g['yaw']=(90+rotation+180)%360-180
            self.assertEqual(navigate(g,target[0],target[2],pose_source='client'),({},True))
            changes,arrived=step_teacher(g,target,world)
            self.assertFalse(arrived);self.assertEqual(changes['forward'],1);self.assertEqual(changes['jump'],0)
            self.assertEqual(step_teacher(pose(x=7.5,y=201,rotation=rotation),target,world),({},True))
            self.assertEqual(step_teacher(pose(x=7.5,y=201.5,ground=0,rotation=rotation),target,world),({},False))

    def test_invalid_braking_profile_and_arrival_radius_are_rejected(self):
        from minecraftrl.fixture_probe import navigate
        world,_=fixture()
        for changes in (dict(goal_profile='unknown'),dict(start_side='upper'),
                        dict(goal_profile='stop-before-step',start_side='unknown')):
            with self.assertRaises(ValueError):validate_layout(dict(world['step_layout'],**changes))
        for radius in (0,-.1,.3,float('nan'),float('inf'),True,'.05'):
            with self.assertRaises(ValueError):navigate(pose(),8.,3.5,arrival_radius=radius)

    def test_released_json_arrays_are_not_pressed_and_both_buttons_are_validated(self):
        rows=[dict(action=dict(attack=attack,use=use)) for attack,use in
              (([0],[0]),(0.,0),([[0]],False),([1],[0]),(0,[1]),([1],[1]))]
        self.assertEqual(terrain_action_count(rows),3)
        for value in ([],[0,1],2,float('nan'),'0'):
            with self.assertRaises(ValueError):terrain_action_count([dict(action=dict(attack=[1],use=value))])

    def test_all_materials_edges_rotations_have_supported_clear_targets(self):
        for material in ('minecraft:stone','minecraft:dirt','minecraft:oak_leaves'):
            for edge in (7,8,9):
                for rotation in (0,90,180,270):
                    world,config=fixture(rotation,material,edge)
                    blocks,start,yaw,points=geometry(world['step_layout'])
                    for p in [start]+points:validate_clearance(blocks,p)
                    self.assertEqual([p[1] for p in points],[202.,202.,201.,202.,202.,201.])
                    self.assertEqual(validate_step_trial(world,config),local_targets(world,config['points']))
                    # Rotation of voxel centers must match continuous coordinates exactly.
                    for x,z in ((1,1),(8,4),(14,14)):
                        bx,bz=rotate(x,z,rotation,block=True)
                        self.assertEqual(rotate(x+.5,z+.5,rotation),(bx+.5,bz+.5))

    def test_invalid_layouts_and_mixed_or_unbound_worlds_rejected(self):
        world,config=fixture()
        for key,value in (('rotation',True),('rotation',45),('edge',True),('edge',10),
                ('start_distance',.2),('start_distance',float('nan')),('yaw_offset',91),
                ('material','minecraft:sand')):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                validate_layout(dict(world['step_layout'],**{key:value}))
        for key,value in (('kind','natural_navigation'),('experimental_split','final'),
                          ('sha256','b'*64),('position',[0,201,0])):
            with self.assertRaises(ValueError):step_targets(dict(world,**{key:value}),config['points'])
        changed=copy.deepcopy(config['points']);changed[0][1]=201.
        with self.assertRaises(ValueError):step_targets(world,changed)
        for key,value in (('mode','ppo'),('pose_source','server'),('arrival_hold_ticks',1),
                          ('world_sha256','b'*64),('split','development')):
            with self.assertRaises(ValueError):validate_step_trial(world,dict(config,**{key:value}))

    def test_rotated_teacher_jumps_only_at_lower_grounded_upward_boundary(self):
        for rotation in (0,90,180,270):
            world,config=fixture(rotation)
            goal=config['points'][0]
            action,_=step_teacher(pose(rotation=rotation),goal,world)
            self.assertEqual((action['forward'],action['jump']),(1,1))
            for g in (pose(x=6.,rotation=rotation),pose(y=201.5,ground=0,rotation=rotation),
                      dict(pose(rotation=rotation),yaw=(rotation+180)%360)):
                self.assertEqual(step_teacher(g,goal,world)[0].get('jump',0),0)
            # Once on top, travelling to another upper goal must not keep jumping.
            g=pose(x=8.5,y=202,rotation=rotation)
            self.assertEqual(step_teacher(g,config['points'][1],world)[0]['jump'],0)
            self.assertEqual(step_teacher(g,config['points'][2],world)[0]['jump'],0)

    def test_teacher_stops_at_actual_height_and_rejects_missing_pose(self):
        world,config=fixture();goal=config['points'][0]
        self.assertEqual(step_teacher(pose(x=8.5,y=202),goal,world),({},True))
        self.assertEqual(step_teacher(pose(x=8.5,y=202.5,ground=0),goal,world),({},False))
        for key,value in (('client_available',0),('client_on_ground',None),('client_on_ground',True),
                          ('screen',1),('error',1),('client_x',float('nan')),('client_x',.5),('yaw',float('inf'))):
            with self.assertRaises(ValueError):step_teacher(dict(pose(),**{key:value}),goal,world)
        with self.assertRaises(ValueError):step_teacher(pose(),goal,world,pose_source='server')

    def test_evaluation_never_hands_control_to_teacher(self):
        world,config=fixture();config['mode']='evaluate';config.pop('student_ticks')
        validate_step_trial(world,config)
        state=TrialState(config['points'],'evaluate',arrival_hold_ticks=12,pose_source='client')
        state.start(pose())
        for tick in range(1,105):
            state.advance(pose(tick=tick),pose(tick=tick+1))
            self.assertEqual(state.owner,'student')
        self.assertFalse(state.receipts[0]['student_success'])

    def test_long_prefix_evaluation_keeps_hundred_action_limit_without_teacher(self):
        world,config=fixture()
        world['navigation_goals']=step_contract(world['step_layout'],world['chunk'],world['sha256'],student_prefix_actions=32)
        config.update(mode='evaluate',goal_contract='step_navigation_goals_v2')
        config.pop('student_ticks')
        validate_step_trial(world,config)
        state=TrialState(config['points'],'evaluate',arrival_hold_ticks=12,pose_source='client')
        state.start(pose())
        for tick in range(1,101):
            state.advance(pose(tick=tick),pose(tick=tick+1))
            self.assertEqual(state.owner,'student')
        self.assertEqual(state.receipts[0]['actions'],100)
        self.assertEqual(state.receipts[0]['outcome'],'failed')
        data=dict(schema='navigation_trials_v1',split='train',eligible=True,world=world,config=config,
                  rows=[dict(owner='student')],receipts=[dict(outcome='completed')]*6)
        with self.assertRaisesRegex(ValueError,'correction BC'):validate_dataset(data,'bc')

    def test_teacher_completion_is_not_learned_success_and_failed_data_rejected(self):
        world,config=fixture()
        state=TrialState(config['points'],'correction',student_ticks=8,arrival_hold_ticks=12,pose_source='client')
        state.start(pose())
        for tick in range(1,9):state.advance(pose(tick=tick),pose(tick=tick+1))
        self.assertEqual(state.owner,'neutral')
        state.advance(pose(tick=9),pose(tick=10))
        self.assertEqual(state.owner,'teacher')
        for tick in range(10,22):state.advance(pose(x=8.5,y=202,tick=tick),pose(x=8.5,y=202,tick=tick+1))
        self.assertEqual(state.receipts[0]['outcome'],'completed')
        self.assertFalse(state.receipts[0]['student_success'])
        data=dict(schema='navigation_trials_v1',split='train',eligible=True,world=world,config=config,
                  rows=[dict(owner='teacher')],receipts=state.receipts)
        with self.assertRaisesRegex(ValueError,'all six'):validate_dataset(data,'bc')

    def test_flat_goal_augmentation_cannot_silently_label_step_world(self):
        from minecraftrl.navigation_goal_queries import (goal_variation_requests,
            dense_goal_variation_requests,approach_goal_variation_requests)
        world,_=fixture()
        for method in (goal_variation_requests,dense_goal_variation_requests,approach_goal_variation_requests):
            with self.assertRaisesRegex(ValueError,'cannot label step'):
                method(dict(world=world),[])


if __name__=='__main__':unittest.main()
