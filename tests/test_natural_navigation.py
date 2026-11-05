import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from minecraftrl.natural_navigation import natural_targets, validate_natural_trial, attempted_goal_outcomes, validate_transfer_admission
from minecraftrl.navigation_refinement import local_targets, validate_dataset
from minecraftrl.navigation_supervision import expert_requests
from minecraftrl.navigation_trial import NavigationTrial


def world_and_config():
    points = [[2.5,65.,.5],[4.5,66.,.5],[2.5,65.,.5],[.5,65.,.5],[.5,66.,2.5],[.5,65.,.5]]
    contract = dict(schema='natural_navigation_goals_v1',coordinates='absolute_xyz',points=points,
        initial_position=[.5,65.,.5],source_archive_sha256='a'*64,goal_count=6,horizontal_radius=8,
        vertical_radius=8,actions_per_goal=100,privileged_targets=True,terrain_modified=False,
        inventory_modified=False,route_given_to_policy=False,geometry_only=True,learned_route_verified=False)
    world = dict(kind='natural_navigation',experimental_split='train',sha256='a'*64,
                 position=[.5,65.,.5],navigation_goals=contract)
    config = dict(schema='navigation_trial_config_v2',goal_contract='natural_navigation_goals_v1',
        mode='evaluate',split='train',world_sha256='a'*64,points=points,
        pose_source='client',arrival_hold_ticks=12,action_selection='delivered_mode')
    return world,config


class NaturalContractTests(unittest.TestCase):
    def test_goal_height_is_preserved_and_input_is_not_mutated(self):
        world,config = world_and_config()
        before = copy.deepcopy((world,config))
        result = validate_natural_trial(world,config)
        self.assertEqual(result[1], [4.5,66.,.5])
        result[0][1] = 201
        self.assertEqual((world,config),before)

    def test_teacher_ppo_and_changed_arrival_contract_are_rejected(self):
        world,config = world_and_config()
        for field,value in [('schema','navigation_trial_config_v1'),('goal_contract',None),
                            ('mode','correction'),('mode','ppo'),('action_selection','sample'),
                            ('pose_source','server'),('arrival_hold_ticks',1),('student_ticks',8),
                            ('split','development'),('world_sha256','b'*64)]:
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                validate_natural_trial(world,dict(config,**{field:value}))

    def test_unfrozen_or_nonfinite_goal_cannot_launch(self):
        for value in (True,float('nan'),float('inf'),100):
            world,config = world_and_config()
            world['navigation_goals']['points'][0][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                natural_targets(world,config['points'])
        world,config = world_and_config()
        altered = copy.deepcopy(config['points']);altered[0][0] += .1
        with self.assertRaises(ValueError):natural_targets(world,altered)

    def test_world_changes_and_reserved_split_are_rejected(self):
        for field,value in [('terrain_modified',True),('inventory_modified',True),
                            ('route_given_to_policy',True),('source_archive_sha256','b'*64)]:
            world,config = world_and_config()
            world['navigation_goals'][field] = value
            with self.assertRaises(ValueError):natural_targets(world,config['points'])
        world,config = world_and_config();world['experimental_split'] = 'final'
        with self.assertRaises(ValueError):natural_targets(world,config['points'])

    def test_early_death_does_not_turn_unattempted_goals_into_failures(self):
        summary = dict(status='environment_done',navigation_finished=False,
                       navigation_receipts=[dict(index=0,outcome='completed',student_success=True)])
        rows = [dict(index=0),dict(index=0),dict(index=1)]
        result = attempted_goal_outcomes(summary,rows,6)
        self.assertEqual((result['arrived'],result['interrupted'],result['not_attempted']), (1,1,4))
        self.assertEqual(result['budget_exhausted'],0)

    def test_root_goal_completion_is_not_navigation_arrival(self):
        summary=dict(status='target_reached',navigation_finished=False,root_inventory_success=True,navigation_receipts=[])
        result=attempted_goal_outcomes(summary,[dict(index=0)],6)
        self.assertEqual((result['arrived'],result['interrupted'],result['not_attempted']),(0,1,5))

    def test_skipped_attempt_or_missing_receipt_is_rejected(self):
        summary = dict(status='target_reached',navigation_finished=True,navigation_receipts=[])
        for rows in ([dict(index=1)],[dict(index=0)]):
            with self.assertRaises(ValueError):attempted_goal_outcomes(summary,rows,6)

    def test_inconsistent_success_cannot_inflate_arrivals(self):
        summary = dict(status='environment_done',navigation_finished=False,
                       navigation_receipts=[dict(index=0,outcome='failed',student_success=True)])
        with self.assertRaises(ValueError):attempted_goal_outcomes(summary,[dict(index=0)],6)



class NaturalTransferAdmissionTests(unittest.TestCase):
    def evidence(self):
        return dict(pipeline=dict(completed=True,active=None,error=None),
            mode_readiness=dict(completed=True,train_calibration_ready=True,counts=dict(near=4,far=4),actor_sha256='b'*64),
            calibration=dict(completed=True,passed=True,arrivals=6),
            calibration_summary=dict(actor_sha256='b'*64,execution_valid=True,worker_exit_code=0,navigation_finished=True,learned_arrivals=6),
            flat_selection=dict(plan=dict(baseline_sha256='a'*64,candidate_sha256='b'*64)),
            flat_comparison=dict(completed=True,gate_passed=False,pairs=[dict(case=label,initial_conditions_equal=True,
                reports=dict(baseline=dict(arrivals=6,actions=250),candidate=dict(arrivals=6,actions=270))) for label in ('a','b')]))

    def test_equal_arrivals_may_transfer_without_claiming_flat_speed_improvement(self):
        result=validate_transfer_admission(self.evidence(),'a'*64,'b'*64)
        self.assertTrue(result['verified'])
        self.assertEqual(result['flat_arrivals'],12)
        self.assertFalse(result['flat_improvement_gate_passed'])
        self.assertFalse(result['speed_improvement_required'])

    def test_incomplete_unbound_or_regressed_candidate_cannot_launch_natural_games(self):
        cases=[]
        evidence=self.evidence();evidence['pipeline']['completed']=False;cases.append(evidence)
        evidence=self.evidence();evidence['mode_readiness']['counts']['far']=3;cases.append(evidence)
        evidence=self.evidence();evidence['calibration_summary']['learned_arrivals']=4;cases.append(evidence)
        evidence=self.evidence();evidence['flat_comparison']['pairs'][0]['initial_conditions_equal']=False;cases.append(evidence)
        evidence=self.evidence();evidence['flat_comparison']['pairs'][0]['reports']['candidate']['arrivals']=5;cases.append(evidence)
        evidence=self.evidence();evidence['flat_selection']['plan']['candidate_sha256']='c'*64;cases.append(evidence)
        for index,evidence in enumerate(cases):
            with self.subTest(case=index),self.assertRaises(ValueError):validate_transfer_admission(evidence,'a'*64,'b'*64)


class NaturalRuntimeIntegrationTests(unittest.TestCase):
    def test_root_acceptance_stops_natural_control_before_next_action(self):
        from minecraftrl.navigation_trial import run_navigation_trial
        for kind,root_complete,finished,expected in (
                ('natural_navigation',True,False,True),
                ('natural_navigation',False,False,False),
                ('natural_navigation',False,True,True),
                ('recovery',True,False,False)):
            policy=SimpleNamespace(state=SimpleNamespace(finished=finished),student=SimpleNamespace(dispose=Mock()))
            wrapper=SimpleNamespace(reward=SimpleNamespace(completed=root_complete))
            def inspect_callback(*args,**kwargs):
                self.assertIs(kwargs['completion_check']({}),expected)
                raise RuntimeError('callback inspected before game launch')
            with self.subTest(kind=kind,root_complete=root_complete,finished=finished):
                with patch('minecraftrl.navigation_trial.RLEnvironment',return_value=wrapper), \
                     patch('minecraftrl.navigation_trial.run_episode',side_effect=inspect_callback):
                    with self.assertRaisesRegex(RuntimeError,'callback inspected'):
                        run_navigation_trial(policy,None,Path('.'),world=dict(kind=kind),gate=Mock(),stop_file=None)
                policy.student.dispose.assert_called_once()

    def test_natural_goals_use_absolute_height_without_fixture_chunk(self):
        world,config=world_and_config()
        self.assertNotIn('chunk',world)
        self.assertEqual(local_targets(world,config['points']),config['points'])
        self.assertEqual(local_targets(world,config['points'])[1][1],66.)

    def test_existing_flat_fixture_coordinates_are_unchanged(self):
        world=dict(kind='recovery',experimental_split='train',chunk=[-2,3],curriculum=dict(mirror_z=False))
        self.assertEqual(local_targets(world,[[2.5,4.5]]),[[-29.5,201.,52.5]])
        world['curriculum']['mirror_z']=True
        self.assertEqual(local_targets(world,[[2.5,4.5]]),[[-29.5,201.,59.5]])

    def test_natural_data_cannot_enter_bc_ppo_or_flat_expert(self):
        world,config=world_and_config()
        data=dict(schema='navigation_trials_v1',split='train',eligible=True,world=world,
                  config=config,rows=[dict(owner='student')])
        for method in ('bc','ppo'):
            with self.subTest(method=method),self.assertRaisesRegex(ValueError,'evaluation only'):
                validate_dataset(data,method)
        with self.assertRaisesRegex(ValueError,'flat expert'):expert_requests(data,[])

    def test_natural_teacher_is_rejected_before_loading_any_policy(self):
        world,config=world_and_config();config['mode']='correction'
        with self.assertRaisesRegex(ValueError,'learned-only'):
            NavigationTrial(None,None,world,None,config,None)

    def test_identical_pixels_cannot_hide_changed_health_inventory_or_world_time(self):
        from minecraftrl.common_navigation import initial_navigation_conditions, POLICY_FILES
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            base=Path(folder);run=base/'run';run.mkdir();(run/'frames').mkdir()
            (run/'frames/000000.png').write_bytes(b'unit-test-identical-pixels')
            sources=base/'source/src/minecraftrl';sources.mkdir(parents=True)
            for name in set(POLICY_FILES)|{'natural_navigation.py','navigation_arrival.py',
                    'navigation_observation.py','workshop_monitor.py','navigation_action_likelihood.py'}:
                (sources/name).write_text('fixed source',encoding='utf-8')
            _,config=world_and_config()
            (base/'protocol.json').write_text(json.dumps(config),encoding='utf-8')
            condition={key:'fixed' for key in ('initial_recurrent_sha256','embedding_sha256','prompt','cond_scale',
                'backbone_tail_sha256','backbone_temporal_sha256','recurrent_commit_mode','reward_config')}
            (run/'rl-condition.json').write_text(json.dumps(condition),encoding='utf-8')
            (run/'runtime-jar-cache.json').write_text(json.dumps(dict(sha256='fixed',launcher_sha256='fixed')),encoding='utf-8')
            state=dict(server_x=.5,server_y=65.,server_z=.5,yaw=0.,pitch=0.,
                client_x=.5,client_y=65.,client_z=.5,client_tick=100,server_tick=100,world_time=100,
                server_health=20.,server_food=20.,server_alive=1,server_on_ground=1,held_key_count=0,server_inventory={})
            def conditions(value):
                (run/'initial-state.json').write_text(json.dumps(dict(server=value)),encoding='utf-8')
                return initial_navigation_conditions(run)
            original=conditions(state)
            for key,value in [('server_health',19.),('server_food',18.),('world_time',101),('server_inventory',{'oak_log':1})]:
                changed=copy.deepcopy(state);changed[key]=value
                with self.subTest(key=key):self.assertNotEqual(original,conditions(changed))
            (sources/'startup_control.py').write_text('fixed startup source',encoding='utf-8')
            config['startup_control']={'restore_speed':3,'resume_client_tick':80}
            (base/'protocol.json').write_text(json.dumps(config),encoding='utf-8')
            controlled=conditions(state)
            self.assertNotEqual(original,controlled)
            self.assertIn('startup_control.py',controlled['policy_sources'])
            config['startup_control']['restore_speed']=0
            (base/'protocol.json').write_text(json.dumps(config),encoding='utf-8')
            self.assertNotEqual(controlled,conditions(state))

