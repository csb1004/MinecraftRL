import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock,patch
import copy

from minecraftrl.tool_collection import PHASES,criteria,make_ledger,ToolCollectionEnvironment,workstation_ready,validate_condition_phase
from minecraftrl.runtime import StopGate
from minecraftrl.world_save_patch import patch_harvest_observation


class ToolCollectionTests(unittest.TestCase):
    def recovery_bundle(self):
        return dict(world_id='w',generation='lost',archive='world.zip',
            observation={'inventory':{'oak_log':103,'wooden_pickaxe':1}},
            controller_state={'tool_mission':dict(schema=1,world_id='w',stages={
                'axe_wood':dict(status='active',baseline=52,target=64,verified_tool_harvests=40,
                    retained_gain=51,last_halt_reason='required_tool_lost')})})

    def test_tool_recovery_requires_loss_history_and_actual_nearby_saved_tool(self):
        b=self.recovery_bundle();before=copy.deepcopy(b)
        with patch('minecraftrl.tool_collection.saved_drop_budget',return_value={'inventory_budget':{'stone_axe':1}}):
            ledger=make_ledger(b,'recover_axe')
        self.assertEqual(b,before);self.assertEqual(ledger['stages']['axe_wood'],b['controller_state']['tool_mission']['stages']['axe_wood'])
        self.assertEqual(ledger['stages']['recover_axe']['baseline'],0)
        with patch('minecraftrl.tool_collection.saved_drop_budget',return_value={'inventory_budget':{}}):
            with self.assertRaises(ValueError):make_ledger(b,'recover_axe')
        b['controller_state']['tool_mission']['stages']['axe_wood']['last_halt_reason']=None
        with self.assertRaises(ValueError):make_ledger(b,'recover_axe')
        with self.assertRaises(ValueError):validate_condition_phase('recover_axe',None)
        validate_condition_phase('recover_axe','condition.json')

    def test_recovery_passes_through_neural_actions_and_stops_at_actual_inventory(self):
        b=self.recovery_bundle();raw=Mock();gate=StopGate()
        with patch('minecraftrl.tool_collection.saved_drop_budget',return_value={'inventory_budget':{'stone_axe':1}}):
            wrapped=ToolCollectionEnvironment(raw,b,Mock(),'recover_axe',gate)
        wrapped.initial_health=8.;wrapped.minimum_health=8.
        g=dict(server_available=1,error=0,server_alive=1,server_health=8,server_selected_item='air',
            server_inventory={'oak_log':103},server_axe_logs=0)
        raw.step.return_value=({'inventory':{},'life_stats':{'food':5}},0,False,{'workshop':g})
        action={'forward':1};wrapped.step(action)
        raw.step.assert_called_once_with(action);self.assertFalse(gate.requested.is_set());self.assertFalse(wrapped.completed)
        g['server_inventory']['stone_axe']=1;wrapped.step({'forward':0})
        self.assertTrue(wrapped.completed);self.assertEqual(wrapped.ledger['stages']['axe_wood']['verified_tool_harvests'],40)
        with self.assertRaises(RuntimeError):wrapped.step({'forward':1})

    def test_recovery_does_not_credit_startup_pickup_as_neural_action(self):
        b=self.recovery_bundle();raw=Mock()
        with patch('minecraftrl.tool_collection.saved_drop_budget',return_value={'inventory_budget':{'stone_axe':1}}):
            wrapped=ToolCollectionEnvironment(raw,b,Mock(),'recover_axe',StopGate())
        raw.reset.return_value={'inventory':{'oak_log':103,'stone_axe':1},'life_stats':{'life':8,'food':5}}
        with patch('minecraftrl.tool_collection.validate_restored_observation',return_value={'inventory_match':False,'startup_pickup':True}):
            wrapped.reset()
        self.assertTrue(wrapped.completed);self.assertEqual(wrapped.completion_source,'restore_pickup')
        raw.step.assert_not_called()

    def test_last_log_pickup_on_restore_completes_only_with_verified_axe_harvests(self):
        for harvests,completed in ((64,True),(63,False)):
            b=dict(world_id='w',generation='g',observation={'inventory':{'oak_log':115,'stone_axe':1}},
                controller_state={'tool_mission':dict(schema=1,world_id='w',stages={
                    'axe_wood':dict(status='active',baseline=52,target=64,verified_tool_harvests=harvests,retained_gain=63)})})
            before=copy.deepcopy(b);raw=Mock()
            wrapped=ToolCollectionEnvironment(raw,b,Mock(),'axe_wood',StopGate())
            raw.reset.return_value={'inventory':{'oak_log':116,'stone_axe':1},'life_stats':{'life':8,'food':5}}
            with patch('minecraftrl.tool_collection.validate_restored_observation',return_value={'startup_pickup':True}):
                wrapped.reset()
            self.assertEqual(wrapped.completed,completed);self.assertEqual(wrapped.stage['retained_gain'],64)
            self.assertEqual(wrapped.completion_source,'restore_pickup' if completed else None)
            self.assertEqual(b,before);raw.step.assert_not_called()

    def test_already_complete_restore_needs_no_new_action_health_budget_but_must_be_alive(self):
        for life,alive in ((1.5,True),(0,False),(1.5,False)):
            b=dict(world_id='w',generation='g',observation={'inventory':{'oak_log':115,'stone_axe':1}},
                controller_state={'tool_mission':dict(schema=1,world_id='w',stages={
                    'axe_wood':dict(status='active',baseline=52,target=64,verified_tool_harvests=64)})})
            raw=Mock();wrapped=ToolCollectionEnvironment(raw,b,Mock(),'axe_wood',StopGate())
            raw.reset.return_value={'inventory':{'oak_log':116,'stone_axe':1},'life_stats':{'life':life,'food':0,'is_alive':alive}}
            with patch('minecraftrl.tool_collection.validate_restored_observation',return_value={'startup_pickup':True}):
                if alive:wrapped.reset();self.assertTrue(wrapped.completed)
                else:
                    with self.assertRaises(ValueError):wrapped.reset()
            raw.step.assert_not_called()

    def test_explicit_axe_condition_does_not_enable_other_unvalidated_phases(self):
        for phase,path in [('axe_wood',None),('axe_wood','condition.json'),('cobblestone',None),('workstation_access','condition.json')]:
            validate_condition_phase(phase,path)
        for phase,path in [('workstation_access',None),('cobblestone','condition.json'),('unknown',None)]:
            with self.assertRaises(ValueError):validate_condition_phase(phase,path)

    def test_zero_food_halts_before_waiting_for_health_loss(self):
        wrapped,raw,gate,gui=self.environment();gui['server_pick_stone']=0
        raw.step.return_value[0]['life_stats']['food']=0
        wrapped.step({'forward':1})
        self.assertEqual(wrapped.halt_reason,'tool_collection_food_guard')
        self.assertTrue(gate.requested.is_set());self.assertEqual(wrapped.minimum_health,8)

    def test_starving_restore_cannot_start_collection(self):
        wrapped,raw,gate,gui=self.environment()
        raw.reset.return_value={'inventory':{'wooden_pickaxe':1},'life_stats':{'life':8,'food':0}}
        with patch('minecraftrl.tool_collection.validate_restored_observation',return_value={'inventory_match':True}):
            with self.assertRaisesRegex(ValueError,'food'):wrapped.reset()
        raw.step.assert_not_called()

    def test_explicit_fed_low_health_profile_keeps_stricter_damage_and_threat_limits(self):
        for danger in ('none','one_damage','health_floor','pursuit','missing_evidence','missing_count','food_empty'):
            wrapped,raw,gate,gui=self.environment()
            wrapped=ToolCollectionEnvironment(raw,self.bundle(),Mock(),'cobblestone',gate,health_profile='fed_low_health')
            raw.reset.return_value={'inventory':{'wooden_pickaxe':1},'life_stats':{'life':3.166666,'food':4}}
            with patch('minecraftrl.tool_collection.validate_restored_observation',return_value={'inventory_match':True}):wrapped.reset()
            gui.update(server_health=3.166666,server_pick_stone=0,pursuit_valid=1,pursuit_count=0)
            if danger=='one_damage':gui['server_health']=2.166666030883789
            if danger=='health_floor':gui['server_health']=2
            if danger=='pursuit':gui['pursuit_count']=1
            if danger=='missing_evidence':gui.pop('pursuit_valid')
            if danger=='missing_count':gui.pop('pursuit_count')
            if danger=='food_empty':raw.step.return_value[0]['life_stats']['food']=0
            wrapped.step({'forward':1})
            self.assertEqual(gate.requested.is_set(),danger!='none')
            raw.step.assert_called_once_with({'forward':1})

    def test_low_health_profile_does_not_allow_starving_or_two_hp_entry(self):
        for life,food in ((2,4),(3,3),(3,0),(float('nan'),4),(3,float('nan'))):
            raw=Mock();wrapped=ToolCollectionEnvironment(raw,self.bundle(),Mock(),'cobblestone',StopGate(),health_profile='fed_low_health')
            raw.reset.return_value={'inventory':{'wooden_pickaxe':1},'life_stats':{'life':life,'food':food}}
            with patch('minecraftrl.tool_collection.validate_restored_observation',return_value={'inventory_match':True}):
                with self.assertRaises(ValueError):wrapped.reset()
        with self.assertRaises(ValueError):ToolCollectionEnvironment(Mock(),self.bundle(),Mock(),'cobblestone',StopGate(),health_profile='unknown')

    def test_standard_profile_still_rejects_low_health_even_after_eating(self):
        wrapped,raw,gate,gui=self.environment()
        raw.reset.return_value={'inventory':{'wooden_pickaxe':1},'life_stats':{'life':3.166666,'food':4}}
        with patch('minecraftrl.tool_collection.validate_restored_observation',return_value={'inventory_match':True}):
            with self.assertRaisesRegex(ValueError,'health'):wrapped.reset()
        raw.step.assert_not_called()

    def test_health_profile_cannot_enable_unrelated_cli_modes(self):
        from minecraftrl import cli
        with patch('sys.argv',['minecraftrl','episode','--policy','steve1','--tool-health-profile','fed_low_health']):
            with self.assertRaisesRegex(SystemExit,'explicit tool collection'):cli.main()

    def bundle(self):
        return dict(world_id='w',generation='g',observation={'inventory':{'wooden_pickaxe':1,'oak_log':64}})

    def test_inventory_or_equipment_alone_cannot_prove_tool_use(self):
        stage=make_ledger(self.bundle(),'cobblestone')['stages']['cobblestone']
        self.assertFalse(criteria(stage,{'cobblestone':3},PHASES['cobblestone']))
        stage['verified_tool_harvests']=3
        self.assertFalse(criteria(stage,{'cobblestone':2},PHASES['cobblestone']))
        self.assertTrue(criteria(stage,{'cobblestone':3},PHASES['cobblestone']))

    def test_old_logs_are_baseline_and_prior_goal_record_is_untouched(self):
        b=self.bundle();b['observation']['inventory']['stone_axe']=1
        b['controller_state']={'tool_mission':dict(schema=1,world_id='w',stages={'cobblestone':{'status':'completed'}})}
        ledger=make_ledger(b,'axe_wood');stage=ledger['stages']['axe_wood'];stage['verified_tool_harvests']=64
        self.assertEqual(stage['baseline'],64)
        self.assertFalse(criteria(stage,{'oak_log':64},PHASES['axe_wood']))
        self.assertTrue(criteria(stage,{'oak_log':100,'birch_log':28},PHASES['axe_wood']))
        self.assertNotIn('axe_wood',b['controller_state']['tool_mission']['stages'])

    def test_prerequisite_and_completed_stage_cannot_be_bypassed(self):
        b=self.bundle();b['observation']['inventory']['stone_axe']=1
        with self.assertRaises(ValueError):make_ledger(b,'axe_wood')
        b['controller_state']={'tool_mission':make_ledger(b,'cobblestone')}
        b['controller_state']['tool_mission']['stages']['cobblestone']['status']='completed'
        with self.assertRaises(ValueError):make_ledger(b,'cobblestone')

    def test_instrumentation_requires_exact_pinned_harvest_and_records_after_it(self):
        marker='block.harvestBlock(this.world, this.player, pos, blockstate, tileentity, itemstack1);'
        patched=patch_harvest_observation('public class PlayerInteractionManager {\n'+marker)
        self.assertEqual(patched.count(marker),1)
        self.assertGreater(patched.index('minecraftRLRecordHarvest(block, itemstack1, pos);'),patched.index(marker))
        self.assertIn('t.equals("stone_axe")',patched)
        for invalid in ('unrecognized',marker*2):
            with self.assertRaises(ValueError):patch_harvest_observation(invalid)

    def environment(self):
        raw=Mock();gate=StopGate();wrapped=ToolCollectionEnvironment(raw,self.bundle(),Mock(),'cobblestone',gate)
        wrapped.initial_health=8.;wrapped.minimum_health=8.
        gui=dict(server_available=1,error=0,server_pick_stone=3,server_selected_item='wooden_pickaxe',
            server_health=8,server_alive=1,server_inventory={'cobblestone':3,'wooden_pickaxe':1})
        raw.step.return_value=({'inventory':{'cobblestone':0},'life_stats':{'food':5}},0,False,{'workshop':gui})
        return wrapped,raw,gate,gui

    def test_actual_server_success_blocks_next_action_despite_stale_client(self):
        wrapped,raw,gate,gui=self.environment()
        observation,*_=wrapped.step({'attack':1})
        self.assertEqual(observation['inventory']['cobblestone'],3);self.assertTrue(wrapped.completed)
        with self.assertRaises(RuntimeError):wrapped.step({'attack':1})
        self.assertEqual(raw.step.call_count,1)

    def test_missing_tool_and_health_loss_halt_without_motor_substitution(self):
        for reason in ('tool','health'):
            wrapped,raw,gate,gui=self.environment();gui['server_pick_stone']=0
            if reason=='tool':gui['server_inventory'].pop('wooden_pickaxe')
            else:gui['server_health']=5.9
            action={'forward':1};wrapped.step(action)
            self.assertFalse(wrapped.completed);self.assertTrue(gate.requested.is_set())
            raw.step.assert_called_once_with(action)

    def test_final_server_harvest_is_preserved_without_another_policy_action(self):
        wrapped,raw,gate,gui=self.environment();gui['server_pick_stone']=2
        gui['server_inventory']['cobblestone']=2;wrapped.step({'attack':1})
        wrapped.restore={'inventory_match':True};wrapped.bundle['task_checkpoint']={'preserved':'old goals'}
        raw.last_observation={'inventory':{'cobblestone':2}};raw.last_info={'workshop':dict(gui,
            server_pick_stone=3,server_inventory={'wooden_pickaxe':1,'cobblestone':3},held_key_count=0)}
        wrapped.store.commit.return_value=Path('saved-generation');gate.stop('time_limit')
        wrapped.close()
        self.assertEqual(wrapped.stage['verified_tool_harvests'],3)
        self.assertTrue(wrapped.completed);self.assertEqual(wrapped.completion_source,'cleanup')
        self.assertEqual(wrapped.final_tool_evidence['server_pick_stone'],3)
        self.assertEqual(raw.step.call_count,1)
        with self.assertRaises(RuntimeError):wrapped.step({'attack':1})

    def test_final_counter_cannot_move_backwards(self):
        wrapped,raw,gate,gui=self.environment();wrapped.step({'attack':1})
        with self.assertRaises(ValueError):wrapped.update_counter(dict(gui,server_pick_stone=2))

    def test_two_hp_loss_is_not_missed_due_to_client_serialization(self):
        wrapped,raw,gate,gui=self.environment()
        wrapped.initial_health=8.166666;gui['server_health']=6.166666030883789;gui['server_pick_stone']=0
        wrapped.step({'forward':1})
        self.assertTrue(gate.requested.is_set())
        self.assertEqual(wrapped.halt_reason,'tool_collection_health_guard')

    def test_finalize_grace_marker_requires_stopped_gate_and_verified_release(self):
        with tempfile.TemporaryDirectory() as folder:
            wrapped,raw,gate,gui=self.environment();wrapped.output=Path(folder)
            raw.release.return_value='neutral_command_sent';raw.release_evidence={'verified':True}
            wrapped.release();self.assertFalse((Path(folder)/'tool-control-finished.json').exists())
            gate.stop('time_limit');raw.release_evidence={'verified':False}
            wrapped.release();self.assertFalse((Path(folder)/'tool-control-finished.json').exists())
            raw.release_evidence={'verified':True};wrapped.release()
            self.assertTrue((Path(folder)/'tool-control-finished.json').exists())

    def test_recovery_requires_surface_contact_usable_space_and_retained_materials(self):
        g=dict(server_available=1,error=0,server_alive=1,screen=0,server_on_ground=1,server_fall_distance=0,
            server_y=70,pursuit_valid=1,pursuit_count=0,server_inventory={'cobblestone':3},table_count=0,candidate=1)
        self.assertTrue(workstation_ready(g,70))
        for change in ({'server_y':61},{'server_on_ground':0},{'candidate':0},{'pursuit_count':1},
                       {'server_inventory':{'cobblestone':2}},{'screen':1}):
            self.assertFalse(workstation_ready(dict(g,**change),70))

    def test_recovery_ledger_preserves_collection_and_uses_observed_workstation_height(self):
        b=self.bundle();ledger=make_ledger(b,'cobblestone');ledger['stages']['cobblestone']['status']='completed'
        b['controller_state']={'tool_mission':ledger,'last_workshop':{'milestones':[
            {'event':'workstation_opened','position':[211,71,-827]}]}}
        result=make_ledger(b,'workstation_access')
        self.assertEqual(result['stages']['workstation_access']['minimum_surface_y'],70)
        self.assertEqual(result['stages']['cobblestone'],ledger['stages']['cobblestone'])
        self.assertNotIn('workstation_access',ledger['stages'])
