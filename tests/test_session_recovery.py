import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

from minecraftrl.session_recovery import SessionRecovery


def server(tick,x=0,logs=5,screen=0):
    return dict(server_available=1,error=0,server_alive=1,server_tick=tick,
        server_x=x,server_y=64.,server_z=0.,yaw=0.,screen=screen,server_health=20.,
        threat_valid=1,threat_scan_radius=24,threat_count=0,threat_distance=-1,
        pursuit_valid=1,pursuit_scan_radius=32,pursuit_count=0,pursuit_distance=-1,
        server_on_ground=1,server_fall_distance=0.,
        wood_v3_kind=1,wood_v3_reachable=0,wood_v3_distance=6.,
        wood_v3_target_x=0,wood_v3_target_y=70,wood_v3_target_z=0,
        server_inventory={'oak_log':logs},wood_server_mining=0)


class RecoveryTests(unittest.TestCase):
    def make(self,history=None,threat_guard=False,hazard_skill=None,pursuit_guard=False):
        policy=SimpleNamespace(goal_embed=np.zeros((1,512),dtype=np.float32),
                               agent=SimpleNamespace(hidden_state={'kept':123}))
        skill=dict(embedding=np.ones((1,512),dtype=np.float32),sha256='a'*64)
        events=[];emit=lambda name,**kw:events.append(dict(event=name,**kw))
        r=SessionRecovery(policy,skill,history,threat_guard=threat_guard,hazard_skill=hazard_skill,pursuit_guard=pursuit_guard)
        task=dict(task_id='wood',revision=1,goal='wood',status='running')
        r.bind('world',task,server(0),0,emit)
        return r,policy,task,events,emit

    def stall(self,r,task,emit):
        for i in range(1,202):r.observe(task,server(i),{},i*.1,emit)

    def test_stall_changes_only_embedding_and_returns_after_actual_displacement(self):
        r,p,task,events,emit=self.make();memory=p.agent.hidden_state;original=copy.deepcopy(task)
        self.stall(r,task,emit)
        self.assertTrue(r.needs_switch(server(202)))
        r.switch(server(202),20.2,emit)
        np.testing.assert_array_equal(p.goal_embed,np.ones((1,512)))
        self.assertIs(p.agent.hidden_state,memory)
        r.observe(task,server(203,x=3),{},20.3,emit)
        r.switch(server(204,x=3),20.4,emit)
        np.testing.assert_array_equal(p.goal_embed,np.zeros((1,512)))
        self.assertEqual(task,original)
        self.assertEqual(r.state()['history'][-1]['outcome'],'displaced')

    def test_stop_before_handoff_and_gui_do_not_switch(self):
        r,p,task,events,emit=self.make();self.stall(r,task,emit)
        self.assertFalse(r.needs_switch(server(202,screen=1)))
        r.end('shutdown',22.,emit)
        self.assertFalse(r.needs_switch(server(203)))
        self.assertEqual(len(r.state()['history']),0)
        np.testing.assert_array_equal(p.goal_embed,np.zeros((1,512)))

    def test_visual_root_guidance_is_restored_after_text_recovery_without_memory_reset(self):
        for scale in (0.,2.,7.):
            r,p,task,events,emit=self.make();p.agent.cond_scale=scale;memory=p.agent.hidden_state
            r.bind('world',task,server(0),0,emit);self.stall(r,task,emit)
            r.switch(server(202),20.2,emit)
            self.assertEqual(p.agent.cond_scale,6.)
            self.assertIs(p.agent.hidden_state,memory)
            r.observe(task,server(203,x=3),{},20.3,emit);r.switch(server(204,x=3),20.4,emit)
            self.assertEqual(p.agent.cond_scale,scale)
            self.assertIs(p.agent.hidden_state,memory)

    def test_hazard_guidance_is_part_of_method_and_failed_settings_cannot_cycle(self):
        r,p,task,events,emit=self.make(threat_guard=True)
        r.observe(task,dict(server(1),server_health=16.),{},.1,emit);r.switch(server(2),.2,emit)
        r.end('recovery_blocked',2.,emit);original=r.state()
        candidate=dict(r.skill,cond_scale=0.)
        other,p,task,events,emit=self.make(original,threat_guard=True,hazard_skill=candidate)
        self.assertIsNone(other.blocked);self.assertIsNone(other.resume_from)
        memory=p.agent.hidden_state;other.switch(server(1),.1,emit)
        self.assertEqual(p.agent.cond_scale,0.)
        self.assertEqual(other.state()['history'][-1]['cond_scale'],0.)
        self.assertIs(p.agent.hidden_state,memory)
        other.end('recovery_blocked',2.,emit)
        restored,*_=self.make(other.state(),threat_guard=True)
        self.assertIsNotNone(restored.blocked)

    def test_same_embedding_guidance_retarget_preserves_budget_and_resume_identity(self):
        r,p,task,events,emit=self.make(threat_guard=True,
            hazard_skill=dict(embedding=np.ones((1,512),dtype=np.float32),sha256='a'*64,cond_scale=0.))
        self.stall(r,task,emit);r.switch(server(202),20.2,emit)
        r.end('shutdown',21.,emit)
        resumed,p,task,events,emit=self.make(r.state(),threat_guard=True,hazard_skill=r.hazard_skill)
        resumed.switch(server(1),.1,emit);entered=resumed.enter_time;attempts=resumed.attempts
        resumed.observe(task,dict(server(2),threat_count=1,threat_distance=5.),{},.2,emit)
        resumed.switch(server(3),.3,emit)
        self.assertEqual(p.agent.cond_scale,0.);self.assertEqual(resumed.enter_time,entered)
        self.assertEqual(resumed.attempts,attempts);self.assertEqual(resumed.saved['history'][-1]['initial_cond_scale'],6.)
        resumed.end('shutdown',.4,emit)
        restored,*_=self.make(resumed.state(),threat_guard=True,hazard_skill=r.hazard_skill)
        self.assertIsNone(restored.blocked)
        self.assertEqual(restored.resume_from,1)

    def test_health_loss_takes_priority_over_inventory_progress(self):
        r,p,task,events,emit=self.make()
        r.observe(task,dict(server(1,logs=6),server_health=16.),{},.1,emit)
        self.assertTrue(r.needs_switch(server(2)))
        r.switch(dict(server(2,logs=6),server_health=16.),.2,emit)
        self.assertEqual(r.state()['history'][-1]['trigger'],'health_loss')
        r.observe(task,dict(server(3,logs=7),server_health=16.),{},.3,emit)
        self.assertFalse(r.needs_switch(server(4)))
        r.observe(task,dict(server(4,x=3,logs=7),server_health=16.),{},.4,emit)
        self.assertFalse(r.needs_switch(server(5,x=3)))
        r.observe(task,dict(server(90,x=12,logs=7),server_health=16.),{},18.,emit)
        self.assertFalse(r.needs_switch(server(91,x=12)))
        r.observe(task,dict(server(102,x=12,logs=7),server_health=16.),{},20.4,emit)
        r.switch(dict(server(103,x=12,logs=7),server_health=16.),20.6,emit)
        self.assertEqual(r.state()['history'][-1]['outcome'],'damage_stabilized')
        self.assertIsNone(r.state()['pending_hazard'])

    def test_occluded_pursuer_preempts_collection_and_prevents_false_stability(self):
        r,p,task,events,emit=self.make(threat_guard=True,pursuit_guard=True)
        chase=lambda tick:dict(server(tick,x=15),pursuit_count=1,pursuit_distance=20.)
        r.observe(task,chase(1),{},.1,emit);r.switch(chase(2),.2,emit)
        self.assertEqual(r.saved['history'][-1]['completion_rule'],'pursuit-v8')
        r.observe(task,dict(chase(150),server_x=30),{},20.,emit)
        self.assertFalse(r.needs_switch(server(151)))
        r.observe(task,server(249,x=30),{},30.,emit)
        self.assertFalse(r.needs_switch(server(249)))
        r.observe(task,server(250,x=30),{},30.2,emit);r.switch(server(251,x=30),30.3,emit)
        self.assertEqual(r.saved['history'][-1]['outcome'],'hazard_stabilized')
        with self.assertRaisesRegex(ValueError,'requires pursuit guard'):self.make(r.state(),threat_guard=True)

    def test_new_pursuit_criterion_retains_old_failure_but_its_own_budget_cannot_restart(self):
        r,p,task,events,emit=self.make(threat_guard=True)
        r.observe(task,dict(server(1),server_health=16.),{},.1,emit);r.switch(server(2),.2,emit)
        r.end('recovery_blocked',2.,emit);old=r.state()
        changed,p,task,events,emit=self.make(old,threat_guard=True,pursuit_guard=True)
        self.assertIsNone(changed.blocked);self.assertEqual(changed.saved['history'],old['history'])
        changed.switch(server(1),.1,emit)
        changed.observe(task,dict(server(901),pursuit_count=1,pursuit_distance=20.),{},160.,emit)
        changed.switch(server(902),160.1,emit)
        restored,*_=self.make(changed.state(),threat_guard=True,pursuit_guard=True)
        self.assertEqual(restored.blocked,'damage_recovery_incomplete')

    def test_quiet_midair_observation_cannot_complete_pursuit_recovery(self):
        r,p,task,events,emit=self.make(threat_guard=True,pursuit_guard=True)
        r.observe(task,dict(server(1),pursuit_count=1,pursuit_distance=20.),{},.1,emit);r.switch(server(2),.2,emit)
        r.observe(task,dict(server(102,x=20),server_on_ground=0,server_fall_distance=5.),{},20.,emit)
        self.assertFalse(r.needs_switch(server(103)))
        self.assertFalse(any(e['event']=='hazard_stability_observed' for e in events))
        r.observe(task,dict(server(103,x=20),server_health=14.),{},20.2,emit)
        self.assertFalse(r.needs_switch(server(104)))
        r.observe(task,dict(server(203,x=20),server_health=14.),{},40.2,emit)
        r.switch(server(204,x=20),40.3,emit)
        self.assertEqual(r.saved['history'][-1]['outcome'],'hazard_stabilized')

    def test_interrupted_pursuit_uses_only_remaining_expanded_budget(self):
        r,p,task,events,emit=self.make(threat_guard=True,pursuit_guard=True)
        r.observe(task,dict(server(1),pursuit_count=1,pursuit_distance=20.),{},.1,emit)
        r.switch(server(2),.2,emit)
        r.observe(task,dict(server(367,x=42),pursuit_count=1,pursuit_distance=15.),{},81.4,emit)
        r.end('shutdown',81.4,emit);saved=r.state();saved['history'][-1]['completion_rule']='pursuit-v5'
        restored,p,task,events,emit=self.make(saved,threat_guard=True,pursuit_guard=True)
        scheduled=next(e for e in events if e['event']=='recovery_resume_scheduled')
        self.assertEqual(scheduled['remaining_ticks'],535)
        self.assertAlmostEqual(scheduled['remaining_seconds'],98.8)
        restored.switch(server(1),.1,emit)
        restored.observe(task,dict(server(536,x=42),pursuit_count=1,pursuit_distance=15.),{},98.,emit)
        restored.switch(server(537,x=42),98.1,emit)
        self.assertEqual(restored.blocked,'damage_recovery_incomplete')
        self.assertEqual(restored.saved['history'][-1]['continued_from'],0)
        self.make(restored.state(),threat_guard=True,pursuit_guard=True)

    def test_clear_grounded_pursuit_handoff_needs_no_arbitrary_further_travel(self):
        r,p,task,events,emit=self.make(threat_guard=True,pursuit_guard=True)
        r.observe(task,dict(server(1),pursuit_count=1,pursuit_distance=20.),{},.1,emit);r.switch(server(2),.2,emit)
        r.observe(task,server(101),{},20.,emit);self.assertFalse(r.needs_switch(server(101)))
        r.observe(task,server(102),{},20.2,emit);r.switch(server(103),20.3,emit)
        self.assertEqual(r.saved['history'][-1]['outcome'],'hazard_stabilized')
        evidence=next(e for e in events if e['event']=='hazard_stability_observed')
        self.assertEqual(evidence['displacement_blocks'],0.);self.assertEqual(evidence['required_blocks'],0.)

    def test_damage_alert_survives_stop_before_switch_and_new_damage_restarts_quiet_ticks(self):
        r,p,task,events,emit=self.make()
        r.observe(task,dict(server(1),server_health=16.),{},.1,emit)
        r.end('shutdown',.2,emit)
        restored,p,task,events,emit=self.make(r.state())
        self.assertTrue(restored.needs_switch(server(1)))
        restored.switch(server(1),.1,emit)
        restored.observe(task,dict(server(90,x=12),server_health=19.),{},18.,emit)
        restored.observe(task,dict(server(101,x=12),server_health=19.),{},20.2,emit)
        self.assertFalse(restored.needs_switch(server(102,x=12)))

    def test_exhausted_damage_skill_cannot_reset_its_budget_on_new_process(self):
        r,p,task,events,emit=self.make()
        r.observe(task,dict(server(1),server_health=16.),{},.1,emit)
        r.switch(dict(server(2),server_health=16.),.2,emit)
        r.needs_switch(server(3),30.3);r.switch(server(3),30.3,emit)
        restored,p,task,events,emit=self.make(r.state())
        self.assertEqual(restored.blocked,'damage_recovery_incomplete')
        self.assertFalse(restored.needs_switch(server(1)))

    def test_hostile_approach_preempts_progress_and_needs_a_quiet_scan_before_return(self):
        r,p,task,events,emit=self.make(threat_guard=True)
        threat=lambda tick,x=0:dict(server(tick,x=x,logs=6),threat_count=1,threat_distance=18.)
        r.observe(task,threat(1),{},.1,emit);r.switch(threat(2),.2,emit)
        self.assertEqual(r.state()['history'][-1]['trigger'],'hostile_proximity')
        r.observe(task,threat(100,x=20),{},20.,emit)
        self.assertFalse(r.needs_switch(threat(101,x=20)))
        r.observe(task,server(199,x=20,logs=7),{},25.,emit)
        self.assertFalse(r.needs_switch(server(200,x=20)))
        r.observe(task,server(200,x=20,logs=7),{},25.2,emit)
        r.switch(server(201,x=20,logs=7),25.4,emit)
        self.assertEqual(r.state()['history'][-1]['outcome'],'hazard_stabilized')
        self.assertIsNone(r.state()['pending_hazard'])

    def test_old_pending_damage_state_migrates_without_forgetting_urgency(self):
        history=dict(schema=1,history=[],high_water={'world:wood:1':5},pending_damage='world:wood:1')
        r,p,task,events,emit=self.make(history,threat_guard=True)
        self.assertEqual(r.state()['pending_hazard'],dict(context='world:wood:1',trigger='health_loss'))
        self.assertTrue(r.needs_switch(server(1)))
        self.assertNotIn('pending_damage',r.state())

    def test_hazard_budget_upgrade_retains_used_ticks_and_original_timeout(self):
        history=dict(schema=1,high_water={'world:wood:1':5},
            pending_hazard=dict(context='world:wood:1',trigger='hostile_proximity'),
            history=[dict(context='world:wood:1',position=[0.,64.,0.],high_water=5,
                outcome='time_budget',elapsed_ticks=125,elapsed_s=30.2,
                trigger='hostile_proximity',completion_rule='hazard-v3')])
        r,p,task,events,emit=self.make(history,threat_guard=True)
        self.assertTrue(r.needs_switch(server(1)))
        resumed=next(e for e in events if e['event']=='recovery_resume_scheduled')
        self.assertEqual(resumed['remaining_ticks'],275)
        self.assertAlmostEqual(resumed['remaining_seconds'],59.8)
        r.switch(server(1),.1,emit)
        r.observe(task,server(101,x=12),{},20.1,emit);r.switch(server(102,x=12),20.2,emit)
        self.assertEqual(r.state()['history'][0],history['history'][0])
        self.assertEqual(r.state()['history'][1]['continued_from'],0)
        self.assertEqual(r.totals(1)[0],225)
        self.make(r.state(),threat_guard=True)

    def test_experimental_condition_changes_hazard_only_without_resetting_memory_or_budget(self):
        hazard=dict(embedding=np.full((1,512),2.,dtype=np.float32),sha256='b'*64,provenance={})
        r,p,task,events,emit=self.make(threat_guard=True,hazard_skill=hazard)
        self.stall(r,task,emit);r.switch(server(202),20.2,emit)
        np.testing.assert_array_equal(p.goal_embed,np.ones((1,512)))
        memory=p.agent.hidden_state
        r.observe(task,dict(server(203),threat_count=1,threat_distance=10.),{},20.3,emit)
        r.switch(server(204),20.4,emit)
        np.testing.assert_array_equal(p.goal_embed,hazard['embedding'])
        self.assertIs(p.agent.hidden_state,memory)
        self.assertEqual(r.enter_tick,202);self.assertEqual(r.enter_time,20.2)
        self.assertEqual(r.attempts,1)
        self.assertEqual(r.state()['history'][-1]['initial_embedding_sha256'],'a'*64)
        r.observe(task,server(305,x=15),{},40.5,emit);r.switch(server(306,x=15),40.6,emit)
        np.testing.assert_array_equal(p.goal_embed,np.zeros((1,512)))
        self.make(r.state(),threat_guard=True,hazard_skill=hazard)

    def test_new_condition_retains_failed_method_and_prevents_a_b_a_retry_cycle(self):
        hazard=dict(embedding=np.full((1,512),2.,dtype=np.float32),sha256='b'*64,provenance={})
        old=dict(context='world:wood:1',position=[0.,64.,0.],high_water=5,outcome='recovery_blocked',
                 elapsed_ticks=40,elapsed_s=10.,trigger='hostile_proximity',completion_rule='hazard-v4',embedding_sha256='a'*64)
        history=dict(schema=1,high_water={'world:wood:1':5},history=[old],
                     pending_hazard=dict(context='world:wood:1',trigger='hostile_proximity'))
        r,p,task,events,emit=self.make(history,threat_guard=True,hazard_skill=hazard)
        self.assertTrue(r.needs_switch(server(1)));r.switch(server(1),.1,emit)
        r.end('recovery_blocked',1.,emit)
        self.assertEqual(r.state()['history'][0],old)
        self.assertNotIn('continued_from',r.state()['history'][1])
        back,p,task,events,emit=self.make(r.state(),threat_guard=True)
        self.assertEqual(back.blocked,'duplicate_area_without_inventory_progress')
        self.assertFalse(back.needs_switch(server(1)))

    def test_hazard_retarget_inside_resumed_stall_keeps_a_loadable_method_chain(self):
        hazard=dict(embedding=np.full((1,512),2.,dtype=np.float32),sha256='b'*64,provenance={})
        r,p,task,events,emit=self.make(threat_guard=True,hazard_skill=hazard)
        self.stall(r,task,emit);r.switch(server(202),20.2,emit)
        r.observe(task,server(210),{},21.,emit);r.end('shutdown',21.1,emit)
        resumed,p,task,events,emit=self.make(r.state(),threat_guard=True,hazard_skill=hazard)
        resumed.switch(server(1),.1,emit)
        resumed.observe(task,dict(server(2),threat_count=1,threat_distance=12.),{},.2,emit)
        resumed.switch(server(3),.3,emit);resumed.end('shutdown',.5,emit)
        again,p,task,events,emit=self.make(resumed.state(),threat_guard=True,hazard_skill=hazard)
        self.assertEqual(again.saved['history'][1]['continued_from'],0)
        self.assertEqual(again.saved['history'][1]['initial_embedding_sha256'],'a'*64)
        self.assertEqual(again.saved['history'][1]['embedding_sha256'],'b'*64)
        self.assertTrue(again.needs_switch(server(1)))

    def test_unresolved_damage_escape_blocks_collection_and_keeps_budget_on_resume(self):
        r,p,task,events,emit=self.make()
        r.observe(task,dict(server(1,logs=6),server_health=16.),{},.1,emit)
        r.switch(dict(server(2,logs=6),server_health=16.),.2,emit)
        r.observe(task,dict(server(3,logs=7),server_health=16.),{},.3,emit)
        r.end('shutdown',10.2,emit)
        resumed,p,task,events,emit=self.make(r.state())
        self.assertTrue(resumed.needs_switch(server(1)))
        resumed.switch(server(1),.1,emit)
        self.assertEqual(resumed.state()['history'][-1]['trigger'],'health_loss')
        self.assertTrue(resumed.needs_switch(server(2),20.2))
        resumed.switch(server(2),20.2,emit)
        self.assertEqual(resumed.blocked,'damage_recovery_incomplete')
        self.assertEqual(resumed.totals(1)[0],1)
        self.make(resumed.state())  # Continued receipt remains loadable after new loot.

    def test_budget_and_duplicate_attempt_history_survive_restore(self):
        r,p,task,events,emit=self.make();self.stall(r,task,emit)
        r.switch(server(202),20.2,emit)
        for i in range(203,403):r.observe(task,server(i),{},20.2+(i-202)*.1,emit)
        self.assertTrue(r.needs_switch(server(403)))
        r.switch(server(403),40.3,emit)
        self.assertEqual(r.state()['history'][-1]['outcome'],'tick_budget')
        restored,p,task,events,emit=self.make(r.state())
        self.stall(restored,task,emit)
        self.assertFalse(restored.needs_switch(server(202)))
        self.assertTrue(any(e['event']=='recovery_suppressed' for e in events))

    def test_interrupted_skill_resumes_only_remaining_budget_without_losing_receipt(self):
        r,p,task,events,emit=self.make();self.stall(r,task,emit);r.switch(server(202),20.2,emit)
        for tick in range(203,271):r.observe(task,server(tick),{},20.2+(tick-202)*15/68,emit)
        r.end('shutdown',35.2,emit);old=r.state()['history'][0]
        resumed,p,task,events,emit=self.make(r.state())
        self.assertTrue(resumed.needs_switch(server(1)))
        resumed.switch(server(1),.1,emit)
        for tick in range(2,134):resumed.observe(task,server(tick),{},.1+(tick-1)*.05,emit)
        resumed.switch(server(134),6.8,emit)
        saved=resumed.state()
        self.assertEqual(saved['history'][0],old)
        self.assertEqual(saved['history'][1]['continued_from'],0)
        self.assertEqual(resumed.totals(1)[0],200)
        self.assertEqual(saved['history'][1]['outcome'],'tick_budget')
        again,p,task,events,emit=self.make(saved)
        self.assertFalse(again.needs_switch(server(1)))
        self.stall(again,task,emit)
        self.assertTrue(any(e['event']=='recovery_suppressed' for e in events))

    def test_new_inventory_cancels_pending_escape_and_invalid_chain_is_rejected(self):
        r,p,task,events,emit=self.make();self.stall(r,task,emit)
        r.observe(task,server(202,logs=6),{},20.2,emit)
        self.assertFalse(r.needs_switch(server(203)))
        r,p,task,events,emit=self.make();self.stall(r,task,emit);r.switch(server(202),20.2,emit)
        r.end('shutdown',21.,emit);invalid=r.state()
        invalid['history'][0]['continued_from']=0
        with self.assertRaises(ValueError):self.make(invalid)

    def test_stop_retains_already_observed_skill_completion(self):
        r,p,task,events,emit=self.make();self.stall(r,task,emit);r.switch(server(202),20.2,emit)
        r.observe(task,server(203,x=3),{},20.3,emit)
        r.end('shutdown',20.4,emit)
        self.assertEqual(r.state()['history'][-1]['outcome'],'displaced')
        self.assertFalse(r.needs_switch(server(204)))

    def test_return_delay_does_not_replace_observed_displacement_with_timeout(self):
        r,p,task,events,emit=self.make();self.stall(r,task,emit);r.switch(server(202),20.2,emit)
        r.observe(task,server(203,x=3),{},49.9,emit)
        # Input release/serialization can finish just after the wall deadline.
        self.assertTrue(r.needs_switch(server(204,x=3),50.3))
        r.switch(server(204,x=3),50.3,emit)
        self.assertEqual(r.state()['history'][-1]['outcome'],'displaced')

    def test_invalid_history_and_invalid_server_are_rejected(self):
        with self.assertRaises(ValueError):self.make({'schema':1,'history':[],'high_water':{'world:wood:1':float('nan')}})
        r,p,task,events,emit=self.make()
        with self.assertRaises(ValueError):r.observe(task,dict(server(1),server_available=0),{},.1,emit)

    def test_wall_budget_and_completion_cancel_further_exploration(self):
        r,p,task,events,emit=self.make();self.stall(r,task,emit)
        r.switch(server(202),20.2,emit)
        self.assertTrue(r.needs_switch(server(202,screen=1),50.3))
        r.switch(server(202,screen=1),50.3,emit)
        self.assertEqual(r.state()['history'][-1]['outcome'],'time_budget')
        np.testing.assert_array_equal(p.goal_embed,np.zeros((1,512)))
        r,p,task,events,emit=self.make();self.stall(r,task,emit)
        r.switch(server(202),20.2,emit)
        r.observe(dict(task,status='completed'),server(203,logs=64),{},20.3,emit)
        self.assertFalse(r.needs_switch(server(204),80))
        self.assertEqual(r.state()['history'][-1]['outcome'],'terminal')

    def runtime(self,complete_on_release=False,death_after=None,death_on_release=False,low_health_after=None,threat_after=None):
        from minecraftrl.task_runtime import run_task_session
        from minecraftrl.tasks import CommandBus
        from minecraftrl.runtime import StopGate
        gate=StopGate();bus=CommandBus()
        bus.submit(dict(command_id='a',op='request',task_id='wood',goal='wood',count=64))
        class Environment:
            tick=0;x=0.;actions=[];releases=0
            def seed(self,seed):pass
            # Deliberately stale/incorrect client inventory must not complete the RM.
            def observation(self):return dict(inventory={'oak_log':69 if self.tick>=50 else 5},pov=None)
            def refresh(self):
                self.last_observation=self.observation()
                self.last_info={'workshop':server(self.tick,x=self.x,logs=69 if complete_on_release and self.releases>=2 else 5)}
                if low_health_after is not None and len(self.actions)>=low_health_after:
                    self.last_info['workshop']['server_health']=7.
                if threat_after is not None and len(self.actions)>=threat_after:
                    self.last_info['workshop'].update(threat_count=1,threat_distance=4.)
                if ((death_after is not None and len(self.actions)>=death_after)
                        or (death_on_release and self.releases>=2)):
                    self.last_info['workshop'].update(server_alive=0,server_health=0,server_inventory={})
                    # Reproduce live server death preceding the client/done signal.
                    self.last_observation['life_stats']={'life':2.,'is_alive':True}
            def reset(self):self.refresh();return self.last_observation
            def step(self,action):
                self.tick+=1;self.x+=action['forward'][0];self.actions.append(action)
                self.refresh();return self.last_observation,0,False,self.last_info
            def release(self):
                self.tick+=1;self.releases+=1;self.refresh();self.release_evidence={'verified':True,'held_key_count':0}
                return 'neutral_command_sent'
            def close(self):pass
        class Policy:
            calls=0
            def __init__(self):self.agent=SimpleNamespace(hidden_state={'kept':123})
            def reset(self,goal):self.goal_embed=np.zeros((1,512),dtype=np.float32)
            def act(self,screen):
                self.calls+=1
                if self.calls==410:gate.stop('test_complete')
                return {'forward':[int(self.goal_embed[0,0])],'attack':[0]}
            def close(self):pass
        policy=Policy();env=Environment()
        r=SessionRecovery(policy,dict(embedding=np.ones((1,512),dtype=np.float32),sha256='a'*64),threat_guard=threat_after is not None)
        with tempfile.TemporaryDirectory() as tmp:
            report=run_task_session(policy,env,tmp,seed=1,limit_s=20,no_progress_s=20,bus=bus,gate=gate,
                save_screen=lambda *a:None,recovery=r)
            events=[json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        return report,events,env

    def test_runtime_switch_release_stop_and_original_goal_are_preserved(self):
        report,events,env=self.runtime()
        switches=[e for e in events if e['event']=='neural_skill_switched']
        self.assertEqual([e['skill'] for e in switches],['leave_stalled_area','collection'])
        self.assertEqual(sum(a['forward'][0] for a in env.actions),3)
        self.assertEqual(report['task']['target_count'],64)
        self.assertEqual(report['task']['progress'],0)
        stopped=next(i for i,e in enumerate(events) if e['event']=='stop_acknowledged')
        self.assertFalse(any(e['event']=='policy_dispatch_started' for e in events[stopped+1:]))
        self.assertTrue(report['release_evidence']['verified'])
        self.assertEqual(report['neural_recovery']['state']['history'][0]['outcome'],'displaced')

    def test_pickup_during_handoff_release_completes_root_without_exploration_action(self):
        report,events,env=self.runtime(complete_on_release=True)
        self.assertEqual(report['status'],'task_completed')
        self.assertEqual(sum(a['forward'][0] for a in env.actions),0)
        accepted=next(i for i,e in enumerate(events) if e['event']=='inventory_target_reached')
        self.assertFalse(any(e['event']=='policy_dispatch_started' for e in events[accepted+1:]))
        self.assertEqual(report['neural_recovery']['state']['history'],[])

    def test_server_death_ends_policy_before_delayed_client_done(self):
        report,events,env=self.runtime(death_after=3)
        self.assertEqual(report['status'],'environment_done')
        self.assertTrue(report['execution_valid']);self.assertNotIn('error',report)
        self.assertEqual(len(env.actions),3)
        terminal=next(e for e in events if e['event']=='server_death_observed')
        self.assertFalse(terminal['raw_done'])
        self.assertFalse(any(e['event']=='policy_dispatch_started' and e['monotonic_s']>terminal['monotonic_s'] for e in events))
        self.assertTrue(report['release_evidence']['verified'])

    def test_server_death_during_skill_handoff_blocks_exploration(self):
        report,events,env=self.runtime(death_on_release=True)
        self.assertEqual(report['status'],'environment_done')
        self.assertNotIn('error',report)
        self.assertEqual(sum(a['forward'][0] for a in env.actions),0)
        self.assertFalse(any(e['event']=='neural_skill_switched' for e in events))

    def test_low_server_health_pauses_before_another_policy_action(self):
        report,events,env=self.runtime(low_health_after=3)
        self.assertEqual(report['status'],'recovery_blocked')
        self.assertEqual(report['recovery_blocked'],'low_health_requires_plan')
        self.assertTrue(report['execution_valid'])
        self.assertEqual(len(env.actions),3)
        self.assertTrue(report['release_evidence']['verified'])

    def test_runtime_threat_switches_before_damage_and_does_not_resume_under_persistent_threat(self):
        report,events,env=self.runtime(threat_after=3)
        self.assertEqual(report['status'],'recovery_blocked')
        self.assertEqual(report['recovery_blocked'],'damage_recovery_incomplete')
        self.assertEqual(sum(a['forward'][0]==0 for a in env.actions),3)
        self.assertTrue(report['neural_recovery']['threat_guard'])
        self.assertTrue(report['release_evidence']['verified'])

    def test_controller_history_is_bound_to_saved_generation(self):
        from test_world_store import WorldStoreTests
        from minecraftrl.world_store import WorldStore
        r,p,task,events,emit=self.make();self.stall(r,task,emit);r.switch(server(202),20.2,emit)
        r.end('shutdown',21.,emit)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);world,checkpoint,obs=WorldStoreTests().fixture(root)
            with WorldStore(root/'store','companion') as store:
                old=store.commit(world,checkpoint,obs)
                store.commit(world,checkpoint,obs,controller_state={'neural_recovery':r.state()})
                self.assertEqual(store.load()['controller_state']['neural_recovery'],r.state())
                self.assertNotIn('controller_state',json.loads((old/'manifest.json').read_text()))

    def test_recovery_receipt_error_cannot_skip_input_release_and_close(self):
        from test_task_session import FakeEnvironment
        from minecraftrl.task_runtime import run_task_session
        from minecraftrl.runtime import StopGate
        from unittest.mock import Mock
        env=FakeEnvironment();env.last_info={'workshop':server(0)}
        env.release_evidence={'verified':True,'held_key_count':0}
        recovery=Mock();recovery.end.side_effect=OSError('receipt failed')
        policy=Mock();gate=StopGate();gate.stop('test')
        with tempfile.TemporaryDirectory() as tmp:
            report=run_task_session(policy,env,tmp,seed=1,gate=gate,recovery=recovery,save_screen=lambda *a:None)
        self.assertEqual(report['status'],'error')
        self.assertIn('receipt failed',report['recovery_close_error'])
        self.assertEqual(report['release'],'neutral_command_sent')
        self.assertTrue(env.closed)
        policy.close.assert_called_once()


if __name__=='__main__':unittest.main()
