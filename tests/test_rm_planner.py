import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from test_planning_recovery import Game
from minecraftrl.replanning import (PlanHistory, ReplanGuard, make_snapshot,
    collection_capability, goal_conditioning_capability)


def setup_packet(folder):
    from minecraftrl.rm_planner import RMPlanScheduler, planning_packet
    obs=Game().state
    snapshot=make_snapshot(obs,world_id='world',episode_id='episode',goal='wood',task=None,
        generation=1,sequence=10,elapsed_s=20.,status='stalled',release_sequence=10,held_key_count=0)
    caps={'collect_wood':collection_capability('wood','policy'),
          'explore_then_resume':goal_conditioning_capability('policy','a'*64,'b'*64)}
    guard=ReplanGuard(PlanHistory(),caps,allow_goal_conditioning=True)
    event=dict(event_id='stall-1',event='stalled',stalled=True,observation_valid=True,
               tick=snapshot['tick'],reason='no_progress')
    guard.record_failure(snapshot,event,'collect_wood')
    request=guard.request(snapshot,event)
    scheduler=RMPlanScheduler(Path(folder)/'schedule.json',snapshot)
    return guard,scheduler,planning_packet(guard,request['request_id'],scheduler),snapshot


class RMPlanTests(unittest.TestCase):
    def setUp(self):
        from minecraftrl.rm_planner import fixture_plan
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.guard,self.scheduler,self.packet,self.current=setup_packet(self.tmp.name)
        self.plan=fixture_plan(self.packet)

    def test_plan_orders_supported_subgoal_before_preserved_root(self):
        root=deepcopy(self.scheduler.state['root'])
        self.scheduler.install(self.plan,self.packet,self.current)
        selection=self.scheduler.select(self.current,self.guard.history.snapshot())
        self.assertEqual(selection['node']['skill_id'],'explore_then_resume')
        self.assertEqual(self.scheduler.state['root']['reference'],root['reference'])
        self.assertEqual(self.scheduler.state['root']['progress'],root['progress'])
        self.assertEqual(self.scheduler.state['root']['status'],'suspended')
        self.assertEqual(self.scheduler.state['plans'][0]['proposal'],self.plan)

    def test_unsupported_plan_is_retained_and_blocks_dependent_root(self):
        from minecraftrl.rm_planner import fixture_plan
        plan=fixture_plan(self.packet,unsupported=True)
        self.scheduler.install(plan,self.packet,self.current)
        selection=self.scheduler.select(self.current,self.guard.history.snapshot())
        self.assertIsNone(selection['node'])
        self.assertEqual(selection['reason'],'missing_capability')
        self.assertEqual(self.scheduler.state['plans'][0]['nodes'][0]['support'],'unsupported')
        self.assertFalse(self.scheduler.state['root']['completed'])

    def test_priority_only_selects_ready_nodes(self):
        self.plan['steps'][-1]['priority']=100
        self.scheduler.install(self.plan,self.packet,self.current)
        result=self.scheduler.select(self.current,self.guard.history.snapshot())
        self.assertEqual(result['node']['id'],'search')

    def test_cycle_and_missing_dependency_rejected(self):
        for dependency in ('root','absent'):
            plan=deepcopy(self.plan);plan['steps'][0]['depends_on']=[dependency]
            with self.assertRaises(ValueError):self.scheduler.install(plan,self.packet,self.current)
        self.assertEqual(self.scheduler.state['plans'],[])

    def test_bound_root_reference_is_normalized_without_mutating_response(self):
        from minecraftrl.rm_planner import validate_plan
        wire=deepcopy(self.plan);wire['steps'][-1]['id']=self.packet['root_id']
        result=validate_plan(wire,self.packet)
        self.assertEqual(result,self.plan)
        self.assertEqual(wire['steps'][-1]['id'],self.packet['root_id'])
        self.scheduler.install(wire,self.packet,self.current)
        self.assertEqual(self.scheduler.state['plans'][0]['proposal'],self.plan)

    def test_root_reference_alias_cannot_rebind_goal_or_hide_duplicate(self):
        from minecraftrl.rm_planner import validate_plan
        wire=deepcopy(self.plan);wire['steps'][-1]['id']='0'*64
        with self.assertRaises(ValueError):validate_plan(wire,self.packet)
        wire=deepcopy(self.plan);duplicate=deepcopy(wire['steps'][-1])
        duplicate['id']=self.packet['root_id'];wire['steps'].append(duplicate)
        with self.assertRaises(ValueError):validate_plan(wire,self.packet)
        wire=deepcopy(self.plan);wire['steps'][0]['id']=self.packet['root_id']
        with self.assertRaises(ValueError):validate_plan(wire,self.packet)

    def test_root_reference_alias_does_not_relax_completion_or_scope(self):
        from minecraftrl.rm_planner import validate_plan
        for change in ({'completion':'unsupported'},{'skill_id':'collect_dirt'},
                       {'depends_on':['root']},{'entry':'active_root'}):
            wire=deepcopy(self.plan);wire['steps'][-1]['id']=self.packet['root_id']
            wire['steps'][-1].update(change)
            with self.assertRaises(ValueError):validate_plan(wire,self.packet)

    def test_cannot_change_root_completion_or_add_motor_fields(self):
        for change in ({'completion':'fresh_target_or_interaction'},{'key':'forward'},
                       {'skill_id':'collect_dirt'},{'entry':'active_root'}):
            plan=deepcopy(self.plan);plan['steps'][-1].update(change)
            with self.assertRaises(ValueError):self.scheduler.install(plan,self.packet,self.current)

    def test_stale_stopped_completed_and_unreleased_do_not_install(self):
        for patch in ({'generation':2},{'status':'user_paused'},{'held_key_count':1},
                      {'elapsed_s':51.}):
            current=deepcopy(self.current);current.update(patch)
            with self.assertRaises(ValueError):self.scheduler.install(self.plan,self.packet,current)
        current=deepcopy(self.current);current['facts']['goal_inventory']=1
        with self.assertRaises(ValueError):self.scheduler.install(self.plan,self.packet,current)
        self.assertEqual(self.scheduler.state['plans'],[])

    def test_duplicate_is_checked_again_at_dispatch(self):
        self.scheduler.install(self.plan,self.packet,self.current)
        data=self.guard.history.snapshot()
        row=deepcopy(data['failures'][0]);cap=self.guard.capabilities['explore_then_resume']
        row.update(skill_id=cap['skill_id'],arguments=cap['arguments'])
        data['failures'].append(row)
        self.assertEqual(self.scheduler.select(self.current,data)['reason'],'same_failed_attempt')

    def test_executable_budget_not_extended_by_llm(self):
        plan=deepcopy(self.plan);plan['steps'][0]['budget_ticks']=201
        with self.assertRaises(ValueError):self.scheduler.install(plan,self.packet,self.current)

    def test_execution_receipt_required_to_complete_subgoal(self):
        self.scheduler.install(self.plan,self.packet,self.current)
        self.scheduler.select(self.current,self.guard.history.snapshot())
        with self.assertRaises(ValueError):
            self.scheduler.finish('search',{'classification':'target_acquired'},self.current)
        self.assertFalse(self.scheduler.state['root']['completed'])

    def test_packet_retains_history_without_executable_alternatives(self):
        from minecraftrl.rm_planner import planning_packet, payload
        self.guard.record_failure(self.current,dict(event_id='stall-2',event='stalled',stalled=True,
            observation_valid=True,tick=0,reason='no_progress'),'explore_then_resume')
        packet=planning_packet(self.guard,self.packet['request_id'],self.scheduler)
        self.assertEqual(len(packet['memory']['blocked_methods']),2)
        self.assertTrue(packet['proposal_allowed'])
        result=payload(packet)
        self.assertIn('steps',result['text']['format']['schema']['properties'])
        self.assertEqual(len(json.loads(result['input'][1]['content'])['memory']['failed_attempts']),2)

    def test_provider_uses_paid_guard_and_full_plan_schema(self):
        from minecraftrl.rm_planner import RMPlanProvider
        with self.assertRaises(PermissionError):RMPlanProvider('fixture','fake-key',allow_paid=False)

    def test_restart_preserves_proposal_and_never_auto_resumes(self):
        from minecraftrl.rm_planner import RMPlanScheduler
        self.scheduler.install(self.plan,self.packet,self.current)
        self.scheduler.select(self.current,self.guard.history.snapshot())
        restored=RMPlanScheduler(Path(self.tmp.name)/'schedule.json',self.current)
        self.assertEqual(restored.state['plans'][0]['proposal'],self.plan)
        self.assertEqual(restored.select(self.current,self.guard.history.snapshot())['reason'],'restored_paused')

    def test_other_world_inventory_cannot_complete_root(self):
        self.scheduler.install(self.plan,self.packet,self.current)
        current=deepcopy(self.current);current['world_id']='other';current['facts']['goal_inventory']=1
        self.assertEqual(self.scheduler.select(current,self.guard.history.snapshot())['reason'],'inactive_root')
        self.assertFalse(self.scheduler.state['root']['completed'])

    def test_new_generation_cannot_dispatch_an_old_installed_plan(self):
        self.scheduler.install(self.plan,self.packet,self.current)
        current=deepcopy(self.current);current['generation']+=1
        self.assertEqual(self.scheduler.select(current,self.guard.history.snapshot())['reason'],'stale_context')

    def test_payload_supports_free_concepts_but_not_reward_or_motor_commands(self):
        from minecraftrl.rm_planner import payload,RMPlanProvider,fixture_plan
        sent=[];response=fixture_plan(self.packet,unsupported=True)
        def transport(request,timeout):
            sent.append(json.loads(request.data))
            return {'status':'completed','output':[{'type':'message','content':[
                {'type':'output_text','text':json.dumps(response)}]}]}
        provider=RMPlanProvider('fixture','test-key',allow_paid=True,transport=transport,max_calls=1)
        self.assertEqual(provider.propose(self.packet)['steps'][0]['skill_id'],'find_obstacle_end')
        self.assertEqual(sent[0]['max_output_tokens'],2048);self.assertFalse(sent[0]['store'])
        with self.assertRaises(RuntimeError):provider.propose(self.packet)
        self.assertEqual(len(sent),1)

    def test_consumed_packet_and_changed_registry_do_not_call_transport(self):
        from minecraftrl.rm_planner import RMPlanProvider
        calls=[];provider=RMPlanProvider('fixture','test-key',allow_paid=True,transport=lambda *a,**kw:calls.append(1))
        packet=deepcopy(self.packet);packet['proposal_allowed']=False
        with self.assertRaises(ValueError):provider.propose(packet)
        packet=deepcopy(self.packet);packet['execution_registry']['explore_then_resume']['max_budget_ticks']=500
        with self.assertRaises(ValueError):provider.propose(packet)
        self.assertEqual(calls,[])

    def test_mailbox_broker_carries_full_plan_and_closes_each_request(self):
        from minecraftrl.recovery_exchange import FileRMPlanProvider
        from minecraftrl.recovery_broker import serve_once
        from minecraftrl.rm_planner import fixture_plan
        calls=[]
        class Provider:
            def propose(inner,packet):calls.append(packet);return fixture_plan(packet)
        exchange=FileRMPlanProvider(Path(self.tmp.name)/'exchange',max_calls=2)
        exchange.start(self.packet)
        report=serve_once(exchange.current,lambda:Provider(),Path(self.tmp.name)/'STOP')
        self.assertEqual(report['status'],'delivered');self.assertEqual(exchange.poll(),self.plan)
        previous=exchange.current
        next_packet=deepcopy(self.packet);next_packet['request_id']='a-new-request'
        exchange.start(next_packet)
        self.assertTrue((previous/'closed.json').exists())
        self.assertIsNone(exchange.poll());exchange.close()
        self.assertTrue((exchange.current/'closed.json').exists())
        self.assertEqual(len(calls),1)

    def test_same_request_cannot_consume_second_mailbox_call(self):
        from minecraftrl.recovery_exchange import FileRMPlanProvider
        exchange=FileRMPlanProvider(Path(self.tmp.name)/'exchange',max_calls=2)
        exchange.start(self.packet)
        with self.assertRaises(RuntimeError):exchange.start(self.packet)
        self.assertEqual(exchange.calls,1)

    def test_archived_unreleased_snapshot_can_be_analyzed_but_never_installed(self):
        from minecraftrl.rm_planner import payload,fixture_plan
        from minecraftrl.recovery_exchange import FileRMPlanProvider
        from minecraftrl.replanning import fingerprint
        packet=deepcopy(self.packet);packet['mode']='archived_review'
        packet['runtime_binding']['held_key_count']=1
        packet['runtime_binding']['release_sequence']=None
        packet['snapshot_sha256']=fingerprint(packet['runtime_binding'])
        self.assertIn('archived',payload(packet)['input'][0]['content'])
        with self.assertRaises(ValueError):
            self.scheduler.install(fixture_plan(packet),packet,packet['runtime_binding'])
        exchange=FileRMPlanProvider(Path(self.tmp.name)/'archive-exchange')
        with self.assertRaises(ValueError):exchange.start(packet)


if __name__=='__main__':unittest.main()
