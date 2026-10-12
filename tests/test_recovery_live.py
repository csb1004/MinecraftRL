import json
import unittest
import test_recovery_draft as fixtures
import test_replanning as snapshots


class RecoveryLiveTests(unittest.TestCase):
    def packet(self):
        from minecraftrl.replanning import PlanHistory,ReplanGuard,collection_capability,goal_conditioning_capability
        from minecraftrl.recovery_draft import planning_context
        packet=fixtures.RecoveryDraftTests().packet()
        packet['runtime_binding']=snapshots.snapshot(status='stalled')
        packet['root_reference']=packet['runtime_binding']['goal_ref']
        caps=[collection_capability('wood','policy'),goal_conditioning_capability('policy','a'*64,'b'*64)]
        g=ReplanGuard(PlanHistory(),{c['skill_id']:c for c in caps},allow_goal_conditioning=True)
        g.record_failure(packet['runtime_binding'],snapshots.evidence(),'collect_wood')
        request=g.request(packet['runtime_binding'],snapshots.evidence())
        packet.update(request_id=request['request_id'],evidence_id=request['evidence']['event_id'],
                      planner_context=planning_context(g,request['request_id']))
        return packet
    def test_live_request_is_advisory_and_uses_bound_current_evidence(self):
        from minecraftrl.recovery_live import LiveRecoveryProvider,LIVE_SCOPE
        sent=[]
        packet=self.packet();raw=fixtures.RecoveryDraftTests().reply()
        raw.update(request_id=packet['request_id'],evidence_id=packet['evidence_id'])
        response={'status':'completed','output':[{'type':'message','content':[
            {'type':'output_text','text':json.dumps(raw)}]}]}
        def transport(request,timeout):sent.append(json.loads(request.data));return response
        p=LiveRecoveryProvider('fixture','test-key',allow_paid=True,transport=transport,max_calls=1)
        reply=p.propose(packet)
        self.assertEqual(reply['scope'],LIVE_SCOPE);self.assertFalse(reply['execution_authorized'])
        self.assertIn('currently paused',sent[0]['input'][0]['content'])
        self.assertFalse(sent[0]['store']);self.assertEqual(sent[0]['max_output_tokens'],1024)
        supplied=json.loads(sent[0]['input'][1]['content'])
        self.assertEqual(supplied['planner_context']['failed_attempts'][0]['skill_id'],'collect_wood')
        with self.assertRaises(RuntimeError):p.propose(packet)
        self.assertEqual(len(sent),1)
    def test_unbound_or_completed_live_packet_never_calls_transport(self):
        from minecraftrl.recovery_live import LiveRecoveryProvider
        sent=[];p=LiveRecoveryProvider('fixture','test-key',allow_paid=True,transport=lambda *a,**k:sent.append(1),max_calls=1)
        for field,value in [('generation',None),('held_key_count',1),('observation_valid',False),('status','stopped')]:
            packet=self.packet();packet['runtime_binding'][field]=value
            with self.assertRaises(ValueError):p.propose(packet)
        packet=self.packet();packet['runtime_binding']['facts']['goal_inventory']=1
        with self.assertRaises(ValueError):p.propose(packet)
        self.assertEqual(sent,[]);self.assertEqual(p.remaining,1)

    def test_missing_memory_or_no_alternative_skips_transport(self):
        from minecraftrl.recovery_live import LiveRecoveryProvider
        sent=[];p=LiveRecoveryProvider('fixture','test-key',allow_paid=True,transport=lambda *a,**k:sent.append(1),max_calls=1)
        packet=self.packet();packet.pop('planner_context')
        with self.assertRaises(ValueError):p.propose(packet)
        packet=self.packet();packet['planner_context'].update(available_skills=[],available_choices=['remain_paused'],call_model=False)
        with self.assertRaises(ValueError):p.propose(packet)
        self.assertEqual(sent,[]);self.assertEqual(p.remaining,1)

    def test_memory_from_another_snapshot_is_rejected(self):
        from minecraftrl.recovery_live import validate_live_packet
        packet=self.packet();packet['runtime_binding']['facts']['target']=[999,64,10]
        with self.assertRaises(ValueError):validate_live_packet(packet)

    def test_blocked_response_is_rejected_independent_of_reason(self):
        from minecraftrl.recovery_draft import validate_draft
        packet=self.packet();packet['planner_context'].update(available_skills=[],available_choices=['remain_paused'],call_model=False)
        reply=fixtures.RecoveryDraftTests().reply();reply.update(request_id=packet['request_id'],evidence_id=packet['evidence_id'])
        with self.assertRaisesRegex(ValueError,'blocked'):validate_draft(reply,packet)
        reply['choice']='remain_paused'
        self.assertEqual(validate_draft(reply,packet)['recovery_budget_ticks'],0)
