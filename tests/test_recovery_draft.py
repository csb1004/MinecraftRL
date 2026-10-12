import importlib.util
import json
import unittest


class RecoveryDraftTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('minecraftrl.recovery_draft'))
        from minecraftrl import recovery_draft
        return recovery_draft

    def packet(self):
        return dict(request_id='probe-1',evidence_id='stall-603',root_goal={'goal':'wood','target_count':1},
            assessment={'classification':'stagnation_suspected','replan_candidate':True},
            status='stalled',remaining_control_s=100.)

    def reply(self,**patch):
        return dict(request_id='probe-1',evidence_id='stall-603',choice='explore_then_resume',reason='Try the bounded exploration condition.',**patch)

    def test_draft_keeps_root_and_has_no_execution_permission(self):
        result = self.module().validate_draft(self.reply(),self.packet())
        self.assertEqual(result['root_goal'],self.packet()['root_goal'])
        self.assertEqual(result['rm_effect'],'preserve'); self.assertFalse(result['execution_authorized'])
        self.assertEqual(result['recovery_budget_ticks'],200)

    def test_unsupported_goal_reward_action_and_wrong_identity_rejected(self):
        m = self.module()
        for field,value in (('reward',1),('goal','dirt'),('action','turn_right'),('choice','detour_right'),
                            ('request_id','stale'),('evidence_id','invented')):
            bad = self.reply(); bad[field] = value
            with self.assertRaises(ValueError): m.validate_draft(bad,self.packet())

    def test_stop_complete_and_no_stall_never_make_provider_request(self):
        m = self.module(); sent=[]
        provider=m.RecoveryDraftProvider('fixture','test-key',allow_paid=True,transport=lambda *a,**k: sent.append(a))
        for status in ('stopped','completed','cancelled','user_paused'):
            packet=self.packet();packet['status']=status
            with self.assertRaises(ValueError):provider.propose(packet)
        packet=self.packet();packet['assessment']['replan_candidate']=False
        with self.assertRaises(ValueError):provider.propose(packet)
        self.assertEqual(sent,[])

    def test_paid_default_refuses_before_any_transport(self):
        m = self.module()
        with self.assertRaises(PermissionError):m.RecoveryDraftProvider('fixture','test-key')

    def test_stubbed_response_preserves_request_and_budget(self):
        m = self.module(); captured=[]
        response={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(self.reply())}]}]}
        def transport(request,timeout): captured.append((json.loads(request.data),timeout));return response
        provider=m.RecoveryDraftProvider('fixture','test-key',allow_paid=True,transport=transport,max_calls=1)
        result=provider.propose(self.packet())
        self.assertEqual(result['choice'],'explore_then_resume');self.assertFalse(result['execution_authorized'])
        self.assertFalse(captured[0][0]['store']);self.assertEqual(captured[0][0]['max_output_tokens'],1024)
        self.assertEqual(captured[0][1],20)
        with self.assertRaises(RuntimeError):provider.propose(self.packet())
        self.assertEqual(len(captured),1)

    def test_pause_has_no_recovery_budget(self):
        m=self.module();reply=self.reply();reply['choice']='remain_paused'
        self.assertEqual(m.validate_draft(reply,self.packet())['recovery_budget_ticks'],0)

    def test_completion_evidence_overrides_stall_label_before_paid_request(self):
        m=self.module(); sent=[]
        response={'status':'completed','output':[{'type':'message','content':[
            {'type':'output_text','text':json.dumps(self.reply())}]}]}
        def transport(request,timeout): sent.append(request);return response
        provider=m.RecoveryDraftProvider('fixture','test-key',allow_paid=True,transport=transport,max_calls=1)
        packet=self.packet();packet['assessment']['goal_completed']=True
        with self.assertRaisesRegex(ValueError,'completed'):
            provider.propose(packet)
        self.assertEqual(sent,[]);self.assertEqual(provider.remaining,1)

    def test_execution_or_rm_replacement_claim_is_not_advisory_evidence(self):
        m=self.module()
        for field,value in (('execution_authorized',True),('rm_effect','replace')):
            with self.subTest(field=field):
                packet=self.packet();packet['assessment'][field]=value
                with self.assertRaises(ValueError):m.payload(packet)
                with self.assertRaises(ValueError):m.validate_draft(self.reply(),packet)
