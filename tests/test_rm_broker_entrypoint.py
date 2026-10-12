import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import test_recovery_live
from test_rm_planner import setup_packet


spec=importlib.util.spec_from_file_location('rm_broker_entrypoint',
    Path(__file__).resolve().parents[1]/'scripts/run_recovery_broker.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class RMBrokerEntrypointTests(unittest.TestCase):
    def test_full_plan_provider_and_fixture_use_matching_schema_and_one_call(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,packet,_=setup_packet(folder)
            reply=entry.fixture_reply(packet,'explore_then_resume',rm_plan=True);sent=[]
            def transport(request,timeout):
                sent.append(json.loads(request.data));self.assertEqual(timeout,20)
                return {'status':'completed','output':[{'type':'message','content':[
                    {'type':'output_text','text':json.dumps(reply)}]}]}
            provider=entry.make_api_provider('fixture','mock-key',rm_plan=True,transport=transport)
            self.assertEqual(provider.propose(packet),reply)
            self.assertEqual(sent[0]['max_output_tokens'],2048)
            self.assertEqual(sent[0]['text']['format']['name'],'minecraft_rm_plan')
            with self.assertRaises(RuntimeError):provider.propose(packet)
            self.assertEqual(len(sent),1)

    def test_pause_fixture_is_a_full_plan_pause_without_nodes(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,packet,_=setup_packet(folder)
            reply=entry.fixture_reply(packet,'remain_paused',rm_plan=True)
            self.assertEqual(reply['decision'],'pause');self.assertEqual(reply['steps'],[])

    def test_legacy_provider_keeps_old_schema_and_budget(self):
        packet=test_recovery_live.RecoveryLiveTests().packet();sent=[]
        reply=entry.fixture_reply(packet,'explore_then_resume',rm_plan=False)
        wire={k:reply[k] for k in ('request_id','evidence_id','choice','reason')}
        def transport(request,timeout):
            sent.append(json.loads(request.data))
            return {'status':'completed','output':[{'type':'message','content':[
                {'type':'output_text','text':json.dumps(wire)}]}]}
        provider=entry.make_api_provider('fixture','mock-key',rm_plan=False,transport=transport)
        self.assertEqual(provider.propose(packet)['choice'],'explore_then_resume')
        self.assertEqual(sent[0]['max_output_tokens'],1024)
        self.assertEqual(sent[0]['text']['format']['name'],'minecraft_live_recovery')

    def test_wrong_entrypoint_mode_rejects_packet_before_transport(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,packet,_=setup_packet(folder);sent=[]
            provider=entry.make_api_provider('fixture','mock-key',rm_plan=False,
                transport=lambda *args:sent.append(1))
            with self.assertRaises(ValueError):provider.propose(packet)
            self.assertEqual(sent,[])
