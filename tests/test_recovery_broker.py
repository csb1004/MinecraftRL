import json
from pathlib import Path
import tempfile
import unittest

import test_recovery_draft as draft_fixture
import test_recovery_live as live_fixture
from minecraftrl.recovery_draft import validate_draft
from minecraftrl.recovery_exchange import FileRecoveryProvider


class RecoveryBrokerTests(unittest.TestCase):
    def setup_case(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        base=Path(temp.name);exchange=FileRecoveryProvider(base/'exchange')
        packet=live_fixture.RecoveryLiveTests().packet()
        exchange.start(packet)
        return base,exchange,packet

    @staticmethod
    def response(packet):
        value=draft_fixture.RecoveryDraftTests().reply()
        value.update(request_id=packet['request_id'],evidence_id=packet['evidence_id'])
        return validate_draft(value,packet)

    def test_proposal_crosses_mailbox_without_execution_authority(self):
        from minecraftrl.recovery_broker import serve_once
        base,exchange,packet=self.setup_case();calls=[]
        class Provider:
            def propose(self,value):
                calls.append(value)
                return RecoveryBrokerTests.response(value)
        report=serve_once(exchange.path,lambda:Provider(),base/'STOP')
        self.assertEqual(report['status'],'delivered');self.assertEqual(len(calls),1)
        self.assertFalse(exchange.poll()['execution_authorized'])
        with self.assertRaises(FileExistsError):serve_once(exchange.path,lambda:Provider(),base/'STOP')
        self.assertEqual(len(calls),1)

    def test_stop_before_request_processing_never_constructs_provider(self):
        from minecraftrl.recovery_broker import serve_once
        base,exchange,packet=self.setup_case();(base/'STOP').touch();calls=[]
        report=serve_once(exchange.path,lambda:calls.append(1),base/'STOP')
        self.assertEqual(report['status'],'stopped');self.assertEqual(calls,[])

    def test_late_completion_is_recorded_but_not_delivered(self):
        from minecraftrl.recovery_broker import serve_once
        base,exchange,packet=self.setup_case()
        class Provider:
            def propose(self,value):
                exchange.close()
                return RecoveryBrokerTests.response(value)
        report=serve_once(exchange.path,lambda:Provider(),base/'STOP')
        self.assertEqual(report['status'],'discarded_after_close')
        self.assertFalse((exchange.path/'response.json').exists())

    def test_missing_generation_and_neutral_proof_block_before_provider(self):
        from minecraftrl.recovery_broker import serve_once
        for field,value in [('generation',None),('held_key_count',1),('release_sequence',None)]:
            base,exchange,packet=self.setup_case();packet['runtime_binding'][field]=value
            (exchange.path/'request.json').write_text(json.dumps({'schema':1,'packet':packet}))
            calls=[]
            with self.assertRaises(ValueError):serve_once(exchange.path,lambda:calls.append(1),base/'STOP')
            self.assertEqual(calls,[])

    def test_provider_failure_consumes_single_request_and_is_not_retried(self):
        from minecraftrl.recovery_broker import serve_once
        base,exchange,packet=self.setup_case();calls=[]
        class Provider:
            def propose(self,value):calls.append(1);raise RuntimeError('test')
        report=serve_once(exchange.path,lambda:Provider(),base/'STOP')
        self.assertEqual(report['status'],'failed');self.assertEqual(calls,[1])
        self.assertFalse((exchange.path/'response.json').exists())
        with self.assertRaises(FileExistsError):serve_once(exchange.path,lambda:Provider(),base/'STOP')

    def test_request_expiring_during_provider_setup_is_not_sent(self):
        import os,time
        from minecraftrl.recovery_broker import serve_once
        base,exchange,packet=self.setup_case();calls=[]
        class Provider:
            def propose(self,value):calls.append(1);raise AssertionError('expired request sent')
        def factory():
            old=time.time()-21;os.utime(exchange.path/'request.json',(old,old))
            return Provider()
        report=serve_once(exchange.path,factory,base/'STOP')
        self.assertEqual(report['status'],'request_expired');self.assertEqual(calls,[])
