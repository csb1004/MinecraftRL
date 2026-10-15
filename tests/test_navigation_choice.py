import json,tempfile,unittest
from pathlib import Path
from minecraftrl.navigation_choice import PaidBudget,choice_packet,validate_choice,validate_packet,CandidateMailbox,CandidateProvider
from minecraftrl.adaptive_route import AdaptiveRoute
from test_exploration_route import pose


class NavigationChoiceTests(unittest.TestCase):
    def folder(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);return Path(tmp.name)

    def test_budget_survives_restart_and_failure_without_retry(self):
        out=self.folder()
        for n in range(4):PaidBudget(out).reserve('request-'+str(n),1000)
        with self.assertRaises(RuntimeError):PaidBudget(out).reserve('fifth',1000)
        ledger=json.loads((out/'paid-budget.json').read_text())
        self.assertEqual(ledger['reserved_micro_usd'],1000000)
        self.assertEqual(len(ledger['calls']),4)

    def test_duplicate_concurrent_and_oversize_do_not_call(self):
        out=self.folder();b=PaidBudget(out)
        with self.assertRaises(ValueError):b.reserve('oversize',16385)
        b.reserve('one',20)
        with self.assertRaises(RuntimeError):b.reserve('one',20)
        (out/'paid-budget.lock').mkdir()
        with self.assertRaises(FileExistsError):b.reserve('two',20)
        self.assertEqual(len(json.loads((out/'paid-budget.json').read_text())['calls']),1)

    def test_choice_rejects_invented_or_blocked_or_stale(self):
        route=AdaptiveRoute(pose(),[]);packet=choice_packet(route,'episode:1',0)
        packet['candidates'][0]['admissible']=False
        reply=dict(request_id=packet['request_id'],candidate_index=1,reason='unvisited')
        self.assertEqual(validate_choice(reply,packet),reply)
        for fault in ({'candidate_index':0},{'candidate_index':4},{'candidate_index':True},{'request_id':'stale'},{'action':'jump'}):
            with self.assertRaises(ValueError):validate_choice(dict(reply,**fault),packet)
        for key in ('wood_target_x','wall_end'):
            with self.assertRaises(ValueError):validate_packet(dict(packet,**{key:1}))

    def test_mailbox_closes_and_reports_broker_failure(self):
        out=self.folder();m=CandidateMailbox(out/'exchange');route=AdaptiveRoute(pose(),[])
        m.start(choice_packet(route,'e:1',0));self.assertIsNone(m.poll())
        (m.current/'broker-result.json').write_text(json.dumps({'status':'failed'}))
        with self.assertRaises(RuntimeError):m.poll()
        m.close();self.assertIsNone(m.poll())
        with self.assertRaises(RuntimeError):m.start(choice_packet(route,'e:2',0))

    def test_provider_has_fixed_transport_scope_and_never_retries_failure(self):
        sent=[]
        def fail(request,timeout):
            sent.append(json.loads(request.data));self.assertEqual(timeout,20)
            self.assertEqual(request.full_url,'https://api.openai.com/v1/responses')
            raise TimeoutError('synthetic timeout')
        p=CandidateProvider('gpt-5.3-codex','test-only',allow_paid=True,max_calls=1,transport=fail)
        packet=choice_packet(AdaptiveRoute(pose(),[]),'e:1',0)
        for _ in range(2):
            with self.assertRaises(RuntimeError):p.propose(packet)
        self.assertEqual(len(sent),1);self.assertFalse(sent[0]['store']);self.assertNotIn('tools',sent[0])
        self.assertEqual(sent[0]['service_tier'],'default');self.assertEqual(sent[0]['max_output_tokens'],2048)


if __name__=='__main__':unittest.main()
