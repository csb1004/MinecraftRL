import copy
import json
from pathlib import Path
import tempfile
import unittest

from minecraftrl.replanning import PlanHistory, ReplanGuard, goal_conditioning_capability
from test_replanning import snapshot, evidence, proposal, review


class PlanningMemoryTests(unittest.TestCase):
    def guard(self, path=None, embed='b'):
        cap = goal_conditioning_capability('policy-sha', 'a'*64, embed*64)
        return ReplanGuard(PlanHistory(path), {cap['skill_id']: cap}, allow_goal_conditioning=True)

    def attempt(self, g, outcome='failed', change=None, event_id='stall-200'):
        before = snapshot(status='stalled')
        before['facts']['target_reachable']=False
        r = g.request(before, dict(evidence(), event_id=event_id))
        cap = g.capabilities['explore_then_resume']
        p = proposal(r, skill_id=cap['skill_id'], arguments=cap['arguments'], evidence_ids=[event_id])
        self.assertTrue(review(g, r['request_id'], p, before)['accepted'])
        after = copy.deepcopy(before)
        after.update(sequence=220, tick=220, elapsed_s=27., release_sequence=None, held_key_count=None)
        if change: change(after)
        row = g.record_attempt(r['request_id'], before, after, outcome=outcome,
                               reason='budget_no_goal_progress', committed_steps=20)
        return before, after, row

    def next_request(self, g, s):
        s = copy.deepcopy(s);s.update(release_sequence=s['sequence'], held_key_count=0)
        e = dict(evidence(), event_id='stall-next', tick=s['tick'])
        r = g.request(s, e)
        return r, s

    def test_failed_execution_reaches_prompt_and_blocks_same_method(self):
        g = self.guard();before, after, row = self.attempt(g)
        r, current = self.next_request(g, after)
        packet = g.planner_input(r['request_id'])
        self.assertEqual(packet['attempts'][0]['outcome'], 'failed')
        self.assertEqual(packet['attempts'][0]['committed_steps'], 20)
        self.assertFalse(packet['call_model'])
        cap = g.capabilities['explore_then_resume']
        value = proposal(r, skill_id=cap['skill_id'], arguments=cap['arguments'],
                         evidence_ids=['stall-next'], reason='Different wording', budget_ticks=100)
        self.assertIn('same_failed_attempt', review(g, r['request_id'], value, current)['reasons'])

    def test_position_slot_and_generation_churn_do_not_clear_failure(self):
        g = self.guard();before, after, row = self.attempt(g)
        after['facts']['region'] = [100, 16, 100]
        after['facts']['selected_item'] = 'wooden_axe';after['generation'] += 1
        r, current = self.next_request(g, after)
        self.assertFalse(g.planner_input(r['request_id'])['call_model'])

    def test_changed_target_can_retry_but_returning_to_old_target_cannot(self):
        g = self.guard();before, after, row = self.attempt(g)
        after['facts']['target'] = [22,64,10]
        r, current = self.next_request(g, after)
        self.assertTrue(g.planner_input(r['request_id'])['call_model'])
        old = copy.deepcopy(before);old.update(tick=230, sequence=230, release_sequence=230, elapsed_s=28.)
        r2 = g.request(old, dict(evidence(), tick=230, event_id='returned'))
        self.assertFalse(g.planner_input(r2['request_id'])['call_model'])

    def test_partial_result_is_not_goal_success_and_prevents_identical_retry(self):
        g = self.guard();before, after, row = self.attempt(g, outcome='partial')
        self.assertEqual(row['outcome'], 'partial')
        r, current = self.next_request(g, after)
        self.assertFalse(g.planner_input(r['request_id'])['call_model'])

    def test_interrupted_execution_is_not_failed_method(self):
        g = self.guard();before, after, row = self.attempt(g, outcome='interrupted')
        r, current = self.next_request(g, after)
        self.assertTrue(g.planner_input(r['request_id'])['call_model'])

    def test_actual_arguments_distinguish_methods(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'history.json';g = self.guard(path)
            before, after, row = self.attempt(g)
            other = self.guard(path, embed='c');r, current = self.next_request(other, after)
            self.assertTrue(other.planner_input(r['request_id'])['call_model'])
            self.assertEqual(len(other.history.snapshot()['attempts']), 1)

    def test_attempt_restores_and_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'history.json';g = self.guard(path)
            before, after, row = self.attempt(g)
            restored = self.guard(path)
            self.assertEqual(restored.history.snapshot(), g.history.snapshot())
            same = restored.record_attempt(row['request_id'], before, after, outcome='failed',
                       reason='budget_no_goal_progress', committed_steps=20)
            self.assertEqual(same, row)
            with self.assertRaises(ValueError):
                restored.record_attempt(row['request_id'], before, after, outcome='partial',
                    reason='rewritten', committed_steps=20)

    def test_unexecuted_or_inventory_free_completion_not_accepted(self):
        g = self.guard();before = snapshot();r = g.request(before, evidence())
        with self.assertRaises(ValueError):
            g.record_attempt(r['request_id'], before, before, outcome='failed', reason='unreviewed', committed_steps=0)
        with self.assertRaises(ValueError):self.attempt(self.guard(), outcome='completed')

    def test_legacy_schema_is_read_without_mutating_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'history.json'
            original = json.dumps(dict(schema=1, revision=0, failures=[], requests=[], decisions=[]))
            path.write_text(original)
            h = PlanHistory(path)
            self.assertEqual(h.snapshot()['attempts'], [])
            self.assertEqual(path.read_text(), original)

    def test_existing_collection_failure_also_ignores_position_churn(self):
        from test_replanning import ReplanningTests
        g = ReplanningTests().guard();s = snapshot();g.record_failure(s, evidence(), 'collect_wood')
        s['facts']['region'][0] += 1
        r = g.request(s, evidence())
        self.assertFalse(g.planner_input(r['request_id'])['call_model'])

    def test_failure_reason_is_preserved_for_llm(self):
        from test_replanning import ReplanningTests
        g = ReplanningTests().guard();s = snapshot();g.record_failure(s, evidence(), 'collect_wood')
        r = g.request(s, evidence())
        self.assertEqual(g.planner_input(r['request_id'])['failed_attempts'][0]['evidence']['reason'],
                         'stationary_repetition')

    def test_observed_resources_or_reachability_allow_reconsideration(self):
        for field,value in [('resources',{'wooden_axe':1}),('target_reachable',True)]:
            g=self.guard();before,after,row=self.attempt(g)
            after['facts'][field]=value
            r,current=self.next_request(g,after)
            self.assertTrue(g.planner_input(r['request_id'])['call_model'])

    def test_stop_with_inventory_gain_still_preserves_interrupted_status(self):
        g=self.guard()
        before,after,row=self.attempt(g,outcome='interrupted',change=lambda s:s['facts'].update(goal_inventory=1))
        self.assertEqual(row['outcome'],'interrupted')

    def test_method_tampering_is_rejected_on_restore(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'history.json';g=self.guard(path);self.attempt(g)
            data=json.loads(path.read_text())
            data['attempts'][0]['method']['arguments']['explore_sha256']='c'*64
            data['decisions'][0]['method']['arguments']['explore_sha256']='c'*64
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):PlanHistory(path)

    def test_prompt_truncation_does_not_drop_duplicate_guard(self):
        from test_replanning import ReplanningTests
        g=ReplanningTests().guard();s=snapshot()
        for i in range(7):
            changed=copy.deepcopy(s);changed['facts']['target']=[18+i,64,10]
            g.record_failure(changed,dict(evidence(),event_id='failure-'+str(i)),'collect_wood')
        r=g.request(s,evidence());packet=g.planner_input(r['request_id'])
        self.assertEqual(len(packet['failed_attempts']),4)
        self.assertEqual(packet['omitted_failures'],3)
        self.assertFalse(packet['call_model'])

    def test_oversized_history_is_rejected_before_replacing_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'history.json';h=PlanHistory(path)
            h.commit(h.snapshot());original=path.read_bytes();data=h.snapshot()
            data['failures']=[{'large':'x'*20000} for _ in range(30)]
            with self.assertRaisesRegex(ValueError,'too large'):h.commit(data)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(PlanHistory(path).snapshot(),h.snapshot())

    def test_failure_event_keeps_method_arguments_and_blocks_the_same_variant(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'history.json';g=self.guard(path);s=snapshot()
            g.record_failure(s,evidence(),'explore_then_resume')
            r=g.request(s,evidence())
            packet=g.planner_input(r['request_id'])
            self.assertFalse(packet['call_model'])
            self.assertEqual(packet['failed_attempts'][0]['arguments'],g.capabilities['explore_then_resume']['arguments'])
            other=self.guard(path,embed='c')
            r2=other.request(s,dict(evidence(),event_id='changed-method'))
            self.assertTrue(other.planner_input(r2['request_id'])['call_model'])

    def test_collection_history_contract_also_applies_to_dirt(self):
        from minecraftrl.replanning import make_snapshot,collection_capability
        from test_replanning import obs
        s=make_snapshot(obs(),world_id='world',episode_id='dirt-episode',goal='dirt',task=None,
            generation=1,sequence=200,elapsed_s=25.,status='stalled',release_sequence=200,held_key_count=0)
        cap=collection_capability('dirt','dirt-policy')
        g=ReplanGuard(PlanHistory(),{'collect_dirt':cap})
        g.record_failure(s,evidence(),'collect_dirt');r=g.request(s,evidence())
        self.assertFalse(g.planner_input(r['request_id'])['call_model'])
        self.assertEqual(g.planner_input(r['request_id'])['failed_attempts'][0]['skill_id'],'collect_dirt')
