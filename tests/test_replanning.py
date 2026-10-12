import copy
import json
from pathlib import Path
import tempfile
import unittest

from minecraftrl.replanning import PlanHistory, ReplanGuard, collection_capability, make_snapshot
from minecraftrl.tasks import TaskState


def obs():
    return dict(server_available=1,error=0,server_tick=200,server_x=10.,server_y=64.,server_z=10.,
                server_inventory={'oak_log':0},server_alive=1,selected_item='air',
                wood_v3_kind=1,wood_v3_target_x=18.,wood_v3_target_y=64.,wood_v3_target_z=10.)


def snapshot(**kwargs):
    values=dict(world_id='world',episode_id='episode',goal='wood',task=None,generation=1,
                sequence=200,elapsed_s=25.,status='running',release_sequence=200,held_key_count=0)
    values.update(kwargs)
    return make_snapshot(obs(),**values)


def evidence():
    return dict(event_id='stall-200',event='stalled',stalled=True,observation_valid=True,
                tick=200,reason='stationary_repetition')


def proposal(request,**kwargs):
    value=dict(request_id=request['request_id'],skill_id='collect_wood',arguments={},
               budget_s=30.,budget_ticks=200,evidence_ids=['stall-200'],reason='retry after change')
    value.update(kwargs);return value


def review(guard, request_id, value, current, now=None):
    return guard.review(request_id,value,current,now_elapsed_s=current['elapsed_s'] if now is None else now)


class ReplanningTests(unittest.TestCase):
    def test_monitor_float_key_count_is_normalized_without_truncation(self):
        self.assertEqual(snapshot(held_key_count=0.)['held_key_count'],0)
        self.assertIs(type(snapshot(held_key_count=0.)['held_key_count']),int)
        for value in (True,-1.,.5,float('nan'),float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):snapshot(held_key_count=value)

    def guard(self,path=None):
        return ReplanGuard(PlanHistory(path),{'collect_wood':collection_capability('wood','policy-sha')})

    def test_valid_review_never_dispatches(self):
        g=self.guard();s=snapshot();r=g.request(s,evidence())
        verdict=review(g, r['request_id'],proposal(r),s)
        self.assertTrue(verdict['accepted']);self.assertFalse(verdict['execution_authorized'])

    def test_same_failure_rejected_even_if_reason_changes(self):
        g=self.guard();s=snapshot();g.record_failure(s,evidence(),'collect_wood')
        r=g.request(s,evidence())
        verdict=review(g, r['request_id'],proposal(r,reason='a totally new description'),s)
        self.assertIn('same_failed_attempt',verdict['reasons'])

    def test_unsupported_detour_and_hidden_arguments_rejected(self):
        for patch,expected in [({'skill_id':'detour_right'},'unsupported_skill'),
                               ({'arguments':{'target':[20,64,10]}},'unsupported_arguments')]:
            g=self.guard();s=snapshot();r=g.request(s,evidence())
            self.assertIn(expected,review(g, r['request_id'],proposal(r,**patch),s)['reasons'])

    def test_unknown_reward_or_rm_fields_rejected(self):
        g=self.guard();s=snapshot();r=g.request(s,evidence());p=proposal(r);p['reward']=1
        self.assertIn('invalid_proposal',review(g, r['request_id'],p,s)['reasons'])

    def test_late_reply_after_stop_pause_cancel_or_completion(self):
        for status in ('stopped','user_paused','cancelled','completed','superseded'):
            g=self.guard();s=snapshot();r=g.request(s,evidence());current=copy.deepcopy(s);current['status']=status
            self.assertIn('inactive_goal',review(g, r['request_id'],proposal(r),current)['reasons'])

    def test_generation_change_invalidates_pause_resume_reply(self):
        g=self.guard();s=snapshot();r=g.request(s,evidence());current=copy.deepcopy(s);current['generation']+=1
        self.assertIn('stale_context',review(g, r['request_id'],proposal(r),current)['reasons'])

    def test_observation_counter_alone_does_not_expire_plan(self):
        g=self.guard();s=snapshot();r=g.request(s,evidence());current=copy.deepcopy(s)
        current['sequence']+=1;current['tick']+=1;current['elapsed_s']+=.1;current['release_sequence']+=1
        self.assertTrue(review(g, r['request_id'],proposal(r),current)['accepted'])

    def test_related_fact_change_requires_fresh_plan(self):
        g=self.guard();s=snapshot();r=g.request(s,evidence());current=copy.deepcopy(s);current['facts']['target']=[19,64,10]
        self.assertIn('changed_preconditions',review(g, r['request_id'],proposal(r),current)['reasons'])

    def test_missing_release_and_missing_evidence_rejected(self):
        g=self.guard();s=snapshot(release_sequence=None,held_key_count=None);r=g.request(s,evidence())
        v=review(g, r['request_id'],proposal(r,evidence_ids=['invented']),s)
        self.assertIn('inputs_not_released',v['reasons']);self.assertIn('unproven_evidence',v['reasons'])

    def test_request_budget_and_single_use_response(self):
        g=self.guard();s=snapshot()
        for i in range(3):
            e=evidence();e['event_id']='stall-'+str(i);r=g.request(s,e)
        with self.assertRaises(ValueError):g.request(s,dict(evidence(),event_id='fourth'))
        p=proposal(r,evidence_ids=[e['event_id']]);self.assertTrue(review(g, r['request_id'],p,s)['accepted'])
        self.assertIn('request_consumed',review(g, r['request_id'],p,s)['reasons'])

    def test_timeout_and_remaining_session_time(self):
        g=self.guard();s=snapshot();r=g.request(s,evidence());current=copy.deepcopy(s);current['elapsed_s']=80.
        self.assertIn('request_expired',review(g, r['request_id'],proposal(r),current)['reasons'])
        g=self.guard();s=snapshot(elapsed_s=170.);r=g.request(s,evidence())
        self.assertIn('budget_exceeds_remaining_time',review(g, r['request_id'],proposal(r),s)['reasons'])

    def test_failure_memory_and_consumed_response_survive_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'history.json';g=self.guard(path);s=snapshot()
            g.record_failure(s,evidence(),'collect_wood');r=g.request(s,evidence())
            review(g, r['request_id'],proposal(r),s);restored=self.guard(path)
            self.assertEqual(g.history.snapshot(),restored.history.snapshot())
            self.assertIn('request_consumed',review(restored, r['request_id'],proposal(r),s)['reasons'])

    def test_existing_task_progress_receipts_preserved(self):
        task=TaskState();task.apply(dict(command_id='start',task_id='task',op='request_plan',
             steps=[{'goal':'dirt','count':1},{'goal':'wood','count':3}]),{})
        task.observe({'dirt':1},1);task.observe({'dirt':1,'oak_log':1},2)
        before=task.checkpoint('world');s=snapshot(task=task.snapshot());g=self.guard();r=g.request(s,evidence())
        review(g, r['request_id'],proposal(r),s)
        self.assertEqual(before,task.checkpoint('world'))
        self.assertEqual(r['snapshot']['task']['completed_stages'],task.snapshot()['completed_stages'])

    def test_clock_expires_reply_even_with_unchanged_observation(self):
        g=self.guard();s=snapshot();r=g.request(s,evidence())
        self.assertIn('request_expired',review(g,r['request_id'],proposal(r),s,now=56.)['reasons'])

    def test_neutral_step_can_follow_stall_before_request(self):
        s=snapshot();s['tick']+=1;s['sequence']+=1;s['release_sequence']+=1
        g=self.guard();r=g.request(s,evidence())
        self.assertTrue(review(g,r['request_id'],proposal(r),s)['accepted'])

    def test_no_available_alternative_skips_model_call(self):
        g=self.guard();s=snapshot();g.record_failure(s,evidence(),'collect_wood');r=g.request(s,evidence())
        value=g.planner_input(r['request_id'])
        self.assertFalse(value['call_model']);self.assertEqual(value['available_skills'],[])
        self.assertEqual(value['blocked_reason'],'no_available_alternative')
        self.assertEqual(len(value['failed_attempts']),1)

    def test_material_change_allows_new_request_without_erasing_failure(self):
        g=self.guard();s=snapshot();g.record_failure(s,evidence(),'collect_wood')
        new=copy.deepcopy(s);new['facts']['target']=[22,64,10]
        r=g.request(new,evidence())
        self.assertTrue(review(g,r['request_id'],proposal(r),new)['accepted'])
        self.assertEqual(len(g.history.snapshot()['failures']),1)

    def test_stale_history_writer_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'history.json';first=self.guard(path);second=self.guard(path);s=snapshot()
            first.request(s,evidence());saved=path.read_bytes()
            with self.assertRaises(ValueError):second.request(s,evidence())
            self.assertEqual(path.read_bytes(),saved)

    def test_request_duplicate_does_not_reset_budget_or_expiry(self):
        g=self.guard();s=snapshot();first=g.request(s,evidence());second=g.request(s,evidence())
        self.assertEqual(first,second);self.assertEqual(len(g.history.snapshot()['requests']),1)

    def test_unknown_generation_never_passes_review(self):
        g=self.guard();s=snapshot(generation=None);r=g.request(s,evidence())
        self.assertIn('unproven_generation',review(g,r['request_id'],proposal(r),s)['reasons'])

    def test_invalid_numeric_budgets_are_consumed_and_rejected(self):
        for value in (True,float('nan'),-1.,0.):
            g=self.guard();s=snapshot();r=g.request(s,evidence())
            self.assertIn('invalid_proposal',review(g,r['request_id'],proposal(r,budget_s=value),s)['reasons'])
            self.assertIn('request_consumed',review(g,r['request_id'],proposal(r),s)['reasons'])

    def test_goal_switch_and_task_revision_change_rejected(self):
        task=TaskState();task.apply(dict(command_id='r',task_id='task',op='request',goal='wood',count=2),{})
        g=self.guard();s=snapshot(task=task.snapshot());r=g.request(s,evidence())
        task.apply(dict(command_id='m',task_id='task',op='modify',expected_revision=1,goal='wood',count=3),{})
        current=snapshot(task=task.snapshot())
        self.assertIn('stale_context',review(g,r['request_id'],proposal(r),current)['reasons'])

    def test_pause_in_task_cannot_be_overridden_by_snapshot_status(self):
        task=TaskState();task.apply(dict(command_id='r',task_id='task',op='request',goal='wood',count=2),{})
        g=self.guard();s=snapshot(task=task.snapshot());r=g.request(s,evidence())
        task.apply(dict(command_id='p',task_id='task',op='pause',expected_revision=1),{})
        current=snapshot(task=task.snapshot())
        self.assertIn('inactive_goal',review(g,r['request_id'],proposal(r),current)['reasons'])

    def test_satisfied_inventory_never_creates_planning_request(self):
        g=self.guard();s=snapshot();s['facts']['goal_inventory']=1
        with self.assertRaises(ValueError):g.request(s,evidence())
        self.assertEqual(g.history.snapshot()['requests'],[])

    def test_forged_unconsumed_response_receipt_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'history.json';g=self.guard(path);s=snapshot();r=g.request(s,evidence())
            review(g,r['request_id'],proposal(r),s)
            value=json.loads(path.read_text());value['requests'][0]['consumed']=False
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):PlanHistory(path)

    def test_request_payload_and_verdict_cannot_mutate_task_or_history(self):
        g=self.guard();s=snapshot();r=g.request(s,evidence());before=g.history.snapshot()
        r['snapshot']['facts']['target'][0]=999
        payload=g.planner_input(r['request_id']);payload['snapshot']['facts']['target'][0]=888
        self.assertEqual(g.history.snapshot(),before)

    def test_internal_pause_revision_does_not_erase_failed_method(self):
        task=TaskState();task.apply(dict(command_id='r',task_id='task',op='request',goal='wood',count=2),{})
        g=self.guard();before=snapshot(task=task.snapshot());g.record_failure(before,evidence(),'collect_wood')
        task.pause_for_no_progress(20.,10.)
        current=snapshot(task=task.snapshot(),status='stalled',generation=2)
        request=g.request(current,evidence())
        self.assertIn('same_failed_attempt',review(g,request['request_id'],proposal(request),current)['reasons'])
        self.assertEqual(len(g.planner_input(request['request_id'])['failed_attempts']),1)

    def test_request_budget_survives_process_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'history.json';g=self.guard(path);s=snapshot()
            for i in range(3):g.request(s,dict(evidence(),event_id='budget-'+str(i)))
            restored=self.guard(path)
            with self.assertRaises(ValueError):restored.request(s,dict(evidence(),event_id='over-budget'))

    def test_world_change_and_old_release_receipt_rejected(self):
        for field,value,reason in [('world_id','different-world','stale_context'),
                                   ('release_sequence',199,'inputs_not_released'),
                                   ('held_key_count',1,'inputs_not_released')]:
            g=self.guard();s=snapshot();r=g.request(s,evidence());current=copy.deepcopy(s);current[field]=value
            self.assertIn(reason,review(g,r['request_id'],proposal(r),current)['reasons'])

    def test_registered_skill_cannot_change_goal(self):
        g=ReplanGuard(PlanHistory(),{'collect_dirt':collection_capability('dirt','dirt-policy')})
        s=snapshot();r=g.request(s,evidence())
        self.assertIn('goal_mismatch',review(g,r['request_id'],proposal(r,skill_id='collect_dirt'),s)['reasons'])

    def test_policy_change_after_request_requires_new_review(self):
        g=self.guard();s=snapshot();r=g.request(s,evidence())
        g.capabilities['collect_wood']=collection_capability('wood','new-policy')
        self.assertIn('capability_changed',review(g,r['request_id'],proposal(r),s)['reasons'])

    def test_future_or_old_stall_evidence_rejected(self):
        for tick in (201,195,float('nan')):
            with self.assertRaises(ValueError):self.guard().request(snapshot(),dict(evidence(),tick=tick))


if __name__=='__main__':unittest.main()
