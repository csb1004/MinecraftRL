import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from test_goal_recovery import RecordedActor, gui
from minecraftrl.rl_collect import RLEnvironment
from minecraftrl.runtime import StopGate, run_episode


class Actor(RecordedActor):
    def features(self,screen):
        return np.zeros((2,1,8)), {'commits':self.agent.hidden_state['commits']+1}
    def commit(self,action,reward,terminal):
        super().commit(action,reward,terminal)
        self.rows[-1].update(env_action=deepcopy(action))


class Game:
    def __init__(self,change=None):
        self.raw=SimpleNamespace(action_space=SimpleNamespace(noop=lambda:{'attack':0,'camera':[0.,0.]}))
        self.state=gui();self.state.update(server_alive=1,selected_item='air')
        self.sent=[];self.change=change;self.closed=False
    def seed(self,seed):pass
    def reset(self):pass
    def step(self,action):
        self.sent.append(deepcopy(action));self.state['server_tick']+=1
        self.state['held_key_count']=int(bool(action['attack']))
        if self.change:self.change(self)
        obs=dict(pov=np.zeros((2,2,3),dtype=np.uint8),inventory=deepcopy(self.state['server_inventory']))
        return obs,0.,False,{'workshop':deepcopy(self.state)}
    def release(self):
        self.state['held_key_count']=0
        self.release_evidence={'verified':True,'held_key_count':0}
        return 'neutral_command_sent'
    def close(self):self.closed=True


class PlanningRecoveryTests(unittest.TestCase):
    def test_actual_provider_packet_contains_failed_methods_and_capabilities(self):
        p,a,g,r,events,provider=self.run_case()
        context=provider.packet['planner_context']
        self.assertEqual(context['request_id'],provider.packet['request_id'])
        self.assertEqual(context['failed_attempts'][0]['skill_id'],'collect_wood')
        self.assertEqual(context['available_choices'],['explore_then_resume','remain_paused'])
        self.assertEqual(len(p.guard.history.snapshot()['attempts']),1)
        self.assertEqual(p.guard.history.snapshot()['attempts'][0]['outcome'],'failed')

    def test_no_available_method_skips_provider_and_exploration(self):
        from unittest.mock import patch
        from minecraftrl.replanning import ReplanGuard
        original=ReplanGuard.planner_input
        def exhausted(guard,request_id):
            data=original(guard,request_id);data.update(available_skills=[],call_model=False)
            return data
        with patch.object(ReplanGuard,'planner_input',exhausted):
            p,a,g,r,events,provider=self.run_case()
        self.assertEqual(provider.calls,0);self.assertEqual(p.switches,[])
        self.assertEqual(r['status'],'planning_paused');self.assertEqual(g.state['held_key_count'],0)
        self.assertEqual(len(p.guard.history.snapshot()['attempts']),0)

    def run_case(self,choice='explore_then_resume',change=None,delay=.002,stop_wait=False,limit=180,
                 wait_limit=None,reply_fault=None,stop_reply=False,provider_factory=None,
                 continue_after_return=False,expire_after=None,stop_after=None,stop_recovery_inference=False):
        from minecraftrl.planning_recovery import PlanningRecoveryController, FixedRecoveryProvider
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);out=Path(tmp.name)
        embed=out/'explore.npy';np.save(embed,np.ones((1,512),dtype=np.float32))
        actor=Actor();actor.checkpoint='fixture-policy'
        game=Game(change);gate=StopGate()
        provider=provider_factory(out,gate) if provider_factory else FixedRecoveryProvider(choice,delay)
        if stop_wait:
            old_start=provider.start
            def start(packet):old_start(packet);gate.stop('test_stop_in_wait')
            provider.start=start
        if reply_fault or stop_reply:
            old_poll=provider.poll
            def poll():
                result=old_poll()
                if result is not None:
                    if stop_reply:gate.stop('test_stop_at_reply')
                    if reply_fault=='identity':result['request_id']='stale'
                    if reply_fault=='error':raise RuntimeError('provider test failure')
                return result
            provider.poll=poll
        p=PlanningRecoveryController(actor,embed,out,raw=game.raw,world_id='fixture-world',provider=provider)
        game.policy=p
        if stop_recovery_inference:
            old_act=actor.act
            def act(screen):
                action=old_act(screen)
                if p.mode=='intervention':gate.stop('test_stop_during_recovery_inference')
                return action
            actor.act=act
        prepare=p.prepare_control
        def prepare_with_boundary(**kwargs):
            count=sum(r['mode']=='intervention' for r in p.rows)
            if expire_after is not None and count>=expire_after:p.recovery_expires=0.
            if stop_after is not None and count>=stop_after:gate.stop('test_stop_in_recovery')
            return prepare(**kwargs)
        p.prepare_control=prepare_with_boundary
        if wait_limit is not None:p.wait_limit_s=wait_limit
        env=RLEnvironment(game,p,strict_ticks=True,is_stopped=gate.requested.is_set, reward_mode='wood-potential-v4')
        p.environment=env;p.is_stopped=gate.requested.is_set
        # Terminate after return to wood, keeping actual goal completion separate.
        def completion(obs):return env.reward.completed or (not continue_after_return and len(p.switches)==2)
        report=run_episode(p,env,out,seed=1,goal='collect wood',limit_s=limit,gate=gate,
            save_screen=lambda *a:None,completion_check=completion)
        events=[json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
        return p,actor,game,report,events,provider

    def test_failed_recovery_pauses_without_blind_return_or_second_request(self):
        p,a,g,r,events,provider=self.run_case()
        self.assertNotIn('error',r)
        neutral=[x for x in p.rows if x['owner']=='planner_neutral']
        self.assertEqual(len(neutral),2)
        self.assertTrue(all(x['server_after']['held_key_count']==0 for x in neutral))
        self.assertEqual(len(a.rows),len(p.rows));self.assertEqual(a.agent.hidden_state['commits'],len(p.rows))
        self.assertEqual(sum(x['mode']=='intervention' for x in p.rows),200)
        self.assertEqual(provider.calls,1);self.assertEqual(a.resets,1)
        self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(len(p.switches),1)
        self.assertEqual(len(a.inputs),400)
        self.assertEqual(p.attempts[0]['assessment']['classification'],'recovery_budget_exhausted')
        self.assertEqual(p.attempts[0]['next_phase'],'paused')
        self.assertFalse(p.environment.reward.completed)
        self.assertEqual(p.environment.reward.metadata['name'],'wood-potential-v4')
        self.assertEqual(g.state['held_key_count'],0)
        self.assertTrue(all(not x['loss_eligible'] for x in p.rows))
        self.assertEqual(sum(e['event']=='context_encoding_started' for e in events),2)
        self.assertEqual(sum(e['event']=='inference_started' for e in events),len(a.inputs))

    def test_current_interaction_resumes_wood_without_claiming_completion(self):
        def change(g):
            if len(g.sent)==270:
                g.state.update(wood_v3_reachable=1,wood_v3_distance=2.)
        p,a,g,r,events,provider=self.run_case(change=change)
        self.assertEqual(p.switches[-1]['reason'],'wood_interaction_ready')
        self.assertEqual(p.attempts[0]['assessment']['classification'],'interaction_ready')
        self.assertFalse(p.attempts[0]['assessment']['goal_completed'])
        self.assertEqual(p.attempts[0]['next_phase'],'resumed')
        self.assertEqual(sum(x['mode']=='intervention' for x in p.rows),8)

    def test_motion_and_historical_target_do_not_authorize_return(self):
        def change(g):
            if len(g.sent)>262:
                g.state['server_z']+=.1
                # Progress followed by losing the target is not a current handoff.
                g.state['wood_v3_distance']=3.
                if len(g.sent)>280:g.state['wood_v3_kind']=0
        p,a,g,r,events,provider=self.run_case(change=change)
        self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(len(p.switches),1)
        result=p.attempts[0]['assessment']
        self.assertTrue(result['same_target_approach_progress'])
        self.assertFalse(result['current_target_tracked'])
        self.assertFalse(result['goal_completed'])

    def test_approach_to_remembered_target_does_not_prove_fresh_visual_handoff(self):
        def change(g):
            if len(g.sent)>262:g.state['wood_v3_distance']=4.
        p,a,g,r,events,provider=self.run_case(change=change)
        self.assertEqual(p.attempts[0]['next_phase'],'paused')
        self.assertEqual(p.attempts[0]['assessment']['classification'],'recovery_incomplete_with_progress')
        self.assertEqual(len(p.switches),1)
        self.assertFalse(p.environment.reward.completed)

    def test_lost_approach_progress_does_not_resume_same_failed_state(self):
        def change(g):
            if len(g.sent)>262:g.state['wood_v3_distance']=4. if len(g.sent)<300 else 5.
        p,a,g,r,events,provider=self.run_case(change=change)
        self.assertEqual(r['status'],'planning_paused')
        self.assertTrue(p.attempts[0]['assessment']['same_target_approach_progress'])
        self.assertFalse(p.attempts[0]['assessment']['current_same_target_approach_progress'])
        self.assertEqual(len(p.switches),1)

    def test_stop_during_first_recovery_inference_never_commits_that_action(self):
        p,a,g,r,events,provider=self.run_case(stop_recovery_inference=True)
        self.assertEqual(r['status'],'cancelled')
        self.assertEqual(len(p.rows),202)
        self.assertEqual(len(a.inputs),201)
        self.assertFalse(p.switches[0]['committed'])
        self.assertEqual(p.attempts[0]['assessment']['classification'],'stopped')
        self.assertEqual(p.attempts[0]['committed_transitions'],0)
        self.assertEqual(g.state['held_key_count'],0)

    def test_recovery_wall_clock_expiry_is_consumed_before_next_inference(self):
        p,a,g,r,events,provider=self.run_case(expire_after=3)
        self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(len(a.inputs),203)
        self.assertEqual(p.attempts[0]['assessment']['boundary_reason'],'planning_time_budget')
        self.assertEqual(p.attempts[0]['assessment']['committed_ticks'],3)

    def test_new_current_target_allows_handoff_without_inventory_success(self):
        def change(g):
            if not g.policy.switches:g.state['wood_v3_kind']=0
            elif g.policy.mode=='intervention':g.state['wood_v3_kind']=1
        p,a,g,r,events,provider=self.run_case(change=change)
        result=p.attempts[0]['assessment']
        self.assertEqual(result['classification'],'target_acquired')
        self.assertTrue(result['current_target_tracked'])
        self.assertTrue(result['target_acquired_on_last_transition'])
        self.assertEqual(p.attempts[0]['next_phase'],'resumed')
        self.assertEqual(p.switches[-1]['reason'],'wood_target_acquired')
        self.assertEqual(sum(r['mode']=='intervention' for r in p.rows),1)
        self.assertFalse(result['goal_completed'])

    def test_expired_recovery_never_starts_first_neural_action(self):
        p,a,g,r,events,provider=self.run_case(expire_after=0)
        self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(p.switches,[])
        self.assertEqual(p.attempts,[])
        self.assertEqual(len(a.inputs),200)

    def test_stop_during_recovery_records_no_retry_and_dispatches_nothing_new(self):
        p,a,g,r,events,provider=self.run_case(stop_after=3)
        self.assertEqual(r['status'],'cancelled')
        self.assertEqual(len(a.inputs),203)
        self.assertEqual(p.attempts[0]['assessment']['classification'],'stopped')
        self.assertEqual(p.attempts[0]['next_phase'],'ended')
        self.assertEqual(g.state['held_key_count'],0)

    def test_actual_inventory_completion_during_recovery_wins(self):
        def change(g):
            if len(g.sent)==265:g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.run_case(change=change)
        self.assertEqual(r['status'],'target_reached')
        self.assertEqual(p.attempts[0]['assessment']['classification'],'goal_completed')
        self.assertEqual(p.attempts[0]['next_phase'],'completed')
        self.assertEqual(len(p.switches),1)

    def test_resumed_collection_restall_is_recorded_without_more_calls(self):
        def change(g):
            if len(g.sent)==270:g.state.update(wood_v3_reachable=1,wood_v3_distance=2.)
            if len(g.sent)>272:g.state.update(wood_v3_reachable=0,wood_v3_kind=0)
        p,a,g,r,events,provider=self.run_case(change=change,continue_after_return=True)
        self.assertEqual(r['status'],'planning_paused')
        self.assertTrue(any(e.get('reason')=='resumed_collection_stalled_request_budget_exhausted' for e in p.planner_events))
        self.assertEqual(provider.calls,1)
        self.assertEqual(len(p.switches),2)

    def test_persistent_root_stall_cannot_hide_resumed_skill_stall(self):
        def change(g):
            g.state['wood_v3_distance']=2.
            if len(g.sent)<=60:return
            if not g.policy.switches:
                g.state['wood_v3_kind']=0;g.state['server_z']+=.2
            else:
                g.state.update(wood_v3_kind=1,wood_v3_reachable=1)
            # A bounded fixture exit if the controller misses the second stall.
            if len(g.sent)>1500:g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.run_case(change=change,continue_after_return=True)
        self.assertEqual(r['status'],'planning_paused')
        self.assertEqual(len(p.switches),2)
        self.assertTrue(any(e['event']=='resumed_collection_stalled' for e in p.planner_events))
        self.assertEqual(provider.calls,1)

    def test_stop_during_wait_sends_nothing_after_neutral(self):
        p,a,g,r,events,provider=self.run_case(stop_wait=True)
        self.assertEqual(r['status'],'cancelled');self.assertEqual(p.switches,[])
        self.assertEqual(len(p.rows),201);self.assertEqual(len(a.inputs),200)
        self.assertTrue(provider.closed);self.assertTrue(g.closed)
        self.assertEqual(g.state['held_key_count'],0)

    def test_goal_completion_during_first_neutral_skips_provider(self):
        def change(g):
            if len(g.sent)==261:g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.run_case(change=change)
        self.assertEqual(r['status'],'target_reached');self.assertEqual(provider.calls,0)
        self.assertEqual(len(p.rows),201);self.assertEqual(p.switches,[])

    def test_goal_completion_at_refresh_wins_over_reply(self):
        def change(g):
            if len(g.sent)==262:g.state['server_inventory']['oak_log']=1
        p,a,g,r,events,provider=self.run_case(change=change)
        self.assertEqual(r['status'],'target_reached');self.assertEqual(provider.calls,1)
        self.assertEqual(p.switches,[]);self.assertEqual(len(p.rows),202)

    def test_changed_target_rejects_response(self):
        def change(g):
            if len(g.sent)==262:g.state['wood_v3_target_x']=12
        p,a,g,r,events,provider=self.run_case(change=change)
        self.assertEqual(r['status'],'planning_paused');self.assertEqual(p.switches,[])
        reviewed=next(e for e in p.planner_events if e['event']=='reviewed')
        self.assertIn('changed_preconditions',reviewed['verdict']['reasons'])

    def test_remain_paused_is_not_an_exploration_fallback(self):
        p,a,g,r,events,provider=self.run_case(choice='remain_paused')
        self.assertEqual(r['status'],'planning_paused');self.assertEqual(p.switches,[])
        self.assertEqual(len(a.inputs),200)

    def test_insufficient_remaining_time_does_not_request(self):
        p,a,g,r,events,provider=self.run_case(limit=40)
        self.assertEqual(r['status'],'planning_paused');self.assertEqual(provider.calls,0)
        self.assertEqual(p.switches,[])

    def test_unreleased_keys_never_request(self):
        def change(g):
            if len(g.sent)>260:g.state['held_key_count']=1
        p,a,g,r,events,provider=self.run_case(change=change)
        self.assertEqual(r['status'],'planning_paused');self.assertEqual(provider.calls,0)
        self.assertEqual(p.switches,[])

    def test_timeout_has_no_refresh_or_fallback(self):
        p,a,g,r,events,provider=self.run_case(delay=1,wait_limit=.025)
        self.assertEqual(r['status'],'planning_paused');self.assertEqual(len(p.rows),201)
        self.assertTrue(any(e.get('reason')=='planner_timeout' for e in p.planner_events))
        self.assertEqual(p.switches,[]);self.assertTrue(provider.closed)

    def test_bad_or_failed_reply_never_resumes(self):
        for fault in ('identity','error'):
            with self.subTest(fault=fault):
                p,a,g,r,events,provider=self.run_case(reply_fault=fault)
                self.assertEqual(r['status'],'planning_paused');self.assertEqual(len(p.rows),201)
                self.assertEqual(p.switches,[]);self.assertEqual(provider.calls,1)

    def test_stop_at_reply_prevents_refresh_dispatch(self):
        p,a,g,r,events,provider=self.run_case(stop_reply=True)
        self.assertEqual(r['status'],'cancelled');self.assertEqual(len(p.rows),201)
        self.assertEqual(p.switches,[]);self.assertEqual(len(a.inputs),200)

    def test_goal_conditioning_capability_requires_explicit_opt_in(self):
        from minecraftrl.replanning import PlanHistory,ReplanGuard,goal_conditioning_capability
        cap=goal_conditioning_capability('policy','a'*64,'b'*64)
        with self.assertRaises(ValueError):ReplanGuard(PlanHistory(),{cap['skill_id']:cap})
        ReplanGuard(PlanHistory(),{cap['skill_id']:cap},allow_goal_conditioning=True)

    def test_mailbox_broker_integration_preserves_neural_return_and_live_scope(self):
        import threading
        from minecraftrl.recovery_exchange import FileRecoveryProvider
        from minecraftrl.recovery_broker import serve_once
        from minecraftrl.recovery_draft import validate_draft
        from minecraftrl.recovery_live import LIVE_SCOPE
        for live in (False,True):
            with self.subTest(live=live):
                reports=[];threads=[]
                def factory(out,gate):
                    provider=FileRecoveryProvider(out/'exchange',live=live)
                    class BrokerFixture:
                        def propose(self,packet):
                            reply=validate_draft(dict(request_id=packet['request_id'],evidence_id=packet['evidence_id'],
                                choice='explore_then_resume',reason='Synthetic broker integration.'),packet)
                            if live:reply['scope']=LIVE_SCOPE
                            return reply
                    old_start=provider.start
                    def start(packet):
                        old_start(packet)
                        thread=threading.Thread(target=lambda:reports.append(serve_once(provider.path,BrokerFixture,out/'STOP',live=live)))
                        threads.append(thread);thread.start()
                    provider.start=start
                    return provider
                p,a,g,r,events,provider=self.run_case(provider_factory=factory)
                for thread in threads:thread.join(3);self.assertFalse(thread.is_alive())
                self.assertEqual(reports[0]['status'],'delivered')
                self.assertEqual(len(p.switches),1)
                self.assertEqual(r['status'],'planning_paused')
                self.assertEqual(sum(x['mode']=='intervention' for x in p.rows),200)
                self.assertTrue((provider.path/'closed.json').exists())
                metadata=p.condition_metadata['planning']
                self.assertEqual(metadata['provider'],'windows_mailbox')
                self.assertEqual(metadata['actual_llm_calls'],None if live else 0)
