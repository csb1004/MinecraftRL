import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from test_planning_recovery import Actor,Game
from minecraftrl.adaptive_control import AdaptiveWaypointController
from minecraftrl.rl_collect import RLEnvironment
from minecraftrl.runtime import StopGate,run_episode
from minecraftrl.navigation_choice import validate_choice


class Provider:
    def __init__(self,gate,stop_when=None):self.calls=0;self.gate=gate;self.stop_when=stop_when;self.closed=False
    def start(self,packet):
        self.packet=packet;self.calls+=1
        if self.stop_when=='waiting':self.gate.stop('test_wait_stop')
    def poll(self):
        if self.stop_when=='reply':self.gate.stop('test_reply_stop')
        if self.stop_when=='provider_error':raise RuntimeError('test transport error')
        if self.stop_when=='stale_reply':return dict(request_id='other',candidate_index=1,reason='test')
        return validate_choice(dict(request_id=self.packet['request_id'],candidate_index=1,reason='test alternative'),self.packet)
    def close(self):self.closed=True


class AdaptiveControlTests(unittest.TestCase):
    def run_case(self,*,llm=False,stop_when=None,probe=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);out=Path(tmp.name);gate=StopGate()
        p=AdaptiveWaypointController.__new__(AdaptiveWaypointController)
        class Student(Actor):
            def act(self,screen):
                action=super().act(screen);self.pending={}
                if stop_when=='inference' and p.mode=='waypoint':gate.stop('test_inference_stop')
                return action
        p.student=Student();p.output=out;p.actor_path='fixture';p.actor_sha256='fixture'
        p.world=dict(sha256='fixture',chunk=[0,0],curriculum=dict(mirror_z=False,wall_end=6))
        p.head=SimpleNamespace(goal=None);p.rows=[];p.decisions=[];p.route=None;p.pending=None;p.ended=False
        p.mode='root';p.trigger=None;p.route_start=None;p.failure=None;p.switches=[];p.start_mode='stall'
        p.planner_mode='adaptive_llm' if llm else 'adaptive_rule';p.provider=Provider(gate,stop_when) if llm else None
        def change(game):
            if p.mode=='waypoint' and p.phase=='ready':
                target=p.route.targets[p.route.index]
                if not probe or not p.route.receipts:
                    game.state.update(server_x=target[0],server_y=target[1],server_z=target[2])
                elif not p.probes:
                    game.state.update(server_x=p.recovery_origin[0]+4,server_z=p.recovery_origin[2])
                if not probe:game.state.update(wood_v3_reachable=1,wood_v3_distance=1.)
            elif p.mode=='resumed':game.state['server_inventory']['oak_log']=1
            elif stop_when=='stale_pose' and getattr(p,'phase',None)=='refresh':game.state['server_x']+=1
        game=Game(change);env=RLEnvironment(game,p,strict_ticks=True,reward_mode='wood-potential-v4',is_stopped=gate.requested.is_set)
        p.environment=env;p.is_stopped=gate.requested.is_set
        report=run_episode(p,env,out,seed=1,goal='wood',limit_s=180,gate=gate,save_screen=lambda *a:None,completion_check=lambda obs:env.reward.completed)
        events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
        return p,game,report,events

    def test_observed_access_returns_to_same_root_rm(self):
        p,g,r,e=self.run_case();self.assertNotIn('error',r);self.assertEqual(r['status'],'target_reached')
        self.assertEqual(p.mode,'resumed');self.assertIs(p.root_reward,p.environment.reward)
        self.assertEqual(p.student.resets,1);self.assertEqual(p.switches[-1]['reason'],'wood_interaction_ready')

    def test_llm_choice_releases_refreshes_and_changes_only_goal(self):
        p,g,r,e=self.run_case(llm=True);self.assertNotIn('error',r);self.assertEqual(r['status'],'target_reached')
        neutrals=[r for r in p.rows if r['mode']=='planner_neutral']
        self.assertEqual(len(neutrals),2);self.assertTrue(all(r['server_after']['held_key_count']==0 for r in neutrals))
        self.assertEqual(p.route.decisions[0]['selected'],1);self.assertEqual(p.route.decisions[0]['source'],'llm')
        self.assertTrue(all(r['action']['attack']==1 for r in p.rows if r['mode']!='planner_neutral'))
        self.assertEqual(len(p.student.rows),len(p.rows));self.assertTrue(p.provider.closed)

    def test_stop_wait_reply_and_inference_send_no_navigation(self):
        for where in ('waiting','reply','inference'):
            with self.subTest(where=where):
                p,g,r,e=self.run_case(llm=True,stop_when=where);self.assertNotIn('error',r)
                self.assertEqual(r['status'],'cancelled');self.assertEqual(sum(x['mode']=='waypoint' for x in p.rows),0)
                self.assertEqual(g.state['held_key_count'],0)
                with self.assertRaises(InterruptedError):p.act(None)

    def test_escape_probe_is_bounded_and_does_not_claim_access(self):
        p,g,r,e=self.run_case(probe=True);self.assertNotIn('error',r)
        self.assertEqual(r['status'],'planning_paused');self.assertEqual(len(p.probes),1)
        self.assertEqual(sum(x['mode']=='root_probe' for x in p.rows),100)
        self.assertTrue(any(x.get('reason')=='root_probe_no_access' for x in p.switches))
        self.assertFalse(p.root_reward.completed);self.assertLessEqual(p.recovery_steps,400)

    def test_invalid_or_failed_provider_pauses_without_rule_fallback(self):
        for failure in ('provider_error','stale_reply','stale_pose'):
            with self.subTest(failure=failure):
                p,g,r,e=self.run_case(llm=True,stop_when=failure);self.assertNotIn('error',r)
                self.assertEqual(r['status'],'planning_paused')
                self.assertEqual(sum(x['mode']=='waypoint' for x in p.rows),0)
                self.assertEqual(g.state['held_key_count'],0)


if __name__=='__main__':unittest.main()
