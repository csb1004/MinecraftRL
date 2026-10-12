import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from test_goal_recovery import RecordedActor
from test_planning_recovery import Game
from minecraftrl.waypoint_control import WaypointController
from minecraftrl.rl_collect import RLEnvironment
from minecraftrl.runtime import StopGate,run_episode


class WaypointControlTests(unittest.TestCase):
    def run_case(self,stop_inference=False,start_mode='stall',planner_mode='fixture'):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);out=Path(temp.name)
        gate=StopGate();p=WaypointController.__new__(WaypointController)
        class Student(RecordedActor):
            def act(self,screen):
                action=super().act(screen);self.pending={}
                if stop_inference and p.mode=='waypoint':gate.stop('test_inference_stop')
                return action
        student=Student();p.student=student;p.output=out;p.actor_path='fixture';p.actor_sha256='fixture'
        p.world=dict(sha256='fixture',chunk=[0,0],curriculum=dict(mirror_z=False,wall_end=6))
        p.head=SimpleNamespace(goal=None);p.rows=[];p.decisions=[];p.route=None;p.pending=None;p.ended=False
        p.mode='root';p.trigger=None;p.route_start=None;p.failure=None;p.switches=[]
        p.start_mode=start_mode
        p.planner_mode=planner_mode
        def change(game):
            if p.mode=='waypoint':
                target=p.route.targets[p.route.index];game.state.update(server_x=target[0],server_y=target[1],server_z=target[2])
            elif p.mode=='resumed':game.state['server_inventory']['oak_log']=1
        game=Game(change);env=RLEnvironment(game,p,strict_ticks=True,reward_mode='wood-potential-v4',is_stopped=gate.requested.is_set)
        p.environment=env;p.is_stopped=gate.requested.is_set
        report=run_episode(p,env,out,seed=1,goal='wood',limit_s=180,gate=gate,save_screen=lambda *a:None,
            completion_check=lambda obs:env.reward.completed)
        events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
        return p,student,game,report,events

    def test_real_stall_route_switch_and_root_resume_preserve_neural_actions(self):
        p,student,game,report,events=self.run_case()
        self.assertNotIn('error',report);self.assertEqual(report['status'],'target_reached')
        self.assertTrue(p.route.completed);self.assertEqual(len(p.route.receipts),4)
        self.assertIs(p.root_reward,p.environment.reward);self.assertEqual(student.resets,1)
        self.assertEqual(p.mode,'resumed');self.assertIsNone(p.head.goal)
        self.assertTrue(all(row['action']=={'attack':1,'camera':[0.,0.]} for row in p.rows))
        self.assertEqual(game.state['held_key_count'],0)

    def test_stop_during_first_waypoint_inference_discards_it(self):
        p,student,game,report,events=self.run_case(True)
        self.assertNotIn('error',report);self.assertEqual(report['status'],'cancelled')
        self.assertEqual(sum(r['mode']=='waypoint' for r in p.rows),0)
        self.assertEqual(len(student.inputs),len(p.rows)+1)
        self.assertTrue(any(e['event']=='action_discarded' for e in events))
        self.assertEqual(game.state['held_key_count'],0)
        with self.assertRaises(InterruptedError):p.act(None)

    def test_initial_diagnostic_uses_learned_actions_without_claiming_stall(self):
        p,student,game,report,events=self.run_case(start_mode='initial')
        self.assertEqual(report['status'],'target_reached')
        self.assertEqual(p.rows[0]['mode'],'waypoint')
        self.assertEqual(p.switches[0]['sequence'],0)
        self.assertEqual(p.condition_metadata['waypoint_start'],'initial')
        self.assertEqual(p.trigger['event'],'diagnostic_route_start')
        self.assertEqual(p.decisions,[])
        self.assertTrue(p.route.completed)
        self.assertIs(p.root_reward,p.environment.reward)
        self.assertEqual(game.state['held_key_count'],0)

    def test_visit_planner_never_calls_fixture_route(self):
        from unittest.mock import patch
        with patch('minecraftrl.waypoint_control.recovery_waypoints',side_effect=AssertionError('oracle accessed')):
            p,student,game,report,events=self.run_case(planner_mode='visit_memory')
        self.assertEqual(report['status'],'target_reached')
        self.assertEqual(len(p.route.receipts),3)
        self.assertIs(p.root_reward,p.environment.reward)
        self.assertTrue(p.trigger['stalled'])
        self.assertEqual(p.condition_metadata['waypoint_planner'],'visit_memory')
        self.assertNotIn('wood_target_x',p.route.decisions[0]['pose'])

    def test_stop_discards_first_blind_navigation_action(self):
        p,student,game,report,events=self.run_case(True,planner_mode='visit_memory')
        self.assertEqual(report['status'],'cancelled')
        self.assertEqual(sum(r['mode']=='waypoint' for r in p.rows),0)
        self.assertEqual(game.state['held_key_count'],0)


if __name__=='__main__':unittest.main()
