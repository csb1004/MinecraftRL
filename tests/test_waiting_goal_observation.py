import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minecraftrl import task_runtime
from minecraftrl.runtime import StopGate
from minecraftrl.scheduled_tasks import ScheduledTasks
from minecraftrl.tasks import CommandBus
from test_task_session import FakeEnvironment


def enqueue(key,goal,priority=0,**kw):
    return dict(command_id='add-'+key,op='enqueue',task_id=key,goal=goal,count=1,priority=priority,**kw)


class WaitingGoalObservationTests(unittest.TestCase):
    def test_incidental_pickup_completes_waiting_goal_without_selecting_it(self):
        queue=ScheduledTasks(); bus=CommandBus(); gate=StopGate(); env=FakeEnvironment()
        high=enqueue('wood','wood',100);high['count']=64
        bus.submit(high);bus.submit(enqueue('dirt','dirt'))
        class Policy:
            calls=0
            def reset(self,goal): pass
            def act(self,screen):
                self.calls+=1
                if self.calls==2: gate.stop('test_complete')
                return 'dirt'
            def close(self): pass
        with tempfile.TemporaryDirectory() as folder, patch.object(task_runtime.time,'monotonic',return_value=0.):
            report=task_runtime.run_task_session(Policy(),env,folder,seed=1,limit_s=180,gate=gate,bus=bus,
                    scheduled_tasks=queue,save_screen=lambda *a:None)
            events=[json.loads(line) for line in (Path(folder)/'events.jsonl').read_text().splitlines()]
        self.assertEqual(report['goals']['dirt']['status'],'completed')
        self.assertEqual(report['goals']['wood']['status'],'running')
        receipts=[e for e in events if e['event']=='inventory_target_reached']
        self.assertEqual([e['task_id'] for e in receipts],['dirt'])
        self.assertFalse(any(e['event']=='goal_selected' and e['selection']['goal_id']=='dirt' for e in events))

    def test_one_observation_can_finish_two_goals_without_an_extra_action(self):
        queue=ScheduledTasks();bus=CommandBus();gate=StopGate()
        bus.submit(enqueue('wood','wood',100));bus.submit(enqueue('dirt','dirt'))
        class Environment(FakeEnvironment):
            def step(self,action):
                self.inventory['dirt']=self.inventory.get('dirt',0)+1
                return super().step(action)
        class Policy:
            calls=0
            def reset(self,goal): pass
            def act(self,screen):
                self.calls+=1
                if self.calls>1:gate.stop('unexpected_extra_inference')
                return 'oak_log'
            def close(self): pass
        policy=Policy();env=Environment()
        with tempfile.TemporaryDirectory() as folder, patch.object(task_runtime.time,'monotonic',return_value=0.):
            report=task_runtime.run_task_session(policy,env,folder,seed=1,limit_s=180,gate=gate,bus=bus,
                scheduled_tasks=queue,save_screen=lambda *a:None)
            events=[json.loads(line) for line in (Path(folder)/'events.jsonl').read_text().splitlines()]
        self.assertEqual(policy.calls,1)
        self.assertEqual(report['status'],'task_completed')
        self.assertEqual({e['task_id'] for e in events if e['event']=='inventory_target_reached'},{'wood','dirt'})

    def test_observation_does_not_auto_complete_paused_or_dependency_blocked_goal(self):
        queue=ScheduledTasks()
        queue.apply(enqueue('wood','wood',100),{})
        queue.apply(enqueue('dependent','dirt',depends_on=['wood']),{})
        queue.apply(enqueue('paused','dirt'),{})
        queue.apply(dict(command_id='pause',op='pause',task_id='paused',expected_revision=1),{})
        queue.select({});queue.observe({'dirt':1},1)
        self.assertEqual(queue.states['dependent'].task['status'],'running')
        self.assertEqual(queue.states['paused'].task['status'],'paused')


if __name__=='__main__': unittest.main()
