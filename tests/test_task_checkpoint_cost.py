import json
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minecraftrl import task_runtime
from minecraftrl.runtime import StopGate
from minecraftrl.scheduled_tasks import ScheduledTasks
from minecraftrl.tasks import CommandBus
from test_task_session import FakeEnvironment


class CheckpointCostTests(unittest.TestCase):
    def test_one_durable_checkpoint_per_unchanged_policy_step_preserves_observations(self):
        gate=StopGate(); bus=CommandBus(); queue=ScheduledTasks(); env=FakeEnvironment()
        bus.submit(dict(command_id='wood',op='enqueue',task_id='wood',goal='wood',count=64,priority=100))
        original=task_runtime.atomic_json; writes=[]
        def record(path, value):
            original(path,value); writes.append(value)
        class Policy:
            calls=0
            def reset(self,goal): pass
            def act(self,screen):
                self.calls+=1
                if self.calls==21: gate.stop('test_complete')
                # The previous observation must already be durable before the next inference.
                self_test.assertEqual(writes[-1]['last_observation'], len(env.actions))
                return 'stone'
            def close(self): pass
        self_test=self
        with tempfile.TemporaryDirectory() as folder, patch.object(task_runtime,'atomic_json',side_effect=record), \
                patch.object(task_runtime.time,'monotonic',side_effect=itertools.count(0., .001)):
            report=task_runtime.run_task_session(Policy(),env,folder,seed=1,limit_s=180,gate=gate,
                bus=bus,scheduled_tasks=queue,save_screen=lambda *a:None)
            saved=json.loads((Path(folder)/'task-checkpoint.json').read_text())
        self.assertEqual(report['steps'],20)
        self.assertEqual(saved['last_observation'],20)
        self.assertEqual(saved['goals']['wood']['task']['progress'],0)
        self.assertEqual(saved['priorities']['records']['wood'],report['priorities']['records']['wood'])
        self.assertLessEqual(len(writes),report['steps']+6,
            'Unchanged selection should not fsync a second full checkpoint per policy action')


if __name__=='__main__': unittest.main()
