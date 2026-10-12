import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import run_inplace_recovery as experiment
from minecraftrl import task_runtime
from minecraftrl.tasks import TaskState, CommandBus
from test_task_session import FakeEnvironment, cmd


class InplaceRecoveryTests(unittest.TestCase):
    def test_experiment_retries_only_one_observed_stall(self):
        self.assertTrue(hasattr(experiment,'OneRetry'))
        budget=experiment.OneRetry()
        state=TaskState(); state.apply(cmd('request','a',goal='dirt',count=2),{'dirt':4})
        self.assertIsNone(budget.on_stall(state.snapshot()))
        state.apply(cmd('pause','p',expected_revision=1),{})
        self.assertIsNone(budget.on_stall(state.snapshot()))
        state.apply(cmd('resume','r',expected_revision=1),{})
        state.observe({'dirt':5},0); state.pause_for_no_progress(60,60)
        before=state.snapshot(); command=budget.on_stall(before)
        self.assertEqual(command['op'],'resume')
        state.apply(command,{'dirt':5}); experiment.assert_preserved(before,state.snapshot())
        bad=state.snapshot(); bad['baseline']=5
        with self.assertRaises(ValueError): experiment.assert_preserved(before,bad)
        state.pause_for_no_progress(60,60)
        self.assertIsNone(budget.on_stall(state.snapshot()))

    def test_runtime_resets_same_policy_without_losing_partial_or_completed_stages(self):
        bus=CommandBus(); bus.submit(cmd('request_plan','a',steps=[{'goal':'dirt','count':1},{'goal':'wood','count':2}]))
        clock=[0.0]
        class Env(FakeEnvironment):
            def release(self):
                result=super().release()
                if self.releases==2: bus.submit(cmd('resume','retry',expected_revision=3))
                return result
        class Policy:
            def __init__(self): self.calls=0; self.goals=[]; self.closed=False
            def reset(self,goal): self.goal=goal; self.goals.append(goal)
            def act(self,frame):
                self.calls+=1; clock[0]+=11 if self.calls==3 else 1
                return 'dirt' if self.goal=='collect dirt' else 'oak_log'
            def close(self): self.closed=True
        policy=Policy(); env=Env()
        with tempfile.TemporaryDirectory() as tmp, patch.object(task_runtime.time,'monotonic',side_effect=lambda:clock[0]):
            report=task_runtime.run_task_session(policy,env,tmp,seed=1,limit_s=50,no_progress_s=10,bus=bus,save_screen=lambda *a:None)
            rows=[json.loads(x) for x in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertEqual(report['status'],'task_completed')
        self.assertEqual(report['worker_pid'],os.getpid())
        self.assertEqual({r['worker_pid'] for r in rows if r['event']=='policy_reset_finished'},{os.getpid()})
        self.assertEqual(env.resets,1)
        self.assertEqual(policy.goals,['collect dirt','collect wood','collect wood'])
        self.assertEqual(env.actions,['dirt','oak_log','oak_log'])
        blocked=next(r['task'] for r in rows if r['event']=='task_no_progress')
        resumed=next(r['task'] for r in rows if r['event']=='command_acknowledged' and r['command_id']=='retry')
        self.assertEqual((blocked['progress'],len(blocked['completed_stages'])),(1,1))
        for key in ('goal','baseline','progress','plan','completed_stages','stage_index'):
            self.assertEqual(blocked[key],resumed[key])
        self.assertEqual(len(report['task']['completed_stages']),2)
        self.assertTrue(policy.closed)
