import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minecraftrl.tasks import TaskState, CommandBus
from minecraftrl import task_runtime
from test_task_session import FakeEnvironment, cmd


class FailureRecoveryTests(unittest.TestCase):
    def test_watchdog_counts_only_new_high_water_progress_and_active_time(self):
        self.assertTrue(hasattr(task_runtime, 'ProgressWatchdog'))
        watch = task_runtime.ProgressWatchdog(10)
        task = dict(task_id='g', revision=1, status='running', progress=0)
        self.assertIsNone(watch.sample(task, 0))
        task['progress'] = 2
        self.assertIsNone(watch.sample(task, 9))
        task['progress'] = 1
        self.assertIsNone(watch.sample(task, 15))
        task['progress'] = 2
        self.assertEqual(watch.sample(task, 19), 10)
        task['status'] = 'paused'
        self.assertIsNone(watch.sample(task, 50))
        task['status'] = 'running'
        self.assertIsNone(watch.sample(task, 100))
        task['revision'] = 2
        self.assertIsNone(watch.sample(task, 109))
        self.assertEqual(watch.sample(task, 119), 10)

    def test_failure_keeps_completed_prefix_and_baseline_through_restore_resume(self):
        state = TaskState()
        state.apply(cmd('request_plan', 'a', steps=[{'goal':'wood','count':1}, {'goal':'dirt','count':3}]), {'dirt':4})
        state.observe({'oak_log':1,'dirt':4}, 1)
        state.observe({'oak_log':1,'dirt':5}, 2)
        self.assertTrue(hasattr(state, 'pause_for_no_progress'))
        state.pause_for_no_progress(60, 60)
        self.assertEqual((state.task['status'],state.task['revision'],state.task['baseline'],state.task['progress']), ('paused',3,4,1))
        saved = state.checkpoint('w')
        self.assertEqual(saved['schema'],3)
        restored = TaskState.restore(saved,'w')
        self.assertEqual(restored.snapshot(),state.snapshot())
        with self.assertRaises(ValueError): restored.apply(cmd('resume','old',expected_revision=2),{})
        restored.apply(cmd('resume','new',expected_revision=3),{})
        self.assertNotIn('failure',restored.task)
        restored.observe({'oak_log':1,'dirt':7},3)
        self.assertEqual(restored.task['status'],'completed')
        self.assertEqual(len(restored.task['completed_stages']),2)
        for change in [dict(reason='resource_unavailable'),dict(elapsed_s=float('nan')),dict(threshold_s=0),dict(observation_sequence=99)]:
            bad=copy.deepcopy(saved); bad['task']['failure'].update(change)
            with self.assertRaises(ValueError): TaskState.restore(bad,'w')

    def test_long_inference_is_discarded_on_stall_then_explicit_recovery_completes(self):
        # Virtual time avoids sleeps. release() acknowledges the pause and supplies a new command.
        bus=CommandBus(); bus.submit(cmd('request_plan','a',steps=[{'goal':'wood','count':1},{'goal':'dirt','count':1}]))
        clock=[0.0]
        class Env(FakeEnvironment):
            def release(self):
                result=super().release()
                if self.releases==1:
                    bus.submit(cmd('modify_plan','b',expected_revision=2,steps=[{'goal':'dirt','count':1}]))
                    bus.submit(cmd('resume','c',expected_revision=3))
                return result
        class Policy:
            def __init__(self): self.calls=0
            def reset(self,goal): pass
            def act(self,frame):
                self.calls+=1
                clock[0]+=11 if self.calls==1 else 1
                return 'dirt'
            def close(self): pass
        env=Env()
        with tempfile.TemporaryDirectory() as tmp, patch.object(task_runtime.time,'monotonic',side_effect=lambda:clock[0]):
            report=task_runtime.run_task_session(Policy(),env,tmp,seed=1,limit_s=50,bus=bus,save_screen=lambda *a:None,no_progress_s=10)
            rows=[json.loads(x) for x in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertEqual(report['status'],'task_completed')
        self.assertEqual(env.actions,['dirt'])
        self.assertEqual(sum(r['event']=='task_no_progress' for r in rows),1)
        self.assertTrue(any(r['event']=='action_discarded' and r['reason']=='no_progress' for r in rows))
        blocked=next(i for i,r in enumerate(rows) if r['event']=='task_no_progress')
        self.assertEqual(rows[blocked-1]['event'],'input_release')
        self.assertEqual(rows[blocked]['task']['status'],'paused')

    def test_console_sync_exposes_failure_and_invalidates_pending_confirmation(self):
        from minecraftrl.dialogue_console import SessionDialogue
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); events=folder/'events.jsonl'
            state=TaskState(); state.apply(cmd('request','a',goal='wood',count=3),{})
            events.write_text(json.dumps(dict(event='session_ready'))+'\n'+json.dumps(dict(event='command_acknowledged',command_id='a',task=state.snapshot()))+'\n')
            console=SessionDialogue(folder); old=console.sync(); before=old['_epoch']
            failure=dict(reason='no_progress',elapsed_s=60,threshold_s=60,observation_sequence=-1)
            paused=state.snapshot(); paused.update(status='paused',revision=2,failure=failure)
            with events.open('a') as stream: stream.write(json.dumps(dict(event='task_no_progress',task=paused))+'\n')
            current=console.sync()
            self.assertEqual(current['status'],'paused')
            self.assertEqual(current['failure'],failure)
            self.assertGreater(current['_epoch'],before)

    def test_cancel_queued_during_expired_inference_takes_precedence(self):
        bus=CommandBus(); bus.submit(cmd('request','a',goal='wood',count=1))
        clock=[0.0]
        class Policy:
            def reset(self,goal): pass
            def act(self,frame):
                clock[0]=11
                bus.submit(cmd('cancel','b',expected_revision=1))
                return 'oak_log'
            def close(self): pass
        env=FakeEnvironment()
        with tempfile.TemporaryDirectory() as tmp, patch.object(task_runtime.time,'monotonic',side_effect=lambda:clock[0]):
            report=task_runtime.run_task_session(Policy(),env,tmp,seed=1,limit_s=50,bus=bus,save_screen=lambda *a:None,no_progress_s=10)
        self.assertEqual(report['task']['status'],'cancelled')
        self.assertEqual(report['no_progress_events'],0)
        self.assertEqual(env.actions,[])

    def test_single_task_failure_checkpoint_and_cancel_preserve_evidence(self):
        state=TaskState(); state.apply(cmd('request','a',goal='dirt',count=3),{'dirt':4})
        state.observe({'dirt':5},0); state.pause_for_no_progress(60,60)
        state.apply(cmd('cancel','b',expected_revision=2),{})
        restored=TaskState.restore(state.checkpoint('w'),'w')
        self.assertEqual(restored.snapshot(),state.snapshot())
        self.assertEqual(restored.task['status'],'cancelled')
        for threshold in [True,0,-1,float('nan'),float('inf'),181]:
            with self.assertRaises(ValueError): task_runtime.ProgressWatchdog(threshold)


if __name__ == '__main__': unittest.main()
