import copy
import unittest
from minecraftrl.tasks import TaskState, validate_command


def command(op,**kw): return dict(command_id='test',task_id='g',op=op,**kw)
STEPS=[{'goal':'wood','count':2},{'goal':'dirt','count':3}]


class SequenceTests(unittest.TestCase):
    def test_runtime_switches_policy_in_same_world_and_releases_between_stages(self):
        import json,tempfile
        from pathlib import Path
        from minecraftrl.task_runtime import run_task_session
        from minecraftrl.tasks import CommandBus
        from test_task_session import FakeEnvironment
        env=FakeEnvironment(); bus=CommandBus(); bus.submit(command('request_plan',steps=STEPS))
        class Policy:
            def __init__(self): self.goals=[]
            def reset(self,goal): self.goal=goal; self.goals.append(goal)
            def act(self,frame): return 'oak_log' if self.goal=='collect wood' else 'dirt'
            def close(self): pass
        policy=Policy()
        with tempfile.TemporaryDirectory() as tmp:
            report=run_task_session(policy,env,tmp,seed=1,limit_s=2,bus=bus,save_screen=lambda *a:None)
            rows=[json.loads(x) for x in (Path(tmp)/'events.jsonl').read_text().splitlines()]
            transition=next(i for i,r in enumerate(rows) if r['event']=='plan_stage_completed')
            self.assertEqual(rows[transition+1]['event'],'input_release')
        self.assertEqual(report['status'],'task_completed')
        self.assertEqual(env.resets,1)
        self.assertEqual(policy.goals,['collect wood','collect dirt'])
        self.assertEqual(env.actions,['oak_log','oak_log','dirt','dirt','dirt'])
        self.assertEqual(len(report['task']['completed_stages']),2)

    def start(self):
        state=TaskState(); state.apply(command('request_plan',steps=STEPS),{'oak_log':5,'dirt':8}); return state

    def test_stages_use_entry_baseline_and_invalidate_old_revision(self):
        state=self.start(); event=state.observe({'oak_log':7,'dirt':10},1)
        self.assertEqual(event['event'],'plan_stage_completed')
        task=state.snapshot()
        self.assertEqual((task['stage_index'],task['goal'],task['baseline'],task['progress'],task['revision']),(1,'dirt',10,0,2))
        with self.assertRaises(ValueError): state.apply(command('pause',expected_revision=1),{})
        state.observe({'oak_log':7,'dirt':12},2); self.assertEqual(state.task['status'],'running')
        state.observe({'dirt':13},3); self.assertEqual(state.task['status'],'completed')
        self.assertEqual(len(state.task['completed_stages']),2)

    def test_pause_modify_remaining_cancel_and_restore(self):
        state=self.start(); state.observe({'oak_log':7,'dirt':10},1)
        state.apply(command('pause',expected_revision=2),{})
        restored=TaskState.restore(state.checkpoint('world'),'world')
        self.assertEqual(restored.snapshot(),state.snapshot())
        restored.apply(command('modify_plan',expected_revision=2,steps=[{'goal':'dirt','count':1}]),{'dirt':12})
        self.assertEqual((restored.task['status'],restored.task['stage_index'],len(restored.task['completed_stages'])),('paused',1,1))
        restored.apply(command('resume',expected_revision=3),{'dirt':12})
        restored.observe({'dirt':13},2); self.assertEqual(restored.task['status'],'completed')
        other=self.start(); other.apply(command('cancel',expected_revision=1),{})
        other.observe({'oak_log':100,'dirt':100},1); self.assertEqual(other.task['stage_index'],0)

    def test_current_stage_modify_preserves_tail_and_restore_pauses(self):
        state=self.start()
        state.apply(command('modify',expected_revision=1,goal='wood',count=1),{'oak_log':6})
        self.assertEqual(state.task['plan'][1],STEPS[1])
        self.assertEqual(state.task['plan'][0],{'goal':'wood','count':1})
        restored=TaskState.restore(state.checkpoint('w'),'w')
        self.assertEqual(restored.task['status'],'paused')
        restored.observe({'oak_log':99},1); self.assertEqual(restored.task['stage_index'],0)

    def test_invalid_plans_and_forged_checkpoints_fail_closed(self):
        for steps in [[],STEPS*5,[{'goal':'diamond','count':1}],[{'goal':'dirt','count':True}]]:
            with self.assertRaises(ValueError): validate_command(command('request_plan',steps=steps))
        state=self.start(); state.observe({'oak_log':7,'dirt':8},1)
        for mutate in [lambda x:x['task'].update(stage_index=5),
                       lambda x:x['task']['completed_stages'][0].update(progress=0),
                       lambda x:x['task']['plan'][1].update(count=64)]:
            value=copy.deepcopy(state.checkpoint('w')); mutate(value)
            with self.assertRaises(ValueError): TaskState.restore(value,'w')
