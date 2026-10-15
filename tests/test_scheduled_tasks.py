import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from minecraftrl import scheduled_tasks
from minecraftrl.tasks import TaskState, CommandBus
from minecraftrl import task_runtime
from test_task_session import FakeEnvironment


def enqueue(name, goal='wood', priority=50, **kw):
    return dict(command_id='add-'+name, task_id=name, op='enqueue', goal=goal,
                count=1, priority=priority, **kw)


class ScheduledTaskTests(unittest.TestCase):
    def queue(self):
        self.assertTrue(hasattr(scheduled_tasks, 'ScheduledTasks'), 'multi-goal runtime state missing')
        return scheduled_tasks.ScheduledTasks()

    def test_waiting_goal_selected_and_inventory_completion_preserves_other_goal(self):
        queue = self.queue()
        queue.apply(enqueue('wood', priority=0), {})
        queue.apply(enqueue('dirt', goal='dirt', priority=100), {})
        self.assertEqual(queue.select({})['goal_id'], 'dirt')
        queue.advance(60, service_s=60)
        self.assertEqual(queue.select({})['goal_id'], 'wood')
        queue.observe({'oak_log':1}, 1)
        self.assertEqual(queue.select({'oak_log':1})['goal_id'], 'dirt')
        self.assertEqual(queue.states['wood'].task['status'], 'completed')
        self.assertEqual(queue.states['dirt'].task['status'], 'running')

    def test_dependency_and_user_pause_survive_priority_changes(self):
        queue = self.queue()
        queue.apply(enqueue('wood', priority=0), {})
        queue.apply(enqueue('dirt', goal='dirt', priority=100, depends_on=['wood']), {})
        self.assertEqual(queue.select({})['goal_id'], 'wood')
        queue.advance(20, service_s=20)
        queue.apply(dict(op='pause', command_id='pause', task_id='wood', expected_revision=1), {})
        self.assertIsNone(queue.select({})['goal_id'])
        queue.apply(dict(op='reprioritize', command_id='priority', task_id='wood', expected_revision=1, priority=100), {})
        self.assertIsNone(queue.select({})['goal_id'])
        self.assertEqual(queue.ledger.records['wood']['age_s'], 20)

    def test_restore_requires_resume_preserves_clocks_and_original_versions(self):
        queue = self.queue()
        queue.apply(enqueue('wood'), {})
        queue.select({}); queue.advance(12, service_s=12)
        queue.apply(dict(op='modify', command_id='change', task_id='wood', expected_revision=1,
                         goal='wood', count=2), {})
        saved = queue.checkpoint('world')
        restored = TaskState.restore(saved, 'world')
        self.assertIsNone(restored.select({})['goal_id'])
        self.assertEqual(restored.ledger.records['wood']['age_s'], 12)
        self.assertEqual(restored.versions['wood'][0]['target_count'], 1)
        restored.apply(dict(op='resume', command_id='resume', task_id='wood', expected_revision=2), {})
        self.assertEqual(restored.select({})['goal_id'], 'wood')
        self.assertEqual(restored.task['target_count'], 2)
        with self.assertRaises(ValueError): TaskState.restore(saved, 'other-world')

    def test_stalled_goal_is_preserved_and_other_ready_goal_can_run(self):
        queue = self.queue()
        queue.apply(enqueue('wood', priority=100), {})
        queue.apply(enqueue('dirt', goal='dirt', priority=0), {})
        queue.select({}); queue.pause_for_no_progress(60, 60)
        self.assertEqual(queue.select({})['goal_id'], 'dirt')
        queue.advance(60, service_s=60)
        self.assertEqual(queue.states['wood'].task['failure']['reason'], 'no_progress')
        self.assertEqual(queue.select({})['goal_id'], 'dirt')

    def test_runtime_completes_both_goals_and_releases_between_policies(self):
        queue = self.queue()
        bus = CommandBus()
        bus.submit(enqueue('wood', priority=10))
        bus.submit(enqueue('dirt', goal='dirt', priority=90))
        env = FakeEnvironment()
        class Policy:
            def reset(self, goal): self.goal = goal
            def act(self, screen): return 'dirt' if self.goal == 'collect dirt' else 'oak_log'
            def close(self): pass
        with tempfile.TemporaryDirectory() as folder, patch.object(task_runtime.time,'monotonic',return_value=0.):
            report = task_runtime.run_task_session(Policy(), env, folder, seed=1, limit_s=1,
                bus=bus, save_screen=lambda *a:None, scheduled_tasks=queue)
            rows = [json.loads(s) for s in (Path(folder)/'events.jsonl').read_text().splitlines()]
        self.assertEqual(env.actions, ['dirt', 'oak_log'])
        self.assertEqual(report['status'], 'task_completed')
        self.assertEqual(report['completed_goal_count'], 2)
        self.assertTrue(any(r['event']=='input_release' and r['reason']=='goal_switch' for r in rows))
        self.assertEqual(queue.states['dirt'].task['status'], 'completed')

    def test_real_session_loop_serves_old_goal_before_high_priority_goal_finishes(self):
        queue=self.queue(); bus=CommandBus(); now=[0.]
        bus.submit(enqueue('wood',priority=0))
        busy=enqueue('dirt',goal='dirt',priority=100); busy['count']=64; bus.submit(busy)
        class Environment(FakeEnvironment):
            def step(self,action):
                now[0]+=20
                return super().step(action)
        class Policy:
            def reset(self,goal):self.goal=goal
            def act(self,screen):return 'dirt' if self.goal=='collect dirt' else 'oak_log'
            def close(self):pass
        env=Environment()
        with tempfile.TemporaryDirectory() as folder, patch.object(task_runtime.time,'monotonic',side_effect=lambda:now[0]):
            report=task_runtime.run_task_session(Policy(),env,folder,seed=1,limit_s=100,
                bus=bus,save_screen=lambda *a:None,scheduled_tasks=queue,no_progress_s=180)
        self.assertEqual(env.actions[:4],['dirt','dirt','dirt','oak_log'])
        self.assertEqual(report['completed_goal_count'],1)
        self.assertEqual(report['goals']['dirt']['status'],'running')

    def test_stop_during_inference_drops_action_with_another_goal_waiting(self):
        from minecraftrl.runtime import StopGate
        queue=self.queue(); bus=CommandBus(); gate=StopGate(); env=FakeEnvironment()
        bus.submit(enqueue('wood'));bus.submit(enqueue('dirt',goal='dirt'))
        class Policy:
            calls=0
            def reset(self,goal):pass
            def act(self,screen):self.calls+=1;gate.stop('user_stop');return 'oak_log'
            def close(self):pass
        policy=Policy()
        with tempfile.TemporaryDirectory() as folder, patch.object(task_runtime.time,'monotonic',return_value=0.):
            report=task_runtime.run_task_session(policy,env,folder,seed=1,limit_s=1,gate=gate,
                bus=bus,save_screen=lambda *a:None,scheduled_tasks=queue)
        self.assertEqual(env.actions,[])
        self.assertEqual(report['completed_goal_count'],0)
        self.assertGreater(env.releases,0)
        self.assertEqual(policy.calls,1)
        self.assertEqual(gate.reason,'user_stop')

    def test_priority_change_during_inference_drops_old_action(self):
        from minecraftrl.runtime import StopGate
        queue=self.queue(); bus=CommandBus(); gate=StopGate()
        class Environment(FakeEnvironment):
            def step(self,action):
                result=super().step(action);gate.stop('test_observation_complete');return result
        env=Environment()
        bus.submit(enqueue('wood',priority=100));bus.submit(enqueue('dirt',goal='dirt',priority=0))
        class Policy:
            calls=0
            def reset(self,goal):self.goal=goal
            def act(self,screen):
                self.calls+=1
                if self.calls==1:
                    bus.submit(dict(op='pause',command_id='p',task_id='wood',expected_revision=1))
                    bus.submit(dict(op='reprioritize',command_id='r',task_id='dirt',expected_revision=1,priority=100))
                return 'oak_log' if self.goal=='collect wood' else 'dirt'
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder, patch.object(task_runtime.time,'monotonic',return_value=0.):
            report=task_runtime.run_task_session(Policy(),env,folder,seed=1,limit_s=1,gate=gate,
                bus=bus,save_screen=lambda *a:None,scheduled_tasks=queue)
        self.assertEqual(env.actions,['dirt'])
        self.assertEqual(report['goals']['wood']['status'],'paused')

    def test_legacy_checkpoint_migration_and_blocked_repeated_restore(self):
        state=TaskState(); state.apply(dict(op='request',command_id='a',task_id='wood',goal='wood',count=1),{})
        state.pause_for_no_progress(60,60)
        queue=scheduled_tasks.ScheduledTasks.from_legacy(state)
        first=TaskState.restore(queue.checkpoint('world'),'world')
        second=TaskState.restore(first.checkpoint('world'),'world')
        self.assertIsNone(second.select({})['goal_id'])
        self.assertEqual(second.states['wood'].task['failure']['reason'],'no_progress')

    def test_time_spent_on_prerequisite_is_not_parent_motor_stagnation(self):
        queue=self.queue()
        queue.apply(enqueue('wood',priority=0),{})
        queue.apply(enqueue('dirt',goal='dirt',priority=100,depends_on=['wood']),{})
        queue.select({});queue.advance(60,service_s=60)
        queue.observe({'oak_log':1},1)
        self.assertEqual(queue.select({'oak_log':1})['goal_id'],'dirt')
        self.assertEqual(queue.stalled_s(),0)

    def test_accounting_error_still_releases_input_and_closes_environment(self):
        queue=self.queue();env=FakeEnvironment();bus=CommandBus()
        class Policy:
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder, patch.object(task_runtime.time,'monotonic',return_value=0.), patch.object(queue,'advance',side_effect=ValueError('bad timing')):
            report=task_runtime.run_task_session(Policy(),env,folder,seed=1,limit_s=.05,
                bus=bus,save_screen=lambda *a:None,scheduled_tasks=queue)
        self.assertEqual(report['status'],'error')
        self.assertGreater(env.releases,0)
        self.assertTrue(env.closed)

    def test_reprioritization_does_not_hide_previous_motor_stagnation(self):
        queue=self.queue();queue.apply(enqueue('wood'),{});queue.select({});queue.advance(50,service_s=50)
        queue.apply(dict(op='reprioritize',command_id='p',task_id='wood',expected_revision=1,priority=90),{})
        queue.observe({},1)
        self.assertEqual(queue.stalled_s(),50)

    def test_explicit_resume_keeps_failure_evidence_across_world_restore(self):
        queue=self.queue();queue.apply(enqueue('wood'),{});queue.select({})
        queue.pause_for_no_progress(60,60)
        queue.apply(dict(op='resume',command_id='resume',task_id='wood',expected_revision=2),{})
        saved=queue.checkpoint('world')
        self.assertIn('failures',saved,'resuming erased the saved failure evidence')
        restored=TaskState.restore(saved,'world')
        self.assertEqual(restored.failures['wood'][0]['reason'],'no_progress')

    def test_restored_completed_queue_waits_for_new_command_in_same_world(self):
        from minecraftrl.world_store import observation_evidence
        queue=self.queue();queue.apply(enqueue('old'),{});queue.select({});queue.observe({'oak_log':1},1)
        class Environment(FakeEnvironment):
            def observation(self):
                return dict(super().observation(),location_stats=dict(xpos=0.,ypos=64.,zpos=0.))
        env=Environment();env.inventory={'oak_log':1}
        bundle=dict(world_id='world',task_checkpoint=queue.checkpoint('world'),observation=observation_evidence(env.observation()))
        class LaterCommand(CommandBus):
            polls=0
            def drain(self):
                self.polls+=1
                if self.polls==2:self.submit(enqueue('new',goal='dirt'))
                return super().drain()
        class Policy:
            def reset(self,goal):pass
            def act(self,screen):return 'dirt'
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder, patch.object(task_runtime.time,'monotonic',return_value=0.):
            report=task_runtime.run_task_session(Policy(),env,folder,seed=1,limit_s=1,
                bus=LaterCommand(),save_screen=lambda *a:None,restore_bundle=bundle)
        self.assertEqual(env.actions,['dirt'])
        self.assertEqual(report['completed_goal_count'],2)
