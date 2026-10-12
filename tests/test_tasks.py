import unittest
from minecraftrl import tasks


def command(op, cid, **kw):
    return dict(command_id=cid, op=op, task_id='gather', **kw)


class TaskTests(unittest.TestCase):
    def test_revision_baseline_and_observation_authority(self):
        state = tasks.TaskState()
        state.apply(command('request','a',goal='wood',count=4), {'oak_log':8,'dirt':3})
        state.observe({'oak_log':11,'dirt':5}, 1)
        self.assertEqual(state.snapshot()['status'], 'running')
        state.apply(command('modify','b',expected_revision=1,goal='dirt',count=4), {'oak_log':11,'dirt':5})
        self.assertEqual(state.snapshot()['revision'], 2)
        state.observe({'oak_log':50,'dirt':8}, 2)
        self.assertEqual(state.snapshot()['progress'], 3)
        self.assertEqual(state.snapshot()['status'], 'running')
        event = state.observe({'dirt':9}, 3)
        self.assertEqual(event['event'], 'inventory_target_reached')
        self.assertEqual(state.snapshot()['status'], 'completed')

    def test_pause_resume_preserves_progress_and_rejects_stale_commands(self):
        state = tasks.TaskState()
        state.apply(command('request','a',goal='dirt',count=4), {})
        state.observe({'dirt':2}, 1)
        state.apply(command('pause','b',expected_revision=1), {'dirt':2})
        self.assertEqual(state.snapshot()['status'], 'paused')
        with self.assertRaises(ValueError):
            state.apply(command('modify','c',expected_revision=0,goal='wood',count=4), {})
        state.apply(command('resume','d',expected_revision=1), {'dirt':2})
        state.observe({'dirt':4}, 2)
        self.assertEqual(state.snapshot()['status'], 'completed')

    def test_invalid_commands_do_not_replace_task(self):
        state = tasks.TaskState()
        for op, count in [('request',True),('request',0),('completed',4)]:
            with self.assertRaises(ValueError):
                state.apply(command(op,'bad',goal='dirt',count=count), {})
        self.assertIsNone(state.snapshot())

    def test_duplicate_command_is_idempotent_and_payload_conflict_rejected(self):
        bus=tasks.CommandBus(); request=command('request','a',goal='dirt',count=4)
        self.assertTrue(bus.submit(request)); token=bus.token()
        self.assertFalse(bus.submit(dict(request)))
        self.assertEqual(bus.token(),token)
        with self.assertRaises(ValueError): bus.submit(dict(request,count=5))
        self.assertEqual(len(bus.drain()),1)

    def test_non_string_goal_is_a_rejected_command(self):
        with self.assertRaises(ValueError):
            tasks.validate_command(command('request','a',goal=['dirt'],count=4))

    def test_generation_blocks_old_action_even_after_resume(self):
        bus = tasks.CommandBus()
        token = bus.token()
        bus.submit(command('pause','p',expected_revision=1))
        bus.drain()
        bus.submit(command('resume','r',expected_revision=1))
        bus.drain()
        calls=[]
        self.assertEqual(bus.dispatch(token, lambda:calls.append('old')), (False,None))
        self.assertEqual(calls, [])
        self.assertEqual(bus.dispatch(bus.token(), lambda:'new'), (True,'new'))


if __name__ == '__main__': unittest.main()
