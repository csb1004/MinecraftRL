import json
import contextlib
import io
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import Mock
from minecraftrl import companion
from minecraftrl.dialogue_console import SessionDialogue


class CompanionTests(unittest.TestCase):
    def test_natural_completion_is_recorded_before_stop_is_latched(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop=Path(tmp)/'STOP'; observed=[]
            proc=Mock(); proc.wait.side_effect=lambda **kwargs: observed.append(stop.exists())
            self.assertTrue(companion.finish_worker(proc,stop,natural_end=True))
            self.assertEqual(observed,[False])
            self.assertTrue(stop.exists())

    def test_manual_exit_requests_stop_before_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop=Path(tmp)/'STOP'; observed=[]
            proc=Mock(); proc.wait.side_effect=lambda **kwargs: observed.append(stop.exists())
            self.assertTrue(companion.finish_worker(proc,stop,natural_end=False))
            self.assertEqual(observed,[True])

    def test_status_and_exit_do_not_dispatch_game_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); (folder/'events.jsonl').write_text('{"event":"session_ready"}\n')
            inbox=queue.Queue(); inbox.put('/status'); inbox.put('/exit')
            proc=Mock(); proc.poll.return_value=None
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                companion.interact(SessionDialogue(folder),proc,folder/'STOP',inbox)
            self.assertIn('아직 목표가 없습니다',output.getvalue())
            self.assertFalse(list((folder/'commands').glob('*.json')))

    def test_status_explains_stall_and_history(self):
        text=companion.describe_task(dict(status='paused',goal='dirt',progress=1,target_count=3,
            completed_stages=[{}],failure={'reason':'no_progress'}))
        for expected in ('1/3','완료한 단계 1','정체','다시 계속해줘'): self.assertIn(expected,text)

    def test_completed_restored_task_accepts_new_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            task=dict(task_id='old',revision=1,status='completed',goal='dirt',progress=1,target_count=1)
            (folder/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in (
                {'event':'session_ready'},{'event':'world_restored','task':task}))+'\n')
            result=SessionDialogue(folder).turn('원목 2개 모아줘')
            self.assertEqual(result['status'],'submitted')
            command=json.loads(next((folder/'commands').glob('*.json')).read_text())
            self.assertEqual((command['op'],command['goal'],command['count']),('request','wood',2))

    def test_crafting_is_not_partially_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); (folder/'events.jsonl').write_text('{"event":"session_ready"}\n')
            result=SessionDialogue(folder).turn('원목 2개 모아서 작업대를 만들어줘')
            self.assertIsNone(result['command'])
            self.assertFalse(list((folder/'commands').glob('*.json')))

    def test_command_uses_bounded_policy_session_and_named_slot(self):
        command=companion.worker_command('/mnt/d/DATA/Desktop/MinecraftRL','Ubuntu-24.04','companion',104729,180,'results/use-test')
        self.assertIn('steve1',command)
        self.assertEqual(command[command.index('--world-slot')+1],'companion')
        self.assertEqual(command[command.index('--limit-s')+1],'180')
        self.assertNotIn('bash',command)
