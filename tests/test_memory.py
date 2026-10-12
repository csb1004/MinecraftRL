import json
from pathlib import Path
import tempfile
import unittest


class MemoryTests(unittest.TestCase):
    def test_recent_window_and_cumulative_summary_survive_restart(self):
        from minecraftrl.memory import DialogueMemory
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'conversation.json'; memory=DialogueMemory(path)
            memory.add('user','지형을 보존하고 싶어','preference')
            memory.add('user','나중에 집을 만들면 좋겠다','unconfirmed_proposal')
            for i in range(40): memory.add('user','대화 '+str(i),'context')
            restored=DialogueMemory(path)
            self.assertEqual(len(restored.history),12)
            self.assertEqual(restored.history[-1]['content'],'대화 39')
            notes=restored.summary
            self.assertTrue(any(x['category']=='preference' and '지형' in x['excerpt'] for x in notes))
            self.assertTrue(any(x['category']=='unconfirmed_proposal' for x in notes))
            self.assertTrue(all(x['source_id'] for x in notes))
            self.assertLessEqual(len(notes),24)
            self.assertNotIn('command',restored.export())

    def test_stale_writer_and_corrupt_memory_are_rejected(self):
        from minecraftrl.memory import DialogueMemory
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'conversation.json'; first=DialogueMemory(path); stale=DialogueMemory(path)
            first.add('user','first')
            with self.assertRaisesRegex(ValueError,'stale'): stale.add('user','lost update')
            self.assertEqual(DialogueMemory(path).history[-1]['content'],'first')
            path.write_text('{broken',encoding='utf-8')
            with self.assertRaises(ValueError): DialogueMemory(path)

    def test_checkpoint_restores_paused_and_requires_same_world(self):
        from minecraftrl.tasks import TaskState
        state=TaskState(); state.apply({'command_id':'a','task_id':'g','op':'request','goal':'dirt','count':4},{'dirt':5})
        state.observe({'dirt':7},1)
        checkpoint=state.checkpoint('world-a')
        restored=TaskState.restore(checkpoint,'world-a')
        self.assertEqual(restored.task['status'],'paused')
        self.assertEqual(restored.task['progress'],2)
        self.assertEqual(restored.task['baseline'],5)
        with self.assertRaises(ValueError): TaskState.restore(checkpoint,'world-b')
        restored.apply({'command_id':'b','task_id':'g','op':'resume','expected_revision':1},{'dirt':7})
        restored.observe({'dirt':9},2)
        self.assertEqual(restored.task['status'],'completed')

    def test_console_restart_retains_memory_but_never_restores_pending_confirmation(self):
        from minecraftrl.dialogue_console import SessionDialogue
        class Provider:
            def parse(self,*args,**kwargs): return {'intent':'TASK_REQUEST','goal':'wood','count':4}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); (folder/'events.jsonl').write_text('{"event":"session_ready"}\n')
            first=SessionDialogue(folder,Provider()); proposal=first.turn('나무 네 개 부탁해')
            second=SessionDialogue(folder,Provider())
            self.assertTrue(any('나무' in x['content'] for x in second.conversation.history))
            self.assertIsNone(second.turn('확인 '+proposal['confirmation_id'])['command'])
            self.assertEqual(list(folder.glob('commands/*.json')),[])

    def test_model_receives_summary_as_data_alongside_twelve_recent_messages(self):
        from minecraftrl.semantic import OpenAISemantic
        payload=[]
        def transport(request,timeout):
            payload.append(json.loads(request.data))
            return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"intent":"CHAT","goal":null,"count":null}'}]}]}
        client=OpenAISemantic('test','test',allow_paid=True,transport=transport)
        client.parse('안녕',[{'role':'user','content':str(i)} for i in range(12)],None,
                     summary=[{'category':'unconfirmed_proposal','excerpt':'집을 짓자','source_id':'old1'}])
        data=json.loads(payload[0]['input'][1]['content'])
        self.assertEqual(len(data['recent_dialogue']),12)
        self.assertEqual(data['dialogue_summary'][0]['category'],'unconfirmed_proposal')

    def test_runtime_writes_recoverable_checkpoint_with_final_inventory(self):
        from minecraftrl.task_runtime import run_task_session
        from minecraftrl.tasks import CommandBus, TaskState
        from test_task_session import FakeEnvironment
        class Policy:
            def reset(self,goal): pass
            def act(self,frame): return 'dirt'
            def close(self): pass
        bus=CommandBus(); bus.submit({'command_id':'a','task_id':'g','op':'request','goal':'dirt','count':2})
        with tempfile.TemporaryDirectory() as tmp:
            report=run_task_session(Policy(),FakeEnvironment(),tmp,seed=1,limit_s=2,bus=bus,save_screen=lambda *args:None)
            checkpoint=json.loads((Path(tmp)/'task-checkpoint.json').read_text(encoding='utf-8'))
            restored=TaskState.restore(checkpoint,checkpoint['world_id'])
            self.assertEqual(restored.task['progress'],2)
            self.assertEqual(restored.task['status'],'completed')
            self.assertEqual(restored.task,report['task'])

    def test_restart_does_not_resubmit_a_published_command_before_worker_queue(self):
        from minecraftrl.dialogue_console import SessionDialogue
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); (folder/'events.jsonl').write_text('{"event":"session_ready"}\n')
            SessionDialogue(folder).turn('흙 4개 모아줘')
            result=SessionDialogue(folder).turn('흙 4개 모아줘')
            self.assertEqual(result['status'],'awaiting_worker_ack')
            self.assertEqual(len(list(folder.glob('commands/*.json'))),1)

    def test_task_evidence_replay_preserves_superseded_goal_and_latest_progress(self):
        from minecraftrl.dialogue_console import SessionDialogue
        first={'task_id':'g','revision':1,'status':'running','goal':'wood','baseline':0,'progress':0,'target_items':['oak_log'],'target_count':4}
        second=dict(first,revision=2,goal='dirt',target_items=['dirt'],baseline=2)
        rows=[{'event':'session_ready'},
              {'event':'command_acknowledged','command_id':'a','task':first},
              {'event':'command_acknowledged','command_id':'b','task':second},
              {'event':'inventory_observed','task':second,'inventory':{'dirt':5},'observation_sequence':3}]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); (folder/'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            state=SessionDialogue(folder).sync()
            self.assertEqual(state['progress'],3)
            self.assertEqual(state['previous_goals'][0]['status'],'superseded')
            self.assertEqual(state['previous_goals'][0]['goal'],'wood')

    def test_shared_dialogue_memory_crosses_sessions_without_copying_task_or_confirmation(self):
        from minecraftrl.dialogue_console import SessionDialogue
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=root/'a'; b=root/'b'; a.mkdir(); b.mkdir()
            for folder in (a,b): (folder/'events.jsonl').write_text('{"event":"session_ready"}\n')
            memory=root/'user.json'
            first=SessionDialogue(a,memory_path=memory); first.turn('집이 예쁘면 좋겠어')
            second=SessionDialogue(b,memory_path=memory)
            self.assertTrue(any('집' in x['content'] for x in second.conversation.history))
            self.assertIsNone(second.sync())
            self.assertIsNone(second.conversation.pending)
            self.assertEqual(list(b.glob('commands/*.json')),[])

    def test_checkpoint_forged_goal_and_completion_are_rejected(self):
        from minecraftrl.tasks import TaskState
        state=TaskState(); state.apply({'command_id':'a','task_id':'g','op':'request','goal':'dirt','count':4},{})
        checkpoint=state.checkpoint('world')
        checkpoint['task']['policy_goal']='collect diamond'
        with self.assertRaises(ValueError): TaskState.restore(checkpoint,'world')
        checkpoint=state.checkpoint('world'); checkpoint['task']['status']='completed'
        with self.assertRaises(ValueError): TaskState.restore(checkpoint,'world')


if __name__=='__main__': unittest.main()
