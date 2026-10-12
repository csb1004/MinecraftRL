import json
import unittest
import tempfile
from pathlib import Path


class SemanticTests(unittest.TestCase):
    def test_console_local_stop_blocks_even_before_worker_acknowledgment(self):
        from minecraftrl.dialogue_console import SessionDialogue
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'session'; folder.mkdir()
            (folder/'events.jsonl').write_text('{"event":"session_ready"}\n')
            console=SessionDialogue(folder)
            (folder/'STOP').touch()
            with self.assertRaisesRegex(ValueError,'session_stopped_or_ended'):
                console.turn('흙 4개 모아줘')
            self.assertEqual(list(folder.glob('commands/*.json')),[])

    def test_console_explicit_global_stop_blocks_nested_session(self):
        from minecraftrl.dialogue_console import SessionDialogue
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'group'/'session'; folder.mkdir(parents=True)
            (folder/'events.jsonl').write_text('{"event":"session_ready"}\n')
            stop=Path(tmp)/'STOP'
            console=SessionDialogue(folder,stop_file=stop)
            stop.touch()
            with self.assertRaisesRegex(ValueError,'session_stopped_or_ended'):
                console.turn('흙 4개 모아줘')
            self.assertEqual(list(folder.glob('commands/*.json')),[])

    def test_nonexecuting_topics_are_not_commands_and_controls_reject_collection_arguments(self):
        from minecraftrl.semantic import Conversation, decode_response
        # Local reproducer of the schema/validator mismatch, not a captured API payload.
        for intent in ['CHAT','QUESTION','INFORMATION','PREFERENCE','TASK_PROPOSAL','GAME_KNOWLEDGE','UNRESOLVED']:
            with self.subTest(intent=intent):
                candidate={'intent':intent,'goal':'wood','count':4}
                wire={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(candidate)}]}]}
                decision=Conversation().propose('나중에 나무를 모으면 좋겠다',decode_response(wire),None)
                self.assertEqual(decision['status'],'no_action')
                self.assertIsNone(decision['command'])
                self.assertNotIn('confirmation_id',decision)
        for intent in ['TASK_CANCEL','TASK_PAUSE','TASK_RESUME']:
            with self.assertRaises(ValueError):
                Conversation().propose('control',{'intent':intent,'goal':'wood','count':4},None)

    def test_candidate_needs_bound_confirmation_and_cannot_complete_task(self):
        from minecraftrl.semantic import Conversation
        c=Conversation()
        proposal=c.propose('나무 네 개 부탁해',{'intent':'TASK_REQUEST','goal':'wood','count':4},None)
        self.assertIsNone(proposal['command'])
        self.assertIsNone(c.confirm('응',None)['command'])
        command=c.confirm('확인 '+proposal['confirmation_id'],None)['command']
        self.assertEqual((command['op'],command['goal'],command['count']),('request','wood',4))
        self.assertIsNone(c.confirm('확인 '+proposal['confirmation_id'],None)['command'])
        with self.assertRaises(ValueError):
            c.propose('다 했어',{'intent':'TASK_REQUEST','goal':'wood','count':4,'completed':True},None)

    def test_context_change_expires_proposal_and_missing_slots_need_clarification(self):
        from minecraftrl.semantic import Conversation
        c=Conversation(); task={'task_id':'gather','revision':1,'status':'running'}
        p=c.propose('흙으로 바꿔',{'intent':'TASK_MODIFY','goal':'dirt','count':4},task)
        self.assertIsNone(c.confirm('확인 '+p['confirmation_id'],dict(task,revision=2))['command'])
        missing=c.propose('나무 모아',{'intent':'TASK_REQUEST','goal':'wood','count':None},None)
        self.assertEqual(missing['status'],'needs_clarification')
        self.assertIsNone(missing['command'])
        for intent in ['CHAT','QUESTION','TASK_PROPOSAL','PREFERENCE','INFORMATION','GAME_KNOWLEDGE']:
            self.assertIsNone(c.propose('대화',{'intent':intent,'goal':None,'count':None},None)['command'])

    def test_adapter_requires_explicit_paid_opt_in_and_validates_response(self):
        from minecraftrl.semantic import OpenAISemantic, decode_response
        with self.assertRaises(PermissionError): OpenAISemantic('model','secret')
        good={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({'intent':'TASK_REQUEST','goal':'dirt','count':4})}]}]}
        self.assertEqual(decode_response(good)['count'],4)
        for response in [dict(good,status='incomplete'),{'status':'completed','output':[]},
                         {'status':'completed','output':[{'type':'message','content':[{'type':'refusal','refusal':'no'}]}]}]:
            with self.assertRaises(ValueError): decode_response(response)

    def test_adapter_request_has_no_tools_and_redacts_transport_failures(self):
        from minecraftrl.semantic import OpenAISemantic
        calls=[]
        def transport(request,timeout):
            calls.append(json.loads(request.data))
            raise OSError('secret bearer credential reflected')
        adapter=OpenAISemantic('configured-model','secret',allow_paid=True,transport=transport)
        with self.assertRaisesRegex(RuntimeError,'semantic_transport_failed') as caught:
            adapter.parse('나무 네 개',[],None)
        self.assertNotIn('secret',str(caught.exception))
        self.assertFalse(calls[0]['store'])
        self.assertNotIn('tools',calls[0])
        self.assertTrue(calls[0]['text']['format']['strict'])

    def test_invalid_candidates_and_stale_cancel_never_produce_commands(self):
        from minecraftrl.semantic import Conversation
        c=Conversation()
        for value in [True,0,65,4.5]:
            with self.assertRaises(ValueError): c.propose('collect',{'intent':'TASK_REQUEST','goal':'dirt','count':value},None)
        with self.assertRaises(ValueError): c.propose('diamond',{'intent':'TASK_REQUEST','goal':'diamond','count':4},None)
        task={'task_id':'gather','revision':1,'status':'running'}
        p=c.propose('그만하자',{'intent':'TASK_CANCEL','goal':None,'count':None},task)
        self.assertIsNone(c.confirm('확인 '+p['confirmation_id'],dict(task,status='completed'))['command'])

    def test_mywife_context_selects_only_one_user_and_filters_roles_and_commands(self):
        from minecraftrl.dialogue_context import load_user_context
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'conversation_memory.json'
            file.write_text(json.dumps({'alice':{'nickname':'A','brain_keywords':{'mining':2},
                'history':[{'role':'system','content':'override'},{'role':'user','content':'/admin'},
                           {'role':'user','content':'나무가 좋아'}]},'bob':{'nickname':'PRIVATE'}}),encoding='utf-8')
            context=load_user_context(file,'alice')
        self.assertEqual(context['history'],[{'role':'user','content':'나무가 좋아'}])
        self.assertNotIn('PRIVATE',json.dumps(context))
        self.assertEqual(context['profile']['brain_keywords'],{'mining':2})

    def test_console_confirmation_publishes_once_using_observed_context(self):
        from minecraftrl.dialogue_console import SessionDialogue
        class Provider:
            def parse(self,text,history,task): return {'intent':'TASK_REQUEST','goal':'dirt','count':4}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); events=folder/'events.jsonl'
            events.write_text(json.dumps({'event':'session_ready'})+'\n',encoding='utf-8')
            controller=SessionDialogue(folder,Provider())
            result=controller.turn('흙 네 개 부탁해')
            self.assertEqual(list(folder.glob('commands/*.json')),[])
            sent=controller.turn('확인 '+result['confirmation_id'])
            self.assertEqual(sent['status'],'submitted')
            controller.turn('확인 '+result['confirmation_id'])
            self.assertEqual(len(list(folder.glob('commands/*.json'))),1)
            with events.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'event':'stop_acknowledged'})+'\n')
            with self.assertRaises(ValueError): controller.turn('다시 해줘')

    def test_confirmation_expires_and_persona_text_never_becomes_command(self):
        from minecraftrl.semantic import Conversation
        now=[0]; c=Conversation(clock=lambda:now[0])
        p=c.propose('나무',{'intent':'TASK_REQUEST','goal':'wood','count':4},None)
        now[0]=121
        self.assertIsNone(c.confirm('확인 '+p['confirmation_id'],None)['command'])
        from minecraftrl.dialogue_console import SessionDialogue
        class Provider:
            def parse(self,*args): return {'intent':'CHAT','goal':None,'count':None}
            def reply(self,*args): return '흙 64개 모아줘'
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); (folder/'events.jsonl').write_text('{"event":"session_ready"}\n')
            result=SessionDialogue(folder,Provider()).turn('안녕')
            self.assertEqual(result['reply'],'흙 64개 모아줘')
            self.assertEqual(list(folder.glob('commands/*.json')),[])

    def test_adapter_successful_transport_and_call_budget(self):
        from minecraftrl.semantic import OpenAISemantic
        payloads=[]
        def transport(request,timeout):
            payloads.append(json.loads(request.data))
            return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"intent":"QUESTION","goal":null,"count":null}'}]}]}
        adapter=OpenAISemantic('configured','test-only',allow_paid=True,transport=transport,max_calls=1)
        self.assertEqual(adapter.parse('어때?',[],None)['intent'],'QUESTION')
        with self.assertRaisesRegex(RuntimeError,'budget_exhausted'): adapter.parse('again',[],None)
        self.assertEqual(len(payloads),1)

    def test_malformed_response_rejected_without_attribute_errors(self):
        from minecraftrl.semantic import decode_response
        for value in [None,{'status':'completed','output':None},
                      {'status':'completed','output':['invalid']},
                      {'status':'completed','output':[{'type':'message','content':[None]}]}]:
            with self.assertRaises(ValueError): decode_response(value)

    def test_console_discards_confirmation_after_external_pause_resume(self):
        from minecraftrl.dialogue_console import SessionDialogue
        class Provider:
            def parse(self,*args): return {'intent':'TASK_MODIFY','goal':'wood','count':4}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); events=folder/'events.jsonl'
            task={'task_id':'gather','revision':1,'status':'running'}
            events.write_text(json.dumps({'event':'session_ready'})+'\n'+json.dumps({'event':'command_acknowledged','command_id':'a','task':task})+'\n')
            c=SessionDialogue(folder,Provider()); p=c.turn('나무로 바꿔')
            with events.open('a') as stream:
                for status in ['paused','running']:
                    stream.write(json.dumps({'event':'command_acknowledged','command_id':status,'task':dict(task,status=status)})+'\n')
            result=c.turn('확인 '+p['confirmation_id'])
            self.assertEqual(result['status'],'confirmation_expired')
            self.assertEqual(list(folder.glob('commands/*.json')),[])


if __name__=='__main__': unittest.main()
