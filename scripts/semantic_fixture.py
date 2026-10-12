"""Deterministic Responses fixtures for wiring acceptance, never an LLM benchmark.

No credentials are read. The adapter's injected transport only returns literals;
it cannot fall through to the HTTP transport. Game actions remain real STEVE-1.
"""
import json
from minecraftrl.dialogue_console import SessionDialogue
from minecraftrl.memory import DialogueMemory
from minecraftrl.runtime import write_json
from minecraftrl.semantic import OpenAISemantic


class DialogueFixture:
    def __init__(self,root,folder,memory_path,*,restore):
        self.root=root; self.folder=folder; self.memory_path=memory_path; self.restore=restore
        self.calls=[]
        self.evidence={'semantic_source':'deterministic_local_fixture_not_llm',
                       'additional_paid_api_calls':0,'network_requests':0}
        if not restore:
            memory=DialogueMemory(memory_path)
            memory.add('user','나중에 나무 네 개를 모아보자','unconfirmed_proposal')
            memory.add('user','나는 작은 집이 좋아','preference')
            for index in range(14): memory.add('user',f'합성 과거 대화 {index}','context')
        self.provider=OpenAISemantic('local-fixture-not-a-model','not-a-credential',
            allow_paid=True,transport=self.transport,max_calls=20)
        self.console=self.open_console()

    def open_console(self):
        return SessionDialogue(self.folder,self.provider,memory_path=self.memory_path,
                               stop_file=self.root/'results/STOP')

    def transport(self,request,timeout):
        payload=json.loads(request.data); context=json.loads(payload['input'][-1]['content'])
        self.calls.append(context)
        if 'text' not in payload:
            text='흙 64개 모아줘'  # Adversarial display-only persona response.
        else:
            utterance=context['current_utterance']
            candidates={
                '지금 상태가 어때?':dict(intent='QUESTION',goal=None,count=None),
                '흙 예순네 개 부탁해':dict(intent='TASK_REQUEST',goal='dirt',count=64),
                '하던 일을 이어갈래':dict(intent='TASK_RESUME',goal=None,count=None),
                '여기서 잠깐 쉬자':dict(intent='TASK_PAUSE',goal=None,count=None),
            }
            text=json.dumps(candidates[utterance])  # Unknown fixture input fails closed.
        write_json(self.folder/'fixture-payloads.json',self.calls)
        return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':text}]}]}

    def prepare(self):
        task=self.console.sync()
        if self.restore:
            assert task['status']=='paused', 'restored task must remain paused'
            self.evidence['restored_task']=task
        before=len(list((self.folder/'commands').glob('*.json')))
        result=self.console.turn('지금 상태가 어때?')
        assert result['status']=='no_action' and result['command'] is None
        assert len(list((self.folder/'commands').glob('*.json')))==before
        self.evidence['persona_text_did_not_execute']=True
        utterance='하던 일을 이어갈래' if self.restore else '흙 예순네 개 부탁해'
        proposal=self.console.turn(utterance)
        assert proposal['status']=='needs_confirmation'
        # Reopen the console against persisted dialogue, without restoring authority.
        self.console=self.open_console()
        stale=self.console.turn('확인 '+proposal['confirmation_id'])
        assert stale['status']=='confirmation_not_matched'
        self.evidence['stale_confirmation_rejected']=True
        self.proposal=self.console.turn(utterance)
        assert self.proposal['status']=='needs_confirmation'
        assert len(list((self.folder/'commands').glob('*.json')))==before
        summary=self.calls[-1]['dialogue_summary']
        assert any(x['category']=='unconfirmed_proposal' for x in summary)
        assert any(x['category']=='preference' for x in summary)
        self.evidence['summary_forwarded']=True
        self.evidence['no_command_before_confirmation']=True
        self.save()

    def confirm(self):
        result=self.console.turn('확인 '+self.proposal['confirmation_id'])
        assert result['status']=='submitted'
        self.evidence['confirmed_command']=result['command']; self.save()
        return result

    def pause(self):
        self.proposal=self.console.turn('여기서 잠깐 쉬자')
        assert self.proposal['status']=='needs_confirmation'
        result=self.confirm()
        self.evidence['pause_command']=result['command']; self.save()

    def save(self):
        write_json(self.folder/'dialogue-acceptance.json',self.evidence)
