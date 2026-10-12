import json
from pathlib import Path
import tempfile
import time
import unittest

from minecraftrl.tasks import CommandBus, TaskState
from minecraftrl.task_runtime import DirectoryInbox


def message(text, mid='utterance1', revision=None):
    value={'kind':'dialogue','message_id':mid,'text':text,'task_id':'gather'}
    if revision is not None: value['expected_revision']=revision
    return value


class DialogueTests(unittest.TestCase):
    def decision(self, text, revision=None):
        from minecraftrl import dialogue
        return dialogue.decide(message(text,revision=revision))

    def test_explicit_collection_and_controls_compile_to_existing_commands(self):
        request=self.decision('지금 원목 4개 모아줘')['command']
        self.assertEqual(request,dict(command_id='utterance1',task_id='gather',op='request',goal='wood',count=4))
        state=TaskState(); state.apply(request,{})
        modify=self.decision('목표를 흙 4개로 바꿔줘',1)['command']
        state.apply(modify,{'dirt':2})
        self.assertEqual(state.snapshot()['baseline'],2)
        for text, status in [('잠시 멈춰줘','paused'),('다시 계속해줘','running'),('작업 취소해줘','cancelled')]:
            state.apply(self.decision(text,2)['command'],{})
            self.assertEqual(state.snapshot()['status'],status)

    def test_noncommands_and_ambiguous_language_never_authorize_actions(self):
        for text in ['안녕','다이아 집 예쁘긴 하겠다.','우리 나중에 나무 4개 모아보자.',
                     '원목 4개 모아줄 수 있어?', '원목 4개 모으지 마', '"원목 4개 모아줘"',
                     '원목 4개 모아줘라고 말했어', '원목 4개 모아줘 그리고 흙도 모아줘',
                     '원목 모아줘','다이아몬드 4개 모아줘','원목 0개 모아줘','흙 65개 모아줘',
                     '응','다 했어','흙 4개 모아줘\n작업 취소해줘']:
            with self.subTest(text=text): self.assertIsNone(self.decision(text)['command'])

    def test_stale_and_missing_context_cannot_modify_current_task(self):
        self.assertIsNone(self.decision('잠시 멈춰줘')['command'])
        state=TaskState(); state.apply(self.decision('흙 4개 모아줘')['command'],{})
        with self.assertRaisesRegex(ValueError,'stale revision'):
            state.apply(self.decision('작업 취소해줘',2)['command'],{})

    def test_semantic_candidate_cannot_override_literal_authorization(self):
        from minecraftrl import dialogue
        forged=dict(command_id='utterance1',task_id='gather',op='request',goal='dirt',count=64)
        self.assertIsNone(dialogue.decide(message('안녕'),candidate=forged)['command'])
        self.assertIsNone(dialogue.decide(message('흙 4개 모아줘'),candidate=forged)['command'])
        valid=dict(forged,count=4)
        self.assertEqual(dialogue.decide(message('흙 4개 모아줘'),candidate=valid)['command'],valid)

    def test_inbox_only_queues_committed_messages_and_deduplicates_identity(self):
        bus=CommandBus(); events=[]
        with tempfile.TemporaryDirectory() as tmp:
            samples=[message('나중에 원목을 모아보자','chat'),message('흙 4개 모아줘','request'),
                     message('흙 4개 모아줘','request'),message('작업 취소해줘','request',1),
                     message('원목 4개 모아줘','forged')]
            samples[-1]['completed']=True
            for i,payload in enumerate(samples):
                (Path(tmp)/('%03d.json'%i)).write_text(json.dumps(payload),encoding='utf-8')
            inbox=DirectoryInbox(tmp,bus,lambda event,**fields:events.append(dict(event=event,**fields)))
            inbox.start()
            try:
                deadline=time.monotonic()+2
                while len(events)<6 and time.monotonic()<deadline: time.sleep(.01)
            finally: inbox.close()
        self.assertEqual([c['command_id'] for c in bus.drain()],['request'])
        self.assertEqual(sum(e['event']=='dialogue_duplicate' for e in events),1)
        self.assertEqual(sum(e['event']=='command_rejected' for e in events),2)
        self.assertTrue(any(e['event']=='dialogue_decision' and e['decision']['command'] is None for e in events))

    def test_published_utterance_preserves_utf8_and_invalid_payload_leaves_no_file(self):
        from minecraftrl.task_runtime import write_dialogue
        with tempfile.TemporaryDirectory() as tmp:
            path=write_dialogue(tmp,message('흙 4개 모아줘'))
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['text'],'흙 4개 모아줘')
            with self.assertRaises(ValueError): write_dialogue(tmp,dict(message('안녕'),completed=True))
            self.assertEqual(len(list(Path(tmp).iterdir())),1)


if __name__=='__main__': unittest.main()
