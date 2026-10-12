"""Separate dialogue process: network waits never block the policy command inbox."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid
from .dialogue import decide
from .semantic import Conversation, OpenAISemantic
from .dialogue_context import load_user_context
from .task_runtime import write_command
from .memory import DialogueMemory


class SessionDialogue:
    def __init__(self,folder,provider=None,*,persona='',context=None,memory_path=None,stop_file=None):
        self.folder=Path(folder); self.provider=provider; self.persona=persona
        self.stop_files=[self.folder/'STOP',self.folder.parent/'STOP']
        if stop_file is not None: self.stop_files.append(Path(stop_file))
        memory=DialogueMemory(memory_path or self.folder/'conversation-state.json')
        self.conversation=Conversation(memory=memory); self.profile=(context or {}).get('profile',{})
        if memory.data['revision']==0:
            for row in (context or {}).get('history',[])[-12:]: self.conversation.remember(row['role'],row['content'])
        self.position=0; self.partial=b''; self.task=None; self.ready=False; self.closed=False
        self.epoch=0; self.waiting=None; self.queued=set()
        self.finished_commands=set(); self.rejected_files=set(); self.previous_goals=[]
        self.log=self.folder/('dialogue-console-'+uuid.uuid4().hex+'.jsonl')

    def sync(self):
        if any(path.exists() for path in self.stop_files) or (self.folder/'summary.json').exists():
            raise ValueError('session_stopped_or_ended')
        path=self.folder/'events.jsonl'
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(self.position); raw=stream.read(); self.position=stream.tell()
            lines=(self.partial+raw).split(b'\n'); self.partial=lines.pop()
            for line in lines:
                event=json.loads(line); name=event['event']; cid=event.get('command_id')
                if name=='session_ready': self.ready=True
                if name in ('stop_acknowledged','episode_finished'): self.closed=True
                if name=='command_queued': self.queued.add(cid); self.epoch+=1
                if name in ('command_acknowledged','command_rejected') and cid:
                    self.queued.discard(cid); self.epoch+=1
                    self.finished_commands.add(cid)
                    if cid==self.waiting: self.waiting=None
                if name=='command_rejected' and event.get('file'): self.rejected_files.add(event['file'])
                if name in ('plan_stage_completed','task_no_progress'): self.epoch+=1
                if name in ('command_acknowledged','inventory_observed','world_restored','plan_stage_completed','inventory_target_reached','task_no_progress') and event.get('task'):
                    incoming=event['task']
                    if self.task and incoming.get('revision')!=self.task.get('revision'):
                        self.previous_goals.append({k:self.task.get(k) for k in ('task_id','revision','goal','target_count')})
                        self.previous_goals[-1]['status']=('completed_stage' if name=='plan_stage_completed' else
                                                          'interrupted' if name=='task_no_progress' else 'superseded')
                        self.previous_goals=self.previous_goals[-16:]
                    self.task=copy.deepcopy(event['task'])
                    if name=='inventory_observed':
                        self.task['observed_inventory']=event.get('inventory')
                        self.task['observation_sequence']=event.get('observation_sequence')
                        if 'target_items' in self.task and isinstance(event.get('inventory'),dict):
                            self.task['progress']=max(0,sum(event['inventory'].get(k,0) for k in self.task['target_items'])-self.task['baseline'])
                if name=='inventory_target_reached' and self.task:
                    self.task['status']='completed'; self.task['progress']=event['progress']
        # A restart can occur after file publication but before command_queued.
        # Reconcile that gap from the existing outbox, never resend it.
        from .tasks import validate_command
        for path in (self.folder/'commands').glob('*.json'):
            if path.name in self.rejected_files or path.is_symlink(): continue
            try:
                if path.stat().st_size>65536: continue
                command=json.loads(path.read_text(encoding='utf-8'))
                if isinstance(command,dict) and command.get('kind')=='dialogue': command=decide(command)['command']
                if command is None: continue
                validate_command(command)
                if command['command_id'] not in self.finished_commands: self.queued.add(command['command_id'])
            except (ValueError,OSError): continue
        if self.closed: raise ValueError('session_stopped_or_ended')
        if not self.ready: raise ValueError('session_not_ready')
        if self.task is not None:
            self.task['_epoch']=self.epoch
            self.task['previous_goals']=copy.deepcopy(self.previous_goals)
        return copy.deepcopy(self.task)

    def turn(self,text):
        if not isinstance(text,str) or not 0<len(text)<=2048: raise ValueError('invalid_utterance')
        task=self.sync()
        if self.waiting or self.queued: return {'status':'awaiting_worker_ack','command':None}
        if text.strip().startswith('확인 '):
            result=self.conversation.confirm(text,task)
        else:
            self.conversation.pending=None
            message={'kind':'dialogue','message_id':uuid.uuid4().hex,'task_id':task['task_id'] if task else 'gather','text':text}
            if task and task['status'] not in ('completed','cancelled'): message['expected_revision']=task['revision']
            direct=decide(message)
            if direct['command'] is not None:
                result=direct; self.conversation.remember('user',text,'task_context')
            elif self.provider is None:
                self.conversation.remember('user',text,{'PREFERENCE':'preference','TASK_PROPOSAL':'unconfirmed_proposal','QUESTION':'question'}.get(direct['utterance_type'],'context'))
                result=dict(direct,reply='현재는 제한된 한국어 명령만 사용할 수 있습니다.')
            else:
                extra={'summary':self.conversation.memory.summary} if self.conversation.memory.summary else {}
                candidate=self.provider.parse(text,self.conversation.history,task,**extra)
                # Bind to the pre-call context. A slow response cannot silently
                # target a newer task/revision; sync again before confirmation.
                result=self.conversation.propose(text,candidate,task)
                if result['status']=='no_action':
                    extra={'summary':self.conversation.memory.summary} if self.conversation.memory.summary else {}
                    result['reply']=self.provider.reply(text,self.conversation.history,self.sync(),self.persona,self.profile,**extra)
                    self.conversation.remember('assistant',result['reply'])
        if result.get('command') is not None:
            current=self.sync()
            from .tasks import TaskState
            state=TaskState(); state.task=current; state.check(result['command'])
            if self.queued: return {'status':'awaiting_worker_ack','command':None}
            path=write_command(self.folder/'commands',result['command'])
            self.waiting=result['command']['command_id']
            result=dict(result,status='submitted',file=path.name)
        with self.log.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'utterance':text,'result':result},ensure_ascii=False)+'\n')
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--allow-paid-api',action='store_true')
    parser.add_argument('--max-api-calls',type=int,default=5)
    parser.add_argument('--memory-file',type=Path)
    parser.add_argument('--user-id')
    parser.add_argument('--persona-file',type=Path)
    parser.add_argument('--inspect-memory',action='store_true')
    args=parser.parse_args(); root=Path(__file__).resolve().parents[2]
    folder=args.session.resolve()
    if root/'results' not in folder.parents or not folder.is_dir():
        parser.error('session must exist inside project results')
    identity=hashlib.sha256((args.user_id or 'local-operator').encode('utf-8')).hexdigest()[:32]
    memory_path=root/'data/dialogue'/(identity+'.json')
    if args.inspect_memory:
        from .tasks import TaskState
        checkpoint_path=folder/'task-checkpoint.json'; restored=None
        if checkpoint_path.exists():
            value=json.loads(checkpoint_path.read_text(encoding='utf-8'))
            restored=TaskState.restore(value,value['world_id']).snapshot()
        print(json.dumps({'mode':'inspection_only_no_execution','dialogue':DialogueMemory(memory_path).export(),
                          'recovered_task':restored,'world_resume_supported':True,
                          'world_resume_requires_saved_slot':True,'inspection_does_not_load_world':True},ensure_ascii=False,indent=2))
        return 0
    memory_path.parent.mkdir(parents=True,exist_ok=True)
    context=None
    if args.memory_file:
        if not args.user_id: parser.error('--memory-file requires --user-id')
        context=load_user_context(args.memory_file,args.user_id)
    persona_path=args.persona_file or root/'configs/dialogue-persona.txt'
    if persona_path.stat().st_size>32768: parser.error('persona file too large')
    persona=persona_path.read_text(encoding='utf-8-sig')
    provider=OpenAISemantic(os.getenv('OPENAI_MODEL'),os.getenv('OPENAI_API_KEY'),
        allow_paid=True,max_calls=args.max_api_calls) if args.allow_paid_api else None
    session=SessionDialogue(folder,provider,persona=persona,context=context,memory_path=memory_path,
                            stop_file=root/'results/STOP')
    print('대화 콘솔: /exit 종료. 모델 제안은 표시된 확인 문장으로 확정합니다.',flush=True)
    while True:
        try:
            text=input('> ')
            if text.strip()=='/exit': break
            print(json.dumps(session.turn(text),ensure_ascii=False),flush=True)
        except (EOFError,KeyboardInterrupt): break
        except (ValueError,RuntimeError) as exc: print(str(exc),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
