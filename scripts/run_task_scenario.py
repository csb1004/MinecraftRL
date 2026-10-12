"""Real same-world acceptance scenario. Never fabricates inventory or actions."""
import argparse
import datetime
import json
from pathlib import Path
import threading
import time
import traceback
from minecraftrl.cli import episode, condition_for
from minecraftrl.runtime import write_json
from minecraftrl.task_runtime import write_command, write_dialogue


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=130363)
    parser.add_argument('--limit-s',type=float,default=180.)
    parser.add_argument('--dialogue',action='store_true')
    args=parser.parse_args(); root=args.root.resolve()
    stop=root/'results/STOP'
    if stop.exists(): raise SystemExit('Global STOP is latched')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=root/'results'/(('dialogue-scenario-' if args.dialogue else 'task-scenario-')+str(args.seed)+'-'+stamp)
    args.root=root; args.policy='steve1'; args.task_session=True; args.output=output; args.stop_file=stop
    finished=threading.Event(); sent=[]; errors=[]
    print('Scenario output: '+str(output),flush=True)
    def send(op,cid,**fields):
        command=dict(command_id=cid,task_id='gather',op=op,**fields)
        before=time.monotonic()
        if args.dialogue:
            utterances={'request':'지금 원목 4개 모아줘','modify':'목표를 흙 4개로 바꿔줘',
                        'pause':'잠시 멈춰줘','resume':'다시 계속해줘'}
            message=dict(kind='dialogue',message_id=cid,task_id='gather',text=utterances[op])
            if 'expected_revision' in fields: message['expected_revision']=fields['expected_revision']
            file=write_dialogue(output/'commands',message)
        else: file=write_command(output/'commands',command)
        after=time.monotonic()
        sent.append(dict(command=command,file=file.name,before_s=before,after_s=after))
    def watch():
        position=0; pending=b''; stage='ready'; resume_at=None; request_at=None; dirt_steps=0
        try:
            while not finished.wait(.005):
                if stop.exists(): return
                if request_at is not None and time.monotonic()>=request_at:
                    send('request','scenario-request',goal='wood',count=4); request_at=None; stage='wood'
                if resume_at is not None and time.monotonic()>=resume_at:
                    send('resume','scenario-resume',expected_revision=2); resume_at=None; stage='finishing'
                path=output/'events.jsonl'
                if not path.exists(): continue
                with path.open('rb') as stream:
                    stream.seek(position); block=stream.read(); position=stream.tell()
                pieces=(pending+block).split(b'\n'); pending=pieces.pop()
                for piece in pieces:
                    event=json.loads(piece)
                    name=event['event']; task=event.get('task') or {}
                    if name=='session_ready' and stage=='ready':
                        if args.dialogue:
                            for index,text in enumerate(['안녕','나중에 원목 4개 모아보자.','흙 4개 모아줄 수 있어?',
                                                        '"원목 4개 모아줘"','원목 4개 모으지 마','응']):
                                write_dialogue(output/'commands',dict(kind='dialogue',message_id='nonaction-'+str(index),
                                                                      task_id='gather',text=text))
                            request_at=time.monotonic()+1.; stage='nonactions'
                        else:
                            send('request','scenario-request',goal='wood',count=4); stage='wood'
                    elif name=='inference_started' and stage=='wood' and task.get('revision')==1 and event['attempt']>=4:
                        send('modify','scenario-modify',expected_revision=1,goal='dirt',count=4); stage='modifying'
                    elif name=='command_acknowledged' and event['command_id']=='scenario-modify': stage='dirt'
                    elif name=='policy_dispatch_finished' and task.get('revision')==2: dirt_steps+=1
                    elif name=='inference_started' and stage=='dirt' and dirt_steps>=1:
                        send('pause','scenario-pause',expected_revision=2); stage='pausing'
                    elif name=='command_acknowledged' and event['command_id']=='scenario-pause':
                        resume_at=time.monotonic()+2.; stage='paused'
        except BaseException:
            errors.append(traceback.format_exc()); stop.touch(exist_ok=True)
    thread=threading.Thread(target=watch,daemon=True); thread.start()
    try: report=episode(args,condition_for(root,'steve1','wood'))
    finally: finished.set(); thread.join(timeout=2)
    rows=[]
    if (output/'events.jsonl').exists():
        for line in (output/'events.jsonl').read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    acks={r['command_id']:r for r in rows if r['event']=='command_acknowledged'}
    expected=['scenario-request','scenario-modify','scenario-pause','scenario-resume']
    controls_present=all(x in acks for x in expected)
    dispatch=[r for r in rows if r['event']=='policy_dispatch_started']
    old_after_modify=paused_actions=None
    if controls_present:
        old_after_modify=sum(r['monotonic_s']>acks[expected[1]]['monotonic_s'] and r['task']['revision']<2 for r in dispatch)
        paused_actions=sum(acks[expected[2]]['monotonic_s']<=r['monotonic_s']<acks[expected[3]]['monotonic_s'] for r in dispatch)
    release=any(r['event']=='input_release' and r.get('reason')=='pause' and r.get('outcome')=='neutral_command_sent' for r in rows)
    discarded=any(r['event']=='action_discarded' and r['task']['revision']==1 for r in rows)
    completed=[r for r in rows if r['event']=='inventory_target_reached' and r['revision']==2]
    task=report.get('task') or {}
    valid_inventory=bool(completed and completed[-1]['inventory'].get('dirt',0)-task['baseline']>=4)
    checks={'controls_acknowledged':controls_present,'old_revision_dispatches_after_modify':old_after_modify,
            'policy_dispatches_while_paused':paused_actions,'pause_neutral_release':release,
            'old_inference_discarded':discarded,'environment_resets':report.get('environment_resets'),
            'inventory_confirmed_completion':valid_inventory,'task_completed':report.get('task_completed',False),
            'worker_exit_code':report.get('worker_exit_code'),
            'remaining_live_pids':report.get('process_cleanup',{}).get('remaining_live_pids'),
            'watcher_errors':errors,'seed':args.seed,'output':str(output),'benchmark_sample':False}
    checks['scenario_passed']=bool(controls_present and old_after_modify==0 and paused_actions==0 and release and discarded
                                  and report.get('environment_resets')==1 and valid_inventory and report.get('execution_valid')
                                  and report.get('worker_exit_code')==0 and checks['remaining_live_pids']==[] and not errors)
    if args.dialogue:
        decisions=[r for r in rows if r['event']=='dialogue_decision']
        noactions=[r for r in decisions if r['message_id'].startswith('nonaction-')]
        authorized=[r for r in decisions if r['message_id'] in expected and r['decision']['command'] is not None]
        first_ack=acks.get('scenario-request',{}).get('monotonic_s',float('inf'))
        before_request=sum(r['monotonic_s']<first_ack for r in dispatch)
        checks.update(dialogue_nonaction_decisions=len(noactions),dialogue_authorized_decisions=len(authorized),
                      dispatches_before_request=before_request)
        checks['scenario_passed']=bool(checks['scenario_passed'] and len(noactions)==6 and len(authorized)==4
                                      and all(r['decision']['command'] is None for r in noactions) and before_request==0)
    write_json(output/'scenario-commands.json',sent); write_json(output/'scenario-evaluation.json',checks)
    print(json.dumps(checks,indent=2)); return 0 if checks['scenario_passed'] else 2


if __name__=='__main__': raise SystemExit(main())
