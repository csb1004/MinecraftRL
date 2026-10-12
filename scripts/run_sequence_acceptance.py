"""One integrated Windows/WSL acceptance run; explicit paid opt-in and durable cap."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from minecraftrl.semantic import OpenAISemantic, PLANNER, _transport
from minecraftrl.dialogue_console import SessionDialogue
from minecraftrl.memory import atomic_json
from minecraftrl.task_runtime import write_command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-paid-api',action='store_true')
    parser.add_argument('--budget-file',type=Path,required=True)
    parser.add_argument('--max-api-calls',type=int,default=30)
    args=parser.parse_args()
    if not args.allow_paid_api: parser.error('explicit paid API authorization required')
    if not 1<=args.max_api_calls<=30: parser.error('budget must be 1 to 30')
    root=Path(__file__).resolve().parents[1]; global_stop=root/'results/STOP'
    if not global_stop.exists(): raise SystemExit('Inspect prior shutdown and latch STOP before starting this bundle')
    if args.budget_file.resolve().parent!=root/'results': raise SystemExit('Budget must be inside project results')
    key=os.environ.get('OPENAI_API_KEY'); model=os.environ.get('OPENAI_MODEL')
    if not key or not model: raise SystemExit('Process-local OPENAI_API_KEY and OPENAI_MODEL required')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base=root/'results'/('sequence-acceptance-'+stamp); base.mkdir()
    state={'authorized_max_attempts':args.max_api_calls,'attempts':[],'reports':[],'errors':[],
           'scenario':base.name,'slot':'sequence-'+stamp,'passed':False,
           'model':model,'prompt_sha256':hashlib.sha256(PLANNER.encode('utf-8')).hexdigest()}
    with args.budget_file.open('x',encoding='utf-8') as stream: json.dump(state,stream)
    def save(): atomic_json(args.budget_file,state)
    def transport(request,timeout):
        if len(state['attempts'])>=args.max_api_calls: raise RuntimeError('persistent_bundle_budget_exhausted')
        row={'number':len(state['attempts'])+1,'request':json.loads(request.data),'status':'started'}
        state['attempts'].append(row); save(); started=time.monotonic()
        try:
            response=_transport(request,timeout); row.update(status='received',response=response); return response
        except Exception as exc:
            row.update(status='error',error_type=type(exc).__name__); raise RuntimeError('transport_failed') from None
        finally: row['elapsed_s']=time.monotonic()-started; save()
    provider=OpenAISemantic(model,key,allow_paid=True,transport=transport,max_calls=min(20,args.max_api_calls))
    def require(value,message):
        if not value: raise AssertionError(message)
    def until(predicate,proc,timeout=180):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if global_stop.exists(): raise RuntimeError('global_stop')
            if predicate(): return
            if proc.poll() is not None: raise RuntimeError('game_ended_before_condition')
            time.sleep(.03)
        raise RuntimeError('acceptance_wait_timeout')
    def checked_confirmation(console,text,op,steps=None,goal=None,count=None):
        proposal=console.turn(text)
        require(proposal['status']=='needs_confirmation','semantic proposal missing for '+op)
        command=proposal['proposal']; require(command['op']==op,'wrong semantic operation')
        if steps is not None: require(command.get('steps')==steps,'wrong stage order or count')
        if goal is not None: require(command.get('goal')==goal and command.get('count')==count,'wrong current-stage target')
        result=console.turn('확인 '+proposal['confirmation_id']); require(result['status']=='submitted','confirmation failed')
    print('Sequence acceptance: '+str(base),flush=True)
    try:
        # Fixed pre-game contract cases, no persona calls or game dispatch.
        cases=[
            ('원목 두 개를 모은 뒤 흙 세 개를 모아줘.','TASK_PLAN_REQUEST',[{'goal':'wood','count':2},{'goal':'dirt','count':3}]),
            ('흙 한 개를 모으고 그 다음 원목 세 개를 모아줘.','TASK_PLAN_REQUEST',[{'goal':'dirt','count':1},{'goal':'wood','count':3}]),
            ('나중에 원목을 모으고 흙도 모아보면 좋겠다.','TASK_PROPOSAL',None),
            ('원목을 모은 다음 흙도 모아줘. 수량은 아직 안 정했어.','UNRESOLVED',None),
            ('원목 두 개를 모은 다음 다이아몬드 검을 제작해줘.','UNRESOLVED',None),
        ]
        state['semantic_cases']=[]
        for text,intent,steps in cases:
            candidate=provider.parse(text,[],None)
            passed=candidate['intent']==intent and candidate.get('steps')==steps
            state['semantic_cases'].append(dict(text=text,expected_intent=intent,expected_steps=steps,actual=candidate,passed=passed)); save()
            require(passed,'pre-game semantic case failed')
        global_stop.rename(base/'STOP-before.txt')
        linuxroot='/mnt/d/DATA/Desktop/MinecraftRL'
        def linux(path): return linuxroot+'/'+path.relative_to(root).as_posix()
        for phase in range(3):
            folder=base/str(phase); stop=base/('STOP-phase-'+str(phase))
            slot=state['slot'] if phase<2 else state['slot']+'-cancel'
            argv=['wsl.exe','-d','Ubuntu-24.04','--cd',linuxroot,'--','env','GALLIUM_DRIVER=d3d12',
                  'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+linuxroot+'/.runtime/mamba',
                  'XDG_CACHE_HOME='+linuxroot+'/.cache','HF_HOME='+linuxroot+'/.cache/huggingface',
                  linuxroot+'/.runtime/bin/micromamba','run','-p',linuxroot+'/.runtime/steve1',
                  'python','-m','minecraftrl.cli','session','--root',linuxroot,'--policy','steve1','--seed','130363',
                  '--limit-s','180','--world-slot',slot,'--output',linux(folder),'--stop-file',linux(stop)]
            done=threading.Event()
            def forward_stop():
                while not done.wait(.02):
                    if global_stop.exists(): stop.touch(); return
            with (base/('supervisor-'+str(phase)+'.log')).open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                bridge=threading.Thread(target=forward_stop); bridge.start()
                try:
                    console=SessionDialogue(folder,provider,memory_path=base/'dialogue-memory.json',stop_file=global_stop)
                    def ready():
                        try: console.sync(); return True
                        except ValueError as exc:
                            if str(exc)=='session_not_ready': return False
                            raise
                    until(ready,proc,600)
                    if phase==0:
                        checked_confirmation(console,'원목 한 개를 모은 뒤 흙 예순네 개를 모아줘.','request_plan',
                                             steps=[{'goal':'wood','count':1},{'goal':'dirt','count':64}])
                        until(lambda: (console.sync() or {}).get('stage_index')==1 and console.task['progress']>=1,proc)
                        require(console.turn('잠시 멈춰줘')['status']=='submitted','pause not submitted')
                        until(lambda: console.sync()['status']=='paused',proc)
                        checked_confirmation(console,'현재 흙 수집 단계의 목표 수량만 두 개로 바꿔줘.','modify',goal='dirt',count=2)
                        until(lambda: console.sync()['revision']==3 and not console.waiting,proc)
                        require(len(console.task['completed_stages'])==1,'wood completion lost on modify')
                        stop.touch(); proc.wait(timeout=90)
                    elif phase==1:
                        restored=console.sync(); state['restored_task']=restored; save()
                        require(restored['status']=='paused' and restored['stage_index']==1 and len(restored['completed_stages'])==1,'wrong restored stage')
                        time.sleep(1)
                        checked_confirmation(console,'중단했던 순서 작업을 남은 흙 단계부터 이어서 해줘.','resume')
                        proc.wait(timeout=210)
                    else:
                        write_command(folder/'commands',dict(command_id='cancel-test-start',task_id='gather',op='request_plan',
                                      steps=[{'goal':'wood','count':64},{'goal':'dirt','count':1}]))
                        def first_inference():
                            return any(json.loads(line)['event']=='inference_started' for line in
                                       (folder/'events.jsonl').read_text(encoding='utf-8').split('\n')[:-1])
                        until(first_inference,proc)
                        write_command(folder/'commands',dict(command_id='cancel-test-stop',task_id='gather',op='cancel',expected_revision=1))
                        proc.wait(timeout=90)
                finally:
                    stop.touch()
                    try: proc.wait(timeout=90)
                    finally: done.set(); bridge.join(timeout=2)
            report=json.loads((folder/'summary.json').read_text(encoding='utf-8')); state['reports'].append(report); save()
            require(report.get('release')=='neutral_command_sent','neutral release missing')
            require(report.get('process_cleanup',{}).get('remaining_live_pids')==[],'owned processes survived')
            require(report.get('world_save',{}).get('status')=='saved','world save failed')
            require(report['task']['status']==['paused','completed','cancelled'][phase],'unexpected final task state')
            if phase==1:
                require(report.get('world_restore',{}).get('inventory_match') is True,'restore inventory mismatch')
                require(len(report['task']['completed_stages'])==2,'plan not fully completed')
            rows=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            stopped=next(r['monotonic_s'] for r in rows if r['event']=='stop_acknowledged')
            require(not any(r['event']=='policy_dispatch_started' and r['monotonic_s']>stopped for r in rows),'dispatch after STOP')
            if phase==2:
                cancelled=next(r['monotonic_s'] for r in rows if r['event']=='command_acknowledged' and r['op']=='cancel')
                require(not any(r['event']=='policy_dispatch_started' and r['monotonic_s']>cancelled for r in rows),'dispatch after cancel')
            print(json.dumps({'phase':phase,'status':report['task']['status'],'steps':report['steps'],'api_attempts':len(state['attempts'])}),flush=True)
        state['passed']=True
    except Exception as exc:
        state['errors'].append({'type':type(exc).__name__,'reason':str(exc) if isinstance(exc,AssertionError) else 'Inspect local phase evidence; no automatic retry'})
    finally:
        global_stop.write_text('Sequence acceptance ended. Inspect receipt before re-arming.',encoding='utf-8')
        save(); atomic_json(base/'acceptance.json',state)
    print(json.dumps({'passed':state['passed'],'api_attempts':len(state['attempts']),'errors':state['errors']}),flush=True)
    return 0 if state['passed'] else 2


if __name__=='__main__': raise SystemExit(main())
