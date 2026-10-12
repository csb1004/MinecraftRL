"""No-API real-policy probe: induced stall/save, restore/replan/resume."""
import datetime
import json
from pathlib import Path
import subprocess
import threading
import time

from minecraftrl.dialogue_console import SessionDialogue
from minecraftrl.memory import atomic_json
from minecraftrl.task_runtime import write_command


def main():
    root=Path(__file__).resolve().parents[1]
    stop=root/'results/STOP'
    if not stop.exists(): raise SystemExit('Latch STOP and inspect previous shutdown first')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base=root/'results'/('failure-recovery-'+stamp); base.mkdir()
    result=dict(passed=False,api_calls=0,seed=130363,induced_stall_threshold_s=5,
                slot='failure-'+stamp,reports=[],errors=[])
    linuxroot='/mnt/d/DATA/Desktop/MinecraftRL'
    def linux(path): return linuxroot+'/'+path.relative_to(root).as_posix()
    def require(value,message):
        if not value: raise AssertionError(message)
    def rows(folder):
        path=folder/'events.jsonl'
        if not path.exists(): return []
        return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n')[:-1]]
    def until(predicate,proc,timeout=180):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if stop.exists(): raise RuntimeError('global STOP requested')
            if predicate(): return
            if proc.poll() is not None: raise RuntimeError('session ended before condition')
            time.sleep(.03)
        raise RuntimeError('condition timed out; no automatic retry')
    print(str(base),flush=True)
    stop.rename(base/'STOP-before.txt')
    try:
        for phase in range(2):
            folder=base/str(phase); local_stop=base/('STOP-'+str(phase))
            argv=['wsl.exe','-d','Ubuntu-24.04','--cd',linuxroot,'--','env','GALLIUM_DRIVER=d3d12',
                  'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+linuxroot+'/.runtime/mamba',
                  'XDG_CACHE_HOME='+linuxroot+'/.cache','HF_HOME='+linuxroot+'/.cache/huggingface',
                  linuxroot+'/.runtime/bin/micromamba','run','-p',linuxroot+'/.runtime/steve1',
                  'python','-m','minecraftrl.cli','session','--root',linuxroot,'--policy','steve1',
                  '--seed',str(result['seed']),'--limit-s','180','--no-progress-s','5' if phase==0 else '60',
                  '--world-slot',result['slot'],'--output',linux(folder),'--stop-file',linux(local_stop)]
            finished=threading.Event()
            def forward_stop():
                while not finished.wait(.02):
                    if stop.exists(): local_stop.touch(); return
            with (base/('supervisor-'+str(phase)+'.log')).open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                bridge=threading.Thread(target=forward_stop); bridge.start()
                try:
                    console=SessionDialogue(folder,stop_file=stop)
                    def ready():
                        try: console.sync(); return True
                        except ValueError as exc:
                            if str(exc)=='session_not_ready': return False
                            raise
                    until(ready,proc,600)
                    if phase==0:
                        write_command(folder/'commands',dict(command_id='initial',task_id='gather',op='request_plan',
                            steps=[{'goal':'wood','count':64},{'goal':'dirt','count':1}]))
                        until(lambda: (console.sync() or {}).get('failure',{}).get('reason')=='no_progress',proc)
                        blocked=console.sync(); require(blocked['status']=='paused','stall did not pause')
                        require(blocked['stage_index']==0,'unexpected stage completion')
                        result['blocked_task']=blocked
                        time.sleep(1)
                        local_stop.touch(); proc.wait(timeout=90)
                    else:
                        restored=console.sync(); result['restored_task']=restored
                        require(restored['status']=='paused' and restored['failure']==result['blocked_task']['failure'],
                                'failure evidence lost on restore')
                        require(restored['baseline']==result['blocked_task']['baseline'],'baseline changed on restore')
                        time.sleep(1)
                        write_command(folder/'commands',dict(command_id='replan',task_id='gather',op='modify_plan',
                            expected_revision=restored['revision'],steps=[{'goal':'dirt','count':1}]))
                        until(lambda: console.sync()['revision']==restored['revision']+1,proc)
                        revised=console.sync(); require(revised['status']=='paused','replanning resumed without request')
                        write_command(folder/'commands',dict(command_id='resume',task_id='gather',op='resume',
                            expected_revision=revised['revision']))
                        proc.wait(timeout=210)
                finally:
                    local_stop.touch()
                    try: proc.wait(timeout=90)
                    finally: finished.set(); bridge.join(timeout=2)
            report=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            result['reports'].append(report); atomic_json(base/'acceptance.json',result)
            events=rows(folder)
            require(report.get('release')=='neutral_command_sent','input release missing')
            require(report.get('process_cleanup',{}).get('remaining_live_pids')==[],'owned processes survived')
            require(report.get('world_save',{}).get('status')=='saved','world save failed')
            stopped=next(e['monotonic_s'] for e in events if e['event']=='stop_acknowledged')
            require(not any(e['event']=='policy_dispatch_started' and e['monotonic_s']>stopped for e in events),'dispatch after STOP')
            if phase==0:
                blocked_event=next(e for e in events if e['event']=='task_no_progress')
                require(not any(e['event']=='policy_dispatch_started' and e['monotonic_s']>blocked_event['monotonic_s'] for e in events),'dispatch after stall')
                require(report['steps']>0,'no real policy action before induced stall')
                require(report['task']['status']=='paused','wrong saved status')
                result['after_stall_dispatches']=0
            else:
                resumed=next(e['monotonic_s'] for e in events if e['event']=='command_acknowledged' and e['op']=='resume')
                require(not any(e['event']=='policy_dispatch_started' and e['monotonic_s']<resumed for e in events),'dispatch before resume')
                require(report.get('world_restore',{}).get('inventory_match') is True,'restore inventory mismatch')
                require(report['task']['status']=='completed' and report['task']['progress']>=1,'replanned dirt task did not complete')
                result['before_resume_dispatches']=0
            print(json.dumps(dict(phase=phase,status=report['task']['status'],steps=report['steps'],no_progress_events=report['no_progress_events'])),flush=True)
        result['passed']=True
    except Exception as exc:
        result['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        stop.write_text('Failure recovery acceptance ended. Inspect evidence before re-arming.',encoding='utf-8')
        atomic_json(base/'acceptance.json',result)
    print(json.dumps(dict(passed=result['passed'],errors=result['errors'],api_calls=0)),flush=True)
    return 0 if result['passed'] else 2


if __name__=='__main__': raise SystemExit(main())
