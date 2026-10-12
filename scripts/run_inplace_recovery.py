"""Opt-in experiment for at most one same-process retry after natural stagnation."""
import copy
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

from minecraftrl.dialogue_console import SessionDialogue
from minecraftrl.memory import atomic_json
from minecraftrl.task_runtime import write_command
from minecraftrl.tasks import validate_command
from minecraftrl.world_store import WorldStore, digest


class OneRetry:
    """Experiment-only permission for one retry, never a normal-session default."""
    def __init__(self): self.before=None

    def on_stall(self,task):
        if self.before is not None or task.get('status')!='paused' or task.get('failure',{}).get('reason')!='no_progress':
            return None
        command=dict(command_id='retry-once',task_id=task['task_id'],op='resume',expected_revision=task['revision'])
        validate_command(command)
        self.before=copy.deepcopy(task)
        return command


def assert_preserved(before,after):
    keys=('task_id','revision','goal','policy_goal','target_items','target_count','baseline','progress',
          'plan','stage_index','completed_stages')
    if after['status']!='running' or 'failure' in after or any(before.get(k)!=after.get(k) for k in keys):
        raise ValueError('retry changed the committed task or progress')


def main():
    root=Path(__file__).resolve().parents[1]; global_stop=root/'results/STOP'
    if not global_stop.exists(): raise SystemExit('STOP must be latched before starting')
    slot='failure-20261004T032512557525Z'; generation_name='1c1415241ad44a81b6e89a1943472bf0'
    source=root/'data/worlds'/slot; generation=source/generation_name
    original_pointer=(source/'current.json').read_bytes()
    manifest_hash=digest(generation/'manifest.json'); archive_hash=digest(generation/'world.zip')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base=root/'results'/('inplace-recovery-'+stamp); base.mkdir()
    configured=json.loads((root/'configs/experiment.json').read_text())['seeds']
    seeds=[130363]+[seed for seed in configured if seed!=130363]
    state=dict(completed=False,api_calls=0,source_slot=slot,source_generation=generation_name,seeds=seeds,
               archive_sha256=archive_hash,manifest_sha256=manifest_hash,planned_runs=len(seeds),
               max_retries_per_run=1,limit_s=180,no_progress_s=60,reports=[],errors=[])
    atomic_json(base/'experiment.json',state)
    def save(): atomic_json(base/'experiment.json',state)
    def require(condition,message):
        if not condition: raise AssertionError(message)
    linuxroot='/mnt/d/DATA/Desktop/MinecraftRL'
    def linux(path): return linuxroot+'/'+path.relative_to(root).as_posix()
    def wait_for(predicate,proc,timeout=600):
        deadline=time.monotonic()+timeout
        while True:
            if global_stop.exists(): raise RuntimeError('global STOP requested')
            if predicate(): return
            if proc.poll() is not None or time.monotonic()>deadline: raise RuntimeError('session ended or timed out before condition')
            time.sleep(.03)
    print(str(base),flush=True)
    global_stop.rename(base/'STOP-before.txt')
    try:
        for seed in seeds:
            if global_stop.exists(): raise RuntimeError('global STOP requested')
            trial=base/str(seed); clone=trial/'worlds'/slot; clone.mkdir(parents=True)
            shutil.copytree(generation,clone/generation_name)
            atomic_json(clone/'current.json',dict(generation=generation_name,manifest_sha256=manifest_hash))
            with WorldStore(trial/'worlds',slot) as store:
                bundle=store.load(); require(bundle['archive_sha256']==archive_hash,'copy hash mismatch')
            folder=trial/'run'; stop=trial/'STOP'; budget=OneRetry(); finished=threading.Event()
            argv=['wsl.exe','-d','Ubuntu-24.04','--cd',linuxroot,'--','env','GALLIUM_DRIVER=d3d12',
                  'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+linuxroot+'/.runtime/mamba',
                  'XDG_CACHE_HOME='+linuxroot+'/.cache','HF_HOME='+linuxroot+'/.cache/huggingface',
                  linuxroot+'/.runtime/bin/micromamba','run','-p',linuxroot+'/.runtime/steve1',
                  'python','-m','minecraftrl.cli','session','--root',linuxroot,'--policy','steve1',
                  '--seed',str(seed),'--limit-s','180','--no-progress-s','60','--steve-prompt-profile','short',
                  '--world-slot',slot,'--world-store-root',linux(trial/'worlds'),'--output',linux(folder),'--stop-file',linux(stop)]
            def forward_stop():
                while not finished.wait(.02):
                    if global_stop.exists(): stop.touch(); return
            print(json.dumps(dict(starting=seed)),flush=True)
            with (trial/'supervisor.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                bridge=threading.Thread(target=forward_stop); bridge.start()
                try:
                    console=SessionDialogue(folder,stop_file=global_stop)
                    def ready():
                        try: console.sync(); return True
                        except ValueError as exc:
                            if str(exc)=='session_not_ready': return False
                            raise
                    wait_for(ready,proc)
                    restored=console.sync(); require(restored['status']=='paused','expected paused source')
                    write_command(folder/'commands',dict(command_id='setup-dirt',task_id=restored['task_id'],op='modify_plan',
                                  expected_revision=restored['revision'],steps=[{'goal':'dirt','count':1}]))
                    wait_for(lambda: console.sync()['revision']==restored['revision']+1,proc,10)
                    task=console.sync()
                    write_command(folder/'commands',dict(command_id='begin-resume',task_id=task['task_id'],op='resume',expected_revision=task['revision']))
                    deadline=time.monotonic()+210; position=0; partial=b''; exhausted=False
                    while proc.poll() is None:
                        if global_stop.exists(): raise RuntimeError('global STOP requested')
                        with (folder/'events.jsonl').open('rb') as stream:
                            stream.seek(position); chunk=stream.read(); position=stream.tell()
                        lines=(partial+chunk).split(b'\n'); partial=lines.pop()
                        for line in lines:
                            event=json.loads(line)
                            if event['event']=='task_no_progress':
                                command=budget.on_stall(event['task'])
                                if command is None:
                                    exhausted=True; stop.touch(); break
                                if global_stop.exists(): raise RuntimeError('global STOP requested')
                                write_command(folder/'commands',command)
                                atomic_json(trial/'retry-request.json',dict(before=budget.before,command=command))
                                print(json.dumps(dict(seed=seed,retry_requested=1,revision=event['task']['revision'],progress=event['task']['progress'])),flush=True)
                            if event['event']=='command_acknowledged' and event.get('command_id')=='retry-once':
                                assert_preserved(budget.before,event['task'])
                        if exhausted: break
                        if time.monotonic()>deadline: raise RuntimeError('control supervision timeout')
                        time.sleep(.05)
                    proc.wait(timeout=90)
                finally:
                    stop.touch()
                    try: proc.wait(timeout=90)
                    finally: finished.set(); bridge.join(timeout=2)
            report=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            rows=[json.loads(x) for x in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            retries=[e for e in rows if e['event']=='command_acknowledged' and e.get('command_id')=='retry-once']
            stalls=[e for e in rows if e['event']=='task_no_progress']
            resets=[e for e in rows if e['event']=='policy_reset_finished']
            outcome=('recovered_after_retry' if retries else 'initial_completed') if report.get('task_completed') else 'not_completed'
            result=dict(seed=seed,outcome=outcome,retries=len(retries),stalls=len(stalls),policy_resets=len(resets),report=report)
            state['reports'].append(result); save()
            require(report.get('world_restore',{}).get('inventory_match') is True,'restore inventory mismatch')
            require(report['release']=='neutral_command_sent' and report.get('world_save',{}).get('status')=='saved','release or save failed')
            require(report.get('process_cleanup',{}).get('remaining_live_pids')==[],'owned processes survived')
            require(report['status'] in ('task_completed','cancelled','time_limit','environment_done') and 'error' not in report,'technical worker failure')
            require(report['environment_resets']==1 and sum(e['event']=='episode_started' for e in rows)==1,'world restarted')
            require(len(retries)<=1 and len(stalls)<=2,'retry budget exceeded')
            require(resets and {e['worker_pid'] for e in resets}=={report['worker_pid']},'worker identity changed')
            commands=[e for e in rows if e['event']=='command_acknowledged' and e.get('op')=='resume']
            dispatches=[e for e in rows if e['event']=='policy_dispatch_started']
            require(not any(e['monotonic_s']<commands[0]['monotonic_s'] for e in dispatches),'dispatch before initial resume')
            for stall in stalls:
                next_resume=next((e['monotonic_s'] for e in commands if e['monotonic_s']>stall['monotonic_s']),float('inf'))
                require(not any(stall['monotonic_s']<e['monotonic_s']<next_resume for e in dispatches),'dispatch while paused')
            stopped=next(e['monotonic_s'] for e in rows if e['event']=='stop_acknowledged')
            require(not any(e['monotonic_s']>stopped for e in dispatches),'dispatch after STOP')
            if retries:
                assert_preserved(stalls[0]['task'],retries[0]['task'])
                result['retry_task_preserved']=True
                result['reset_after_retry']=any(e['monotonic_s']>retries[0]['monotonic_s'] for e in resets)
                result['condition_resampled']=len(resets)>1 and resets[0]['condition']['embedding_sha256']!=resets[1]['condition']['embedding_sha256']
                if report.get('task_completed') and not result['reset_after_retry']:
                    result['outcome']='completed_on_resume_observation'
            result['control_checks_passed']=True; save()
            print(json.dumps(dict(seed=seed,outcome=result['outcome'],retries=len(retries),stalls=len(stalls),steps=report['steps'])),flush=True)
        state['completed']=True
    except Exception as exc:
        state['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        global_stop.write_text('In-place recovery experiment ended. Inspect results before re-arming.',encoding='utf-8')
        state['source_unchanged']=((source/'current.json').read_bytes()==original_pointer and
                                  digest(generation/'manifest.json')==manifest_hash and digest(generation/'world.zip')==archive_hash)
        state['unrun']=len(seeds)-len(state['reports']); save()
    print(json.dumps(dict(completed=state['completed'],runs=len(state['reports']),errors=state['errors'])),flush=True)
    return 0 if state['completed'] and state['source_unchanged'] else 2


if __name__=='__main__': raise SystemExit(main())
