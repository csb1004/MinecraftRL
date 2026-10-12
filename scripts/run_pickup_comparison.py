"""Paired prompt comparison on immutable copies of one recorded failure world.

PowerShell/Windows launcher. No API calls and no manually supplied motor actions.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

from minecraftrl.dialogue_console import SessionDialogue
from minecraftrl.memory import atomic_json
from minecraftrl.task_runtime import write_command
from minecraftrl.world_store import WorldStore, digest


def main():
    root=Path(__file__).resolve().parents[1]; global_stop=root/'results/STOP'
    if not global_stop.exists(): raise SystemExit('STOP must be latched before starting')
    slot='failure-20261004T032512557525Z'
    source=root/'data/worlds'/slot
    with WorldStore(root/'data/worlds',slot) as store:
        bundle=store.load()
        if bundle is None or bundle['generation']!='3b2f51efcf7c4bbe8edfee14623eb5db':
            raise SystemExit('Expected immutable failure generation is not current; inspect before running')
        pointer=(source/'current.json').read_bytes()
        generation=source/bundle['generation']
        if bundle['task_checkpoint']['task']['goal']!='dirt': raise SystemExit('Expected dirt task')
        manifest_hash=digest(generation/'manifest.json'); archive_hash=digest(generation/'world.zip')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base=root/'results'/('pickup-comparison-'+stamp); base.mkdir()
    seeds=json.loads((root/'configs/experiment.json').read_text())['seeds']
    state=dict(completed=False,api_calls=0,source_slot=slot,source_generation=bundle['generation'],
               archive_sha256=archive_hash,manifest_sha256=manifest_hash,seeds=seeds,
               planned_runs=10,control_limit_s=180,no_progress_s=60,reports=[],errors=[],
               interpretation='Ten paired runtime-seed trials of one saved world, not ten independent terrain worlds.')
    atomic_json(base/'comparison.json',state)
    print(str(base),flush=True)
    linuxroot='/mnt/d/DATA/Desktop/MinecraftRL'
    def linux(path): return linuxroot+'/'+path.relative_to(root).as_posix()
    def events(folder):
        path=folder/'events.jsonl'
        return [json.loads(x) for x in path.read_text(encoding='utf-8').split('\n')[:-1]] if path.exists() else []
    def require(condition,message):
        if not condition: raise AssertionError(message)
    global_stop.rename(base/'STOP-before.txt')
    try:
        for i,seed in enumerate(seeds):
            for profile in (('short','official') if i%2==0 else ('official','short')):
                if global_stop.exists(): raise RuntimeError('global STOP requested')
                trial=base/(profile+'-'+str(seed)); trial.mkdir()
                clone=trial/'worlds'/slot; clone.mkdir(parents=True)
                shutil.copytree(generation,clone/bundle['generation'])
                (clone/'current.json').write_bytes(pointer)
                require(digest(clone/bundle['generation']/'world.zip')==archive_hash,'copied archive mismatch')
                folder=trial/'run'; stop=trial/'STOP'; finished=threading.Event()
                argv=['wsl.exe','-d','Ubuntu-24.04','--cd',linuxroot,'--','env','GALLIUM_DRIVER=d3d12',
                      'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+linuxroot+'/.runtime/mamba',
                      'XDG_CACHE_HOME='+linuxroot+'/.cache','HF_HOME='+linuxroot+'/.cache/huggingface',
                      linuxroot+'/.runtime/bin/micromamba','run','-p',linuxroot+'/.runtime/steve1',
                      'python','-m','minecraftrl.cli','session','--root',linuxroot,'--policy','steve1',
                      '--seed',str(seed),'--limit-s','180','--no-progress-s','60','--steve-prompt-profile',profile,
                      '--world-slot',slot,'--world-store-root',linux(trial/'worlds'),
                      '--output',linux(folder),'--stop-file',linux(stop)]
                def forward_stop():
                    while not finished.wait(.02):
                        if global_stop.exists(): stop.touch(); return
                print(json.dumps(dict(starting=len(state['reports'])+1,profile=profile,seed=seed)),flush=True)
                with (trial/'supervisor.log').open('w',encoding='utf-8') as log:
                    proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    bridge=threading.Thread(target=forward_stop); bridge.start()
                    try:
                        console=SessionDialogue(folder,stop_file=global_stop)
                        deadline=time.monotonic()+600
                        while True:
                            if global_stop.exists(): raise RuntimeError('global STOP requested')
                            try: task=console.sync(); break
                            except ValueError as exc:
                                if str(exc)!='session_not_ready': raise
                            if proc.poll() is not None or time.monotonic()>deadline: raise RuntimeError('startup failed; inspect trial logs')
                            time.sleep(.05)
                        require(task['status']=='paused' and task['failure']['reason']=='no_progress','wrong restored task')
                        write_command(folder/'commands',dict(command_id='resume',task_id=task['task_id'],op='resume',expected_revision=task['revision']))
                        deadline=time.monotonic()+210
                        event_position=0; partial=b''; stalled=False
                        while proc.poll() is None:
                            if global_stop.exists(): raise RuntimeError('global STOP requested')
                            with (folder/'events.jsonl').open('rb') as stream:
                                stream.seek(event_position); chunk=stream.read(); event_position=stream.tell()
                            lines=(partial+chunk).split(b'\n'); partial=lines.pop()
                            stalled=stalled or any(json.loads(line)['event']=='task_no_progress' for line in lines)
                            if stalled:
                                stop.touch(); break
                            if time.monotonic()>deadline: raise RuntimeError('control supervision timeout')
                            time.sleep(.1)
                        proc.wait(timeout=90)
                    finally:
                        stop.touch()
                        try: proc.wait(timeout=90)
                        finally: finished.set(); bridge.join(timeout=2)
                report=json.loads((folder/'summary.json').read_text(encoding='utf-8')); rows=events(folder)
                completed=report.get('task_completed',False)
                outcome='completed' if completed else ('no_progress' if report.get('no_progress_events') else report['status'])
                result=dict(profile=profile,seed=seed,outcome=outcome,report=report,folder=str(folder.relative_to(root)))
                state['reports'].append(result); atomic_json(base/'comparison.json',state)
                require(report.get('world_restore',{}).get('inventory_match') is True,'restored inventory mismatch')
                require(report.get('release')=='neutral_command_sent','input release missing')
                require(report.get('world_save',{}).get('status')=='saved','world save failed')
                require(report.get('process_cleanup',{}).get('remaining_live_pids')==[],'owned processes survived')
                require(outcome in ('completed','no_progress','time_limit','environment_done'),'technical failure: '+outcome)
                resumed=next(e['monotonic_s'] for e in rows if e['event']=='command_acknowledged' and e['op']=='resume')
                dispatches=[e for e in rows if e['event']=='policy_dispatch_started']
                require(not any(e['monotonic_s']<resumed for e in dispatches),'dispatch before resume')
                for boundary in (e for e in rows if e['event'] in ('task_no_progress','stop_acknowledged')):
                    require(not any(e['monotonic_s']>boundary['monotonic_s'] for e in dispatches),'dispatch after stall or STOP')
                conditions=[e['condition'] for e in rows if e['event']=='policy_reset_finished']
                require(len(conditions)==1 and conditions[0]['profile']==profile,'condition trace mismatch')
                result.update(condition=conditions[0],control_checks_passed=True)
                print(json.dumps(dict(finished=len(state['reports']),profile=profile,seed=seed,outcome=outcome,
                                      steps=report['steps'],progress=report['task']['progress'])),flush=True)
                atomic_json(base/'comparison.json',state)
        state['completed']=True
    except Exception as exc:
        state['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        global_stop.write_text('Pickup comparison ended. Inspect results before re-arming.',encoding='utf-8')
        state['source_unchanged']=((source/'current.json').read_bytes()==pointer and
                                   digest(generation/'manifest.json')==manifest_hash and digest(generation/'world.zip')==archive_hash)
        state['unrun']=10-len(state['reports'])
        atomic_json(base/'comparison.json',state)
    print(json.dumps(dict(completed=state['completed'],runs=len(state['reports']),unrun=state['unrun'],errors=state['errors'])),flush=True)
    return 0 if state['completed'] and state['source_unchanged'] else 2


if __name__=='__main__': raise SystemExit(main())
