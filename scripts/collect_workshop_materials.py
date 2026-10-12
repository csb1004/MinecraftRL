"""Collect a natural material save with STEVE-1 before assisted workshop trials."""
import datetime
import json
from pathlib import Path
import subprocess
import time
from minecraftrl.companion import worker_command
from minecraftrl.task_runtime import write_command
from minecraftrl.saved_inventory import verify_saved_inventory
from minecraftrl.memory import atomic_json


def main():
    root=Path(__file__).resolve().parents[1];stop=root/'results/STOP'
    if not stop.exists():raise RuntimeError('STOP must be latched before collecting materials')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    slot='workshop-'+stamp;base=root/'results'/('workshop-materials-'+stamp);base.mkdir()
    folder=base/'run';linuxroot='/mnt/d/DATA/Desktop/MinecraftRL'
    argv=worker_command(linuxroot,'Ubuntu-24.04',slot,104729,180,folder.relative_to(root).as_posix())
    argv+=['--world-store-root',linuxroot+'/'+(base/'worlds').relative_to(root).as_posix()]
    stop.rename(base/'STOP-before.txt');print(base,flush=True)
    state=dict(slot=slot,seed=104729,requested_logs=3,controller='steve1',api_calls=0,completed=False)
    atomic_json(base/'collection.json',state)
    try:
        with (base/'supervisor.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline=time.monotonic()+600
                while not (folder/'commands').is_dir():
                    if stop.exists() or proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError('collection startup did not become ready')
                    time.sleep(.1)
                write_command(folder/'commands',dict(op='request',command_id='natural-materials',task_id='workshop-materials',goal='wood',count=3))
                proc.wait(timeout=270)
            finally:
                if proc.poll() is None:stop.touch();proc.wait(timeout=90)
        r=json.loads((folder/'summary.json').read_text());state['report']=r
        assert proc.returncode==0 and r['execution_valid'] and r['task_completed']
        generation=r['world_save']['generation'];archive=base/'worlds'/slot/generation/'world.zip'
        manifest=json.loads((archive.parent/'manifest.json').read_text())
        inv=manifest['observation']['inventory'];check=verify_saved_inventory(archive,inv)
        assert check['verified'] and sum(v for k,v in inv.items() if k.endswith('_log'))>=3
        state.update(completed=True,generation=generation,inventory=inv,saved_inventory=check)
    except Exception as exc:state['error']=repr(exc)
    finally:
        stop.write_text('Material collection finished; inspect before re-arming.',encoding='utf-8')
        atomic_json(base/'collection.json',state)
    print(json.dumps(state),flush=True)
    return 0 if state['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
