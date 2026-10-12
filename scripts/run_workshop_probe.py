"""Isolated, fixed-seed workstation trials from naturally collected materials."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
from minecraftrl.companion import worker_command
from minecraftrl.memory import atomic_json
from minecraftrl.world_store import digest
from minecraftrl.saved_inventory import verify_saved_inventory
from minecraftrl.saved_block import saved_block


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--goals',nargs='+',choices=['wooden_pickaxe','wooden_axe'],default=['wooden_pickaxe','wooden_axe'])
    parser.add_argument('--stop-at-stage',choices=['store_wooden_pickaxe'],help='Separate cancellation trial while a crafted tool is on the cursor')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];stop=root/'results/STOP'
    if not stop.exists():raise SystemExit('STOP must be latched before starting an isolated experiment')
    source=args.source.resolve()
    if not source.is_relative_to(root/'results'):raise SystemExit('source must be project-local results')
    receipt=json.loads((source/'collection.json').read_text());assert receipt['completed']
    slot=receipt['slot'];generation=receipt['generation'];original=source/'worlds'/slot/generation
    hashes={name:digest(original/name) for name in ('world.zip','manifest.json')}
    assert verify_saved_inventory(original/'world.zip',receipt['inventory'])['verified']
    configured=json.loads((root/'configs/experiment.json').read_text())['seeds'];seeds=args.seeds or configured
    if len(seeds)!=len(set(seeds)) or not set(seeds)<=set(configured):raise SystemExit('invalid fixed seed subset')
    if args.stop_at_stage and len(seeds)!=1:raise SystemExit('cancellation probe requires one seed')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base=root/'results'/('workshop-'+stamp);base.mkdir();print(base,flush=True)
    state=dict(completed=False,source=str(source),source_hashes=hashes,goals=args.goals,seeds=seeds,planned=len(seeds),
               controller='workshop_gui',learned_policy_loaded=False,api_calls=0,limit_s=180,
               stop_at_stage=args.stop_at_stage,reports=[],errors=[])
    def save():atomic_json(base/'experiment.json',state)
    save();stop.rename(base/'STOP-before.txt');linuxroot='/mnt/d/DATA/Desktop/MinecraftRL'
    try:
        for seed in seeds:
            if stop.exists():raise RuntimeError('global STOP requested')
            trial=base/str(seed);clone=trial/'worlds'/slot;clone.mkdir(parents=True)
            shutil.copytree(original,clone/generation)
            atomic_json(clone/'current.json',dict(generation=generation,manifest_sha256=hashes['manifest.json']))
            folder=trial/'run'
            argv=worker_command(linuxroot,'Ubuntu-24.04',slot,seed,180,folder.relative_to(root).as_posix())
            argv[argv.index('session')]='craft-probe'
            argv+=['--world-store-root',linuxroot+'/'+(trial/'worlds').relative_to(root).as_posix(),
                   '--condition','table','--craft-recover-cursor','--craft-workshop','--workshop-goals',*args.goals]
            print(json.dumps(dict(starting=seed)),flush=True)
            with (trial/'supervisor.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    if not args.stop_at_stage:proc.wait(timeout=870)
                    else:
                        deadline=time.monotonic()+870;offset=0;trigger=None
                        while proc.poll() is None:
                            if time.monotonic()>deadline:raise TimeoutError('cancellation probe deadline')
                            path=folder/'events.jsonl'
                            if path.exists() and trigger is None:
                                with path.open(encoding='utf-8') as events_file:
                                    events_file.seek(offset)
                                    while True:
                                        start=events_file.tell();line=events_file.readline()
                                        if not line or not line.endswith('\n'):offset=start;break
                                        offset=events_file.tell();event=json.loads(line)
                                        if event['event']=='transition' and event['info'].get('workshop_stage')==args.stop_at_stage and event['info']['workshop']['cursor_count']>0:
                                            trigger=dict(step=event['step'],stage=args.stop_at_stage,cursor=event['info']['workshop']['cursor_item'])
                                            atomic_json(trial/'stop-trigger.json',trigger)
                                            stop.write_text('Controlled workshop cancellation probe',encoding='utf-8');break
                            time.sleep(.02)
                        if trigger is None:raise RuntimeError('cancellation boundary not observed')
                finally:
                    if proc.poll() is None:stop.touch();proc.wait(timeout=90)
            r=json.loads((folder/'summary.json').read_text());state['reports'].append(dict(seed=seed,report=r));save()
            events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
            if not events:raise RuntimeError('startup failed before any episode action: '+r.get('error',r['status']))
            ack=next(e['monotonic_s'] for e in events if e['event']=='stop_acknowledged')
            assert not any(e['event']=='policy_dispatch_started' and e['monotonic_s']>ack for e in events)
            assert r['world_restore']['inventory_match'] and r['world_save']['status']=='saved'
            check=verify_saved_inventory(clone/r['world_save']['generation']/'world.zip',r['workshop_cleanup']['inventory'])
            atomic_json(trial/'saved-inventory-verification.json',check)
            r['saved_inventory_verified']=check['verified']
            assert check['verified'] and r['release']=='neutral_command_sent' and not r['process_cleanup']['remaining_live_pids']
            if args.stop_at_stage:
                assert r['status']=='cancelled' and not r['workshop_success'] and r['stop_reason']=='stop_file'
                assert r['workshop_cleanup']['cursor_empty'] and r['workshop_cleanup']['inventory']=={'wooden_pickaxe':1,'oak_planks':3,'stick':2}
            else:
                assert proc.returncode==0 and r['execution_valid'] and r['workshop_success'],r.get('error','workshop incomplete')
                reached=next(e['monotonic_s'] for e in events if e['event']=='target_reached')
                assert not any(e['event'] in ('policy_dispatch_started','inference_started') and e['monotonic_s']>reached for e in events)
                assert r['milestones'][-1]['event']=='completed' and r['milestones'][-1]['step']==r['steps']
            block=saved_block(clone/r['world_save']['generation']/'world.zip',r['table_position'])
            atomic_json(trial/'saved-workstation-verification.json',dict(position=r['table_position'],block=block))
            r['saved_workstation_verified']=block=='minecraft:crafting_table'
            atomic_json(folder/'summary.json',r);save()
            assert block=='minecraft:crafting_table','placed workstation not preserved in saved chunk'
            print(json.dumps(dict(seed=seed,steps=r['steps'],success=r['workshop_success'],inventory=r['workshop_cleanup']['inventory'])),flush=True)
        state['completed']=True
    except Exception as exc:state['errors'].append(repr(exc))
    finally:
        stop.write_text('Workshop experiment ended; review before re-arming.',encoding='utf-8')
        state['source_unchanged']=all(digest(original/name)==value for name,value in hashes.items())
        state['unrun']=len(seeds)-len(state['reports']);save()
    print(json.dumps(dict(completed=state['completed'],errors=state['errors'])),flush=True)
    return 0 if state['completed'] and state['source_unchanged'] else 2


if __name__=='__main__':raise SystemExit(main())
