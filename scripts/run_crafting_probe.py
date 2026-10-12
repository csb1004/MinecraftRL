"""Five fixed-seed recipe trials on isolated copies of saved material worlds."""
import datetime
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from minecraftrl.companion import worker_command
from minecraftrl.memory import atomic_json
from minecraftrl.world_store import WorldStore,digest
from minecraftrl.saved_inventory import verify_saved_inventory
from minecraftrl.crafting_probe import RECIPES


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stop-on-success',action='store_true')
    parser.add_argument('--recover-cursor',action='store_true')
    parser.add_argument('--no-quick-move',action='store_true')
    parser.add_argument('--result-slot-guard',action='store_true')
    parser.add_argument('--recipe-book',action='store_true')
    parser.add_argument('--recipe',choices=tuple(RECIPES),default='planks')
    parser.add_argument('--seeds',type=int,nargs='+',help='Ordered subset of the fixed experiment seeds')
    options=parser.parse_args()
    if options.recipe_book and not (options.recover_cursor and options.recipe in ('sticks','table')):
        raise SystemExit('recipe-book requires sticks/table and cursor recovery')
    if options.recipe_book and (options.no_quick_move or options.result_slot_guard):raise SystemExit('select recipe-book alone')
    if options.no_quick_move and not options.recover_cursor: raise SystemExit('action guard requires cursor recovery')
    if options.result_slot_guard and not options.recover_cursor: raise SystemExit('action guard requires cursor recovery')
    if options.result_slot_guard and options.no_quick_move: raise SystemExit('select one action guard')
    root=Path(__file__).resolve().parents[1]; stop=root/'results/STOP'
    if not stop.exists(): raise SystemExit('STOP must be latched; finish the active session first')
    slot='sequence-20261004T025836469371Z'; generation='28222939e73444be84ba333eedf2d4d8'
    source=root/'data/worlds'/slot; original=source/generation
    recipe=RECIPES[options.recipe]
    if options.recipe!='planks':
        if not options.recover_cursor: raise SystemExit('new recipes require --recover-cursor')
        prior=root/'results/craft-recovery-20261004T071020156492Z'
        prior_state=json.loads((prior/'experiment.json').read_text(encoding='utf-8'))
        saved=next(row['report'] for row in prior_state['reports'] if row['seed']==104729)
        assert saved['task_success_after_cleanup']
        generation=saved['world_save']['generation']
        source=prior/'104729/worlds'/slot; original=source/generation
        assert verify_saved_inventory(original/'world.zip',{'dirt':1,'oak_planks':4})['verified']
    pointer=(source/'current.json').read_bytes(); hashes={name:digest(original/name) for name in ('manifest.json','world.zip')}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix='craft-recovery-' if options.recover_cursor else 'craft-retention-' if options.stop_on_success else 'crafting-'
    if options.recipe!='planks': prefix='craft-'+options.recipe+'-'
    if options.no_quick_move: prefix+='no-quick-move-'
    if options.result_slot_guard: prefix+='result-slot-guard-'
    if options.recipe_book: prefix+='recipe-book-'
    base=root/'results'/(prefix+stamp); base.mkdir()
    seeds=json.loads((root/'configs/experiment.json').read_text())['seeds']
    if options.seeds:
        if len(set(options.seeds))!=len(options.seeds) or not set(options.seeds)<=set(seeds):
            raise SystemExit('seeds must be a unique subset of configured fixed seeds')
        seeds=options.seeds
    state=dict(completed=False,api_calls=0,seeds=seeds,planned=len(seeds),source_slot=slot,source_generation=generation,
               source_path=str(source),recipe=options.recipe,recipe_specification=recipe,
               source_hashes=hashes,prompt=recipe['goal'],limit_s=180,stop_on_success=options.stop_on_success,
               recover_cursor=options.recover_cursor,no_quick_move=options.no_quick_move,
               result_slot_guard=options.result_slot_guard,recipe_book=options.recipe_book,
               controller='recipe_book_gui' if options.recipe_book else 'steve1',reports=[],errors=[])
    def save(): atomic_json(base/'experiment.json',state)
    save(); print(base,flush=True)
    linuxroot='/mnt/d/DATA/Desktop/MinecraftRL'
    stop.rename(base/'STOP-before.txt')
    try:
        for seed in seeds:
            if stop.exists(): raise RuntimeError('global STOP requested')
            trial=base/str(seed); clone=trial/'worlds'/slot; clone.mkdir(parents=True)
            shutil.copytree(original,clone/generation)
            atomic_json(clone/'current.json',dict(generation=generation,manifest_sha256=hashes['manifest.json']))
            with WorldStore(trial/'worlds',slot) as store:
                bundle=store.load()
                inventory=bundle['observation']['inventory']
                assert inventory.get(recipe['ingredient'])==recipe['initial'] and not inventory.get(recipe['target'],0)
            folder=trial/'run'
            argv=worker_command(linuxroot,'Ubuntu-24.04',slot,seed,180,folder.relative_to(root).as_posix())
            argv[argv.index('session')]='craft-probe'
            argv+=['--world-store-root',linuxroot+'/'+(trial/'worlds').relative_to(root).as_posix()]
            argv+=['--condition',options.recipe]
            if options.stop_on_success: argv.append('--craft-stop-on-success')
            if options.recover_cursor: argv.append('--craft-recover-cursor')
            if options.no_quick_move: argv.append('--craft-no-quick-move')
            if options.result_slot_guard: argv.append('--craft-result-slot-guard')
            if options.recipe_book: argv.append('--craft-recipe-book')
            print(json.dumps(dict(starting=seed)),flush=True)
            with (trial/'supervisor.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                try: proc.wait(timeout=870)
                finally:
                    if proc.poll() is None: stop.touch(); proc.wait(timeout=90)
            report=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            item=dict(seed=seed,report=report); state['reports'].append(item); save()
            events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            ack=next(e['monotonic_s'] for e in events if e['event']=='stop_acknowledged')
            assert not any(e['event']=='policy_dispatch_started' and e['monotonic_s']>ack for e in events)
            assert report['execution_valid'] and proc.returncode==0,'technical execution failure'
            if options.no_quick_move:
                assert report['no_quick_move']
                transitions=[e for e in events if e['event']=='transition']
                assert transitions and all(e['info']['craft_action_guard']['forwarded_action']['sneak']==0 for e in transitions)
                for e in transitions:
                    audit=e['info']['craft_action_guard']
                    assert audit['proposed_action']==e['action']
                    assert {k:v for k,v in audit['forwarded_action'].items() if k!='sneak'}=={k:v for k,v in e['action'].items() if k!='sneak'}
            if options.result_slot_guard:
                assert report['result_slot_guard']
                transitions=[e for e in events if e['event']=='transition']
                assert transitions
                for e in transitions:
                    audit=e['info']['craft_action_guard']
                    assert audit['mode']=='result_slot_only' and audit['proposed_action']==e['action']
                    assert {k:v for k,v in audit['forwarded_action'].items() if k!='sneak'}=={k:v for k,v in e['action'].items() if k!='sneak'}
                    if audit['reason'] in ('result_slot','gui_transition','bootstrap'):
                        assert audit['forwarded_action']['sneak']==0
                    else:assert audit['forwarded_action']['sneak']==e['action']['sneak']
            assert report['world_restore']['inventory_match'] and report['world_save']['status']=='saved'
            assert report['release']=='neutral_command_sent' and report['process_cleanup']['remaining_live_pids']==[]
            assert not report['craft_cleanup']['gui_open_after'],'GUI remained open after cleanup'
            if options.recover_cursor:
                assert report['cursor_item_verified'],'cursor was not verified empty'
                if report['recovery_trigger']:
                    assert report['status']=='craft_ready_for_recovery'
                    assert report['steps']==report['recovery_trigger']['step']
                    reached=next(e['monotonic_s'] for e in events if e['event']=='craft_ready_for_recovery')
                    assert not any(e['event'] in ('policy_dispatch_started','inference_started') and e['monotonic_s']>reached for e in events)
                verified=verify_saved_inventory(clone/report['world_save']['generation']/'world.zip',
                                                report['craft_cleanup']['after']['inventory'])
                atomic_json(trial/'saved-inventory-verification.json',verified)
                report['saved_inventory_verified']=verified['verified']
                if not verified['verified']:
                    report['task_success_after_cleanup']=False
                    report['craft_state']['accepting']=False
                    report['execution_valid']=False
                atomic_json(folder/'summary.json',report)
                assert verified['verified'],'saved NBT differs from final observation'
            elif options.stop_on_success and report['craft_success']:
                assert report['status']=='target_reached' and report['steps']==report['craft_witness']['step']
                reached=next(e['monotonic_s'] for e in events if e['event']=='target_reached')
                assert not any(e['event'] in ('policy_dispatch_started','inference_started') and e['monotonic_s']>reached for e in events)
                assert report['planks_retained_after_cleanup'],'target lost during cleanup'
            item['control_checks_passed']=True; save()
            print(json.dumps(dict(seed=seed,craft_success=report['craft_success'],task_success_after_cleanup=report['task_success_after_cleanup'],gui_open_steps=report['gui_open_steps'],steps=report['steps'])),flush=True)
        state['completed']=True
    except Exception as exc: state['errors'].append(repr(exc))
    finally:
        stop.write_text('Crafting probe ended. Review results before re-arming.',encoding='utf-8')
        state['source_unchanged']=(source/'current.json').read_bytes()==pointer and all(digest(original/name)==value for name,value in hashes.items())
        state['unrun']=len(seeds)-len(state['reports']); save()
    print(json.dumps(dict(completed=state['completed'],runs=len(state['reports']),errors=state['errors'])),flush=True)
    return 0 if state['completed'] and state['source_unchanged'] else 2


if __name__=='__main__': raise SystemExit(main())
