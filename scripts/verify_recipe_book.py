"""Audit actual GUI clicks, saved inventory, unlock transitions, and stop boundaries."""
import argparse
import json
from pathlib import Path
from minecraftrl.recipe_book import verify_grid,require_recipe_button
from minecraftrl.saved_inventory import verify_saved_inventory
from minecraftrl.world_store import digest
from inspect_recipe_unlocks import inspect


def verify(base,jar,allow_incomplete=False):
    state=json.loads((base/'experiment.json').read_text(encoding='utf-8'))
    assert state['recipe_book'] and state['source_unchanged']
    assert (state['completed'] and not state['errors']) or (allow_incomplete and not state['completed'] and state['errors'])
    assert state['unrun']==0 and state['api_calls']==0
    source=Path(state['source_path'])/state['source_generation']
    assert all(digest(source/name)==sha for name,sha in state['source_hashes'].items())
    recipe=state['recipe']; target=state['recipe_specification']['target']; rows=[];excluded=[]
    for trial in state['reports']:
        report=trial['report'];folder=base/str(trial['seed'])/'run'
        if not trial.get('control_checks_passed'):
            assert allow_incomplete and not report['execution_valid'] and report['steps']==0
            excluded.append(dict(seed=trial['seed'],status=report['status'],reason='technical initialization failure',
                                 counted_as_crafting_trial=False))
            continue
        assert trial['control_checks_passed'] and report['execution_valid'] and report['task_success_after_cleanup']
        assert report['controller']=='recipe_book_gui' and report['learned_policy_loaded'] is False
        assert 'inference_ms' not in report and report['gui_decision_ms']['count']>=report['steps']
        assert report['cursor_item_verified'] and not report['craft_cleanup']['gui_open_after']
        assert report['release']=='neutral_command_sent' and report['process_cleanup']['remaining_live_pids']==[]
        archive=base/str(trial['seed'])/'worlds'/state['source_slot']/report['world_save']['generation']/'world.zip'
        expected={'dirt':1, 'oak_planks':2, 'stick':4} if recipe=='sticks' else {'dirt':1,'crafting_table':1}
        assert verify_saved_inventory(archive,expected)['verified']
        unlocked=inspect(archive,jar)
        (folder.parent/'unlocks.json').write_text(json.dumps(unlocked,indent=2),encoding='utf-8')
        assert all(p['target_unlocks']['wooden_pickaxe']==(recipe=='sticks') for p in unlocked['players'])
        prior=None;stopped=False;selection=output=0;start=last=None;count=0
        for line in (folder/'events.jsonl').open(encoding='utf-8'):
            e=json.loads(line)
            if stopped:assert e['event'] not in ('inference_started','policy_dispatch_started')
            if e['event'] in ('craft_ready_for_recovery','stop_acknowledged'):stopped=True
            if e['event']=='episode_started':start=e['monotonic_s']
            if e['event']=='policy_dispatch_started':last=e['monotonic_s']
            if e['event']!='transition':continue
            count+=1;a=e['action'];g=e['info']['craft_gui']
            assert a['sneak']==0 and a['use']==0 and a['drop']==0
            if prior is None:
                assert g['server_known_wooden_pickaxe']==0
                assert all(g[f'input_{i}_count']==0 for i in range(1,5))
            if a['attack']:
                assert prior is not None and a['camera']==[0.0,0.0]
                x,y=prior['mouse_x'],prior['mouse_y']
                if prior['book_open'] and (x,y)==(prior['book_'+target+'_x'],prior['book_'+target+'_y']):
                    assert require_recipe_button(prior,target)==(x,y);selection+=1
                elif (x,y)==(prior['book_toggle_x'],prior['book_toggle_y']):pass
                elif not prior['book_open'] and (x,y)==(prior['result_x'],prior['result_y']):
                    verify_grid(prior,recipe);output+=1
                else:raise AssertionError('unexpected click outside book/verified output')
            prior=g
        assert stopped and count==report['steps']==report['recovery_trigger']['step']
        assert selection==1 and output==1 and last-start<=180
        rows.append(dict(seed=trial['seed'],steps=count,book_selections=selection,result_clicks=output,
                         saved_inventory=expected,saved_nbt_verified=True,wooden_pickaxe_unlocked=(recipe=='sticks'),
                         last_dispatch_s=last-start,no_actions_after_completion=True))
    result={'batch':str(base),'recipe':recipe,'verified_trials':len(rows),'trials':rows,'excluded_attempts':excluded,
            'source_unchanged':True,'api_calls':0,'controller':'explicit recipe-book GUI assistance'}
    (base/'book-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--jar',required=True,type=Path)
    p.add_argument('--allow-incomplete',action='store_true',help='Audit completed trials while explicitly excluding zero-action initialization failures')
    a=p.parse_args();print(json.dumps(verify(a.folder,a.jar,a.allow_incomplete),indent=2))
