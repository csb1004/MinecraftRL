"""Summarize independent recipe probes, preserving technical/policy distinctions."""
import argparse
import json
from pathlib import Path
from minecraftrl.memory import atomic_json


def craft_stat_baseline(first):
    baseline=dict(first['observation']['craft_item'])
    gui=first['info'].get('craft_gui',{})
    if gui.get('server_available')==1:
        for name,item in (('planks','oak_planks'),('sticks','stick'),('table','crafting_table')):
            if 'server_crafted_'+name in gui: baseline[item]=gui['server_crafted_'+name]
    return baseline


def analyze(base):
    state=json.loads((base/'experiment.json').read_text(encoding='utf-8'))
    recipe=state['recipe_specification']; target=recipe['target']; name=state['recipe']
    rows=[]
    for item in state['reports']:
        report=item['report']; folder=base/str(item['seed'])/'run'
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()] if (folder/'events.jsonl').exists() else []
        transitions=[e for e in events if e['event']=='transition']
        baseline=json.loads((folder/'craft-baseline.json').read_text(encoding='utf-8')) if (folder/'craft-baseline.json').exists() else {}
        peak=max((e['observation']['inventory'].get(target,0) for e in transitions),default=0)
        crafted=max((e['observation']['craft_item'].get(target,0)-baseline.get('initial_crafted',0) for e in transitions),default=0)
        initial_stats=craft_stat_baseline(transitions[0]) if transitions else {}
        other_crafted={}
        for event in transitions:
            for key,value in event['observation']['craft_item'].items():
                delta=value-initial_stats.get(key,0)
                if key!=target and delta>0: other_crafted[key]=max(delta,other_crafted.get(key,0))
        first=next((e for e in transitions if e['observation']['inventory'].get(target,0)+
                    e['info']['craft_gui']['cursor_'+name]+e['info']['craft_gui']['grid_'+name]>=recipe['count']),None)
        after=[e for e in events if first and e['monotonic_s']>first['monotonic_s'] and
               e['event'] in ('inference_started','policy_dispatch_started')]
        cleanup=report.get('craft_cleanup') or {}; valid=item.get('control_checks_passed',False)
        success=valid and report.get('task_success_after_cleanup',False)
        if not valid: failure='technical_failure_or_cancelled'
        elif success: failure=None
        elif cleanup.get('after',{}).get('inventory',{}).get(target,0)>recipe['count']:
            failure='overproduced_target_consuming_reserved_material'
        elif first: failure='target_generated_but_exact_recipe_or_recovery_not_satisfied'
        elif crafted>0: failure='crafted_but_target_quantity_not_recovered'
        elif other_crafted: failure='crafted_a_different_recipe'
        elif report.get('gui_open_steps',0): failure='gui_opened_without_target_crafting'
        else: failure='no_gui_open_observed'
        start=next((e['monotonic_s'] for e in events if e['event']=='episode_started'),None)
        rows.append(dict(seed=item['seed'],valid_policy_trial=valid,success=success,failure=failure,
            guarded_policy_steps=report.get('guarded_policy_steps',0),
            target_quantity_saved=bool(valid and report.get('saved_inventory_verified') and
                cleanup.get('after',{}).get('inventory',{}).get(target,0)>=recipe['count'] and
                any(a['gui'].get('server_crafted_'+name,0)-baseline.get('initial_crafted',0)>=recipe['count']
                    for a in cleanup.get('actions',[]))),
            steps=report.get('steps'),status=report.get('status'),elapsed_control_s=report.get('elapsed_control_s'),
            first_material_step=first['step'] if first else None,
            first_material_s=first['monotonic_s']-start if first and start else None,
            new_policy_events_after_material=len(after),peak_inventory_target=peak,peak_crafted_delta=crafted,
            other_crafted_delta_since_first_transition=other_crafted,
            other_craft_stat_baseline='first-transition server counters for monitored items; client counters for others',
            final_inventory=cleanup.get('after',{}).get('inventory'),
            cursor_assistance=any(a['label']=='place_carried_stack' for a in cleanup.get('actions',[])),
            saved_inventory_verified=report.get('saved_inventory_verified',False),
            controller=report.get('controller','policy'),gui_decision_ms=report.get('gui_decision_ms'),
            inference_ms=report.get('inference_ms'),control_cycle_ms=report.get('control_cycle_ms'),
            gpu=report.get('gpu'),device_vram=report.get('device_vram'),
            release=report.get('release'),process_cleanup=report.get('process_cleanup')))
    result=dict(recipe=name,prompt=recipe['goal'],no_quick_move=state.get('no_quick_move',False),
                result_slot_guard=state.get('result_slot_guard',False),recipe_book=state.get('recipe_book',False),
                controller=state.get('controller','steve1'),completed=state['completed'],planned=state['planned'],
                attempted=len(state['reports']),unrun=state.get('unrun'),errors=state['errors'],
                source_unchanged=state.get('source_unchanged'),api_calls=state['api_calls'],
                valid_policy_trials=sum(r['valid_policy_trial'] for r in rows),
                target_quantity_saved_trials=sum(r['target_quantity_saved'] for r in rows),
                task_successes=sum(r['success'] for r in rows),trials=rows)
    atomic_json(base/'recipe-analysis.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    result=analyze(parser.parse_args().folder)
    print(json.dumps(result,indent=2))
