"""Analyze crafting observations without conflating policy and cleanup recovery."""
import argparse
import json
from pathlib import Path
from minecraftrl.memory import atomic_json


def active(value):
    return any(active(v) for v in value) if isinstance(value,list) else bool(value)


def analyze(base):
    base=Path(base); state=json.loads((base/'experiment.json').read_text(encoding='utf-8'))
    output=dict(completed=state['completed'],api_calls=state['api_calls'],planned=state['planned'],
                source_unchanged=state.get('source_unchanged'),unrun=state.get('unrun'),errors=state['errors'],
                stop_on_success=state.get('stop_on_success',False),recover_cursor=state.get('recover_cursor',False),trials=[])
    for row in state['reports']:
        folder=base/str(row['seed'])/'run'; report=row['report']
        baseline=json.loads((folder/'craft-baseline.json').read_text(encoding='utf-8'))
        first_gui=last_gui=None; gui_steps=inventory_presses=attack=movement=0
        max_crafted=0; final=None; minimum_logs=1; maximum_planks=0
        started=None; success_after_s=None; recovery_after_s=None; craft_baseline=None; craft_peaks={}
        with (folder/'events.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                event=json.loads(line)
                if event['event']=='episode_started': started=event['monotonic_s']
                if event['event']=='craft_ready_for_recovery': recovery_after_s=event['monotonic_s']-started
                if event['event']!='transition': continue
                obs=event['observation']; action=event['action']; step=event['step']
                inventory=obs['inventory']; final={k:v for k,v in inventory.items() if v}
                minimum_logs=min(minimum_logs,inventory.get('oak_log',0))
                maximum_planks=max(maximum_planks,inventory.get('oak_planks',0))
                max_crafted=max(max_crafted,obs['craft_item']['oak_planks']-baseline['initial_crafted'])
                if craft_baseline is None: craft_baseline=obs['craft_item']
                for key,value in obs['craft_item'].items():
                    delta=value-craft_baseline.get(key,0)
                    if delta>0: craft_peaks[key]=max(craft_peaks.get(key,0),delta)
                if report.get('craft_witness',{} ) and step==report['craft_witness']['step']:
                    success_after_s=event['monotonic_s']-started
                if event['info']['isGuiOpen']:
                    gui_steps+=1; last_gui=step
                    if first_gui is None: first_gui=step
                inventory_presses+=int(active(action['inventory'])); attack+=int(active(action['attack']))
                movement+=int(any(active(action[k]) for k in ('forward','back','left','right','jump')))
        valid=row.get('control_checks_passed') is True
        if not valid: outcome='technical_failure'
        elif report['craft_success']: outcome='policy_crafted_and_recovered'
        elif report.get('cleanup_witness'): outcome='recovered_only_during_cleanup'
        elif max_crafted>0: outcome='craft_stat_increased_without_full_inventory_proof'
        elif first_gui is None: outcome='no_gui_open_observed'
        else: outcome='gui_opened_but_no_plank_craft_observed'
        output['trials'].append(dict(seed=row['seed'],outcome=outcome,steps=report['steps'],
            elapsed_control_s=report.get('elapsed_control_s'),first_gui_step=first_gui,last_gui_step=last_gui,
            gui_open_steps=gui_steps,inventory_key_steps=inventory_presses,attack_steps=attack,movement_steps=movement,
            crafted_plank_increase=max_crafted,minimum_oak_logs=minimum_logs,maximum_oak_planks=maximum_planks,
            final_inventory=final,craft_witness=report.get('craft_witness'),cleanup_witness=report.get('cleanup_witness'),
            status=report['status'],planks_retained_after_cleanup=report.get('planks_retained_after_cleanup',False),
            task_success_after_cleanup=report.get('task_success_after_cleanup',False),
            inventory_four_before_cleanup=(report.get('craft_cleanup') or {}).get('before',{}).get('inventory',{}).get('oak_planks',0)>=4,
            recovery_after_s=recovery_after_s,recovery_trigger=report.get('recovery_trigger'),craft_state=report.get('craft_state'),
            cursor_placement_assistance=any(a.get('label')=='place_carried_stack' for a in (report.get('craft_cleanup') or {}).get('actions',[])),
            post_cleanup_inventory=(report.get('craft_cleanup') or {}).get('after',{}).get('inventory'),
            success_after_s=success_after_s,craft_stat_peak_delta_from_first_transition=craft_peaks,
            inference_ms=report.get('inference_ms'),control_cycle_ms=report.get('control_cycle_ms'),
            gpu=report.get('gpu'),device_vram=report.get('device_vram'),control_checks_passed=valid))
    output['policy_successes']=sum(t['outcome']=='policy_crafted_and_recovered' for t in output['trials'])
    output['valid_policy_trials']=sum(t['control_checks_passed'] for t in output['trials'])
    output['retained_after_cleanup_trials']=sum(t['control_checks_passed'] and t['planks_retained_after_cleanup'] for t in output['trials'])
    output['task_successes_after_cleanup']=sum(t['control_checks_passed'] and t['task_success_after_cleanup'] for t in output['trials'])
    output['inventory_four_before_cleanup_trials']=sum(t['control_checks_passed'] and t['inventory_four_before_cleanup'] for t in output['trials'])
    output['cursor_placement_trials']=sum(t['control_checks_passed'] and t['cursor_placement_assistance'] for t in output['trials'])
    atomic_json(base/'analysis.json',output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('folder',type=Path)
    result=analyze(parser.parse_args().folder)
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))
