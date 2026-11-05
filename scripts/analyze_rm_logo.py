"""Independent server-inventory audit of P3 prompts, generated RM and dispatches."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from minecraftrl.goal_contract import digest, read_json
from minecraftrl.logo_validators import SuccessMonitor
from minecraftrl.logo_wood import admit, stage
from minecraftrl.rm_conditioning import StageSelector
from minecraftrl.rm_logo_conditioning import load_bank
from minecraftrl.rm_purpose_conditioning import load_bank as load_purpose_bank, CONTEXT
from minecraftrl.rl_core import WoodReward, ShapedWoodRewardV3
from analyze_rm_purpose import release_check

MODES = ('purpose-fixed','logo-fixed','logo-staged')


def observed_failure(success, first):
    if success:
        return 'success'
    if 'target_known' not in first:
        return 'target_not_registered'
    if 'target_reachable' not in first:
        return 'registered_but_not_reached'
    if 'log_broken' not in first:
        return 'reached_but_no_confirmed_break'
    return 'confirmed_break_without_acquisition'


def audit(folder, spec, bank, validators):
    out = dict(spec)
    if not (folder/'summary.json').exists():
        return dict(out, classification='unexecuted_or_incomplete', success=None)
    summary = read_json(folder/'summary.json')
    out.update({key:summary.get(key) for key in ('status','steps','elapsed_control_s','inference_ms',
        'control_cycle_ms','device_vram','gpu','error','worker_exit_code','goal_switch_count','stage_counts')})
    if 'worker_exit_code' not in summary:
        return dict(out, classification='unexecuted_or_incomplete', success=None)
    if not summary.get('execution_valid'):
        return dict(out, classification='cancelled' if summary['status']=='cancelled' else 'technical_error', success=None)
    rows = read_json(folder/'rm-rows.json')
    initial = previous = read_json(folder/'initial-state.json')['server']
    task = WoodReward(initial['server_inventory'])  # Independent of generated functions.
    monitor = SuccessMonitor(validators, initial)
    selector = StageSelector()
    legacy = spec['mode'] == 'purpose-fixed'
    raw_stage = ShapedWoodRewardV3.state(initial)[0] if legacy else stage(monitor.values, initial)
    selector.observe(raw_stage)
    errors, switches, first = [], [], {name:0 for name,value in monitor.values.items() if value}
    screens, buttons, cells = Counter(), Counter(), set()
    dispatches = transitions = after_ack = 0
    acknowledged = terminal = False
    distance, min_y, max_y = 0., initial['server_y'], initial['server_y']
    last_key = mining_target = None
    mining_streak = longest_mining = inventory_streak = longest_inventory = 0
    mining_ticks = break_without_pickup = 0
    event_hash = hashlib.sha256()
    with (folder/'events.jsonl').open('rb') as handle:
        for line in handle:
            event_hash.update(line)
            event = json.loads(line)
            if event['event']=='stop_acknowledged':
                acknowledged = True
            if event['event']=='policy_dispatch_started':
                dispatches += 1
                after_ack += int(acknowledged)
                if task.completed or terminal:
                    errors.append('dispatch_after_completion_or_terminal')
            if event['event']!='transition':
                continue
            transitions += 1
            if transitions > len(rows):
                errors.append('missing_condition_row')
                break
            row = rows[transitions-1]
            key = 'gather' if legacy else ('seek' if spec['mode']=='logo-staged' and selector.current=='search' else 'root')
            if (row['step']!=transitions or event['step']!=transitions or row['stage']!=selector.current
                    or row['raw_stage']!=raw_stage or row['goal_key']!=key or row['prompt']!=bank['prompts'][key]
                    or row['embedding_sha256']!=bank['embeddings'][key]['vector_sha256']):
                errors.append('condition_mismatch_'+str(transitions))
            if row['env_action']!=event['action']:
                errors.append('action_mismatch_'+str(transitions))
            if legacy:
                if row['context_sha256']!=digest(CONTEXT):
                    errors.append('legacy_context_mismatch')
            elif row['validator_sha256']!=validators.sha256 or row['validators_before']!=monitor.values:
                errors.append('pre_action_predicate_mismatch_'+str(transitions))
            if key!=last_key:
                switches.append(dict(step=transitions,key=key))
                last_key=key
            server=event['info']['workshop']
            terminal=bool(event['done'])
            expected_reward=task.observe(server['server_inventory'])
            generated_reward=monitor.advance(server)
            if row['reward']!=expected_reward or event['reward']!=expected_reward or generated_reward!=expected_reward:
                errors.append('server_reward_mismatch_'+str(transitions))
            if row['terminal']!=bool(terminal or task.completed):
                errors.append('terminal_mismatch_'+str(transitions))
            if not legacy and (row['validators_after']!=monitor.values or row['validator_tick_after']!=server['server_tick']):
                errors.append('post_action_predicate_mismatch_'+str(transitions))
            if server['server_tick']-previous['server_tick']!=1:
                errors.append('tick_mismatch_'+str(transitions))
            for name,value in monitor.values.items():
                if value and name not in first:
                    first[name]=transitions
            break_without_pickup+=int(monitor.values['log_broken'] and not task.completed)
            screens[str(int(server['screen']))]+=1
            for button in ('inventory','attack','drop','forward','jump'):
                buttons[button]+=int(row['env_action'][button][0]!=0)
            mining_ticks+=int(server.get('wood_server_mining')==1)
            target=tuple(server.get('wood_mining_'+a) for a in 'xyz') if server.get('wood_server_mining')==1 else None
            mining_streak=mining_streak+1 if target is not None and target==mining_target else int(target is not None)
            mining_target=target
            longest_mining=max(longest_mining,mining_streak)
            inventory_streak=inventory_streak+1 if server['screen']==1 else 0
            longest_inventory=max(longest_inventory,inventory_streak)
            distance+=math.hypot(server['server_x']-previous['server_x'],server['server_z']-previous['server_z'])
            cells.add((math.floor(server['server_x']),math.floor(server['server_z'])))
            min_y,max_y=min(min_y,server['server_y']),max(max_y,server['server_y'])
            raw_stage=('complete' if task.completed else ShapedWoodRewardV3.state(server)[0]) if legacy else stage(monitor.values,server)
            selector.observe(raw_stage)
            previous=server
    receipt=read_json(folder/'input-release-state.json') if (folder/'input-release-state.json').exists() else {}
    release=release_check(summary,receipt,terminal)
    cleanup=summary.get('process_cleanup',{})
    if transitions!=len(rows) or dispatches!=transitions or transitions!=summary['steps']:
        errors.append('dispatch_count_mismatch')
    if not acknowledged or after_ack:
        errors.append('stop_contract')
    if not release['valid'] or cleanup.get('remaining_live_pids')!=[] or summary['worker_exit_code']!=0:
        errors.append('release_or_process_cleanup')
    if (summary.get('rm_conditioning')!=spec['mode'] or summary.get('optimizer_updates')!=0
            or summary.get('server_target_reached')!=task.completed):
        errors.append('summary_mismatch')
    if summary.get('goal_switch_count')!=max(0,len(switches)-1):
        errors.append('switch_count_mismatch')
    if spec['mode']!='logo-staged' and len(switches)>1:
        errors.append('fixed_condition_switched')
    return dict(out,classification='audit_failed' if errors else 'valid',audit_failures=errors,
        success=task.completed if not errors else None,log_gain=WoodReward.count(previous['server_inventory'])-WoodReward.count(initial['server_inventory']),
        simulation_ticks=transitions,nominal_simulation_s=transitions/20,actual_conditions=switches,
        first_true_steps=first,observed_outcome_stage=observed_failure(task.completed,first),
        no_dispatch_after_stop=acknowledged and not after_ack,release_verification=release,remaining_live_pids=cleanup.get('remaining_live_pids'),
        diagnostics=dict(screen_ticks=dict(screens),button_active_ticks=dict(buttons),accepted_mining_ticks=mining_ticks,
            longest_same_target_mining_streak=longest_mining,longest_inventory_screen_streak=longest_inventory,
            confirmed_break_without_acquisition_ticks=break_without_pickup),
        initial_frame_sha256=hashlib.sha256((folder/'frames/000000.png').read_bytes()).hexdigest(),
        initial_pose={key:initial.get(key) for key in ('server_x','server_y','server_z','yaw','pitch')},
        motion=dict(unique_xz_cells=len(cells),path_xz_blocks=distance,min_y=min_y,max_y=max_y),
        final_health={key:previous.get(key) for key in ('server_alive','server_health','server_recent_damage_type')},
        final_target={key:previous.get(key) for key in ('wood_v3_kind','wood_v3_distance','wood_v3_reachable')},
        events_sha256=event_hash.hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--output',default='audit.json')
    args=parser.parse_args()
    output=args.batch/args.output
    if output.exists():
        raise FileExistsError(output)
    plan=read_json(args.batch/'pilot-plan.json')
    _,logo_bank=load_bank(args.batch/'goal-bank')
    _,purpose_bank=load_purpose_bank(args.batch/'purpose-bank')
    validators,_=admit(logo_bank['proposal'])
    rows=[audit(args.batch/item['name'],item,purpose_bank if item['mode']=='purpose-fixed' else logo_bank,validators) for item in plan['runs']]
    conditions={}
    for mode in MODES:
        selected=[row for row in rows if row['mode']==mode]
        valid=[row for row in selected if row['classification']=='valid']
        conditions[mode]=dict(planned=len(selected),valid=len(valid),success=sum(row['success'] for row in valid),
            policy_failure=sum(not row['success'] for row in valid),excluded_or_unexecuted=len(selected)-len(valid))
    matched=[]
    for seed in plan['seeds']:
        group=[row for row in rows if row['seed']==seed and row['classification']=='valid']
        matched.append(dict(seed=seed,all_valid=len(group)==3,
            same_initial_pose=len({digest(row['initial_pose']) for row in group})==1 if len(group)==3 else None,
            same_initial_frame=len({row['initial_frame_sha256'] for row in group})==1 if len(group)==3 else None))
    result=dict(schema=1,protocol='P3',conditions=conditions,runs=rows,matched=matched,optimizer_updates=0,
        interpretation='Small development experiment; privileged server RM; no learning/generalization claim.')
    with output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(conditions,indent=2))
    if any(row['classification']=='audit_failed' for row in rows):
        raise RuntimeError('audit mismatch; inspect evidence before reporting performance')


if __name__=='__main__':
    main()
