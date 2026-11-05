"""Streaming audit of P2 actual dispatches, sparse server reward and condition selection."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from minecraftrl.goal_contract import read_json, digest
from minecraftrl.rm_conditioning import StageSelector
from minecraftrl.rm_purpose_conditioning import MODES, CONTEXT, next_phase, condition_key, load_bank
from minecraftrl.rl_core import WoodReward, ShapedWoodRewardV3


def release_check(summary, receipt, terminal_observed):
    """Terminal MineRL rejects neutral steps; retain that measurement limit."""
    if summary.get('release') == 'environment_done_no_neutral_step':
        valid = (summary.get('status') == 'environment_done' and terminal_observed is True
                 and summary.get('worker_exit_code') == 0
                 and summary.get('process_cleanup', {}).get('remaining_live_pids') == [])
        return dict(valid=valid, kind='terminal_game_closed', neutral_verified=False, held_key_count=None)
    valid = (summary.get('release') == 'neutral_command_sent' and receipt.get('verified') is True
             and receipt.get('held_key_count') == 0)
    return dict(valid=valid, kind='neutral_step', neutral_verified=valid,
                held_key_count=receipt.get('held_key_count'))


def audit(folder, spec, bank):
    out = dict(spec)
    if not (folder/'summary.json').exists():
        return dict(out, classification='unexecuted_or_incomplete', success=None)
    summary = read_json(folder/'summary.json')
    out.update({key:summary.get(key) for key in ('status','steps','elapsed_control_s','inference_ms',
        'control_cycle_ms','device_vram','gpu','error','worker_exit_code','goal_switch_count','stage_counts')})
    if 'worker_exit_code' not in summary:
        return dict(out, classification='unexecuted_or_incomplete', success=None)
    if not summary.get('execution_valid'):
        return dict(out, classification='cancelled' if summary['status']=='cancelled' else 'technical_error',
                    success=None, cleanup=summary.get('process_cleanup'))
    errors = []
    rows = read_json(folder/'rm-rows.json')
    previous = initial = read_json(folder/'initial-state.json')['server']
    task = WoodReward(initial['server_inventory'])
    selector = StageSelector()
    raw_stage = ShapedWoodRewardV3.state(initial)[0]
    phase = next_phase('seeking', selector.observe(raw_stage))
    dispatches = transitions = after_ack = 0
    acknowledged = False
    terminal_observed = False
    cells, distance = set(), 0.
    first_target_step = 0 if phase=='gathering' else None
    min_y = max_y = initial['server_y']
    last_key, switches = None, []
    screens, buttons = Counter(), Counter()
    mining_ticks = collect_sensor_ticks = 0
    mining_target = None
    mining_streak = longest_mining_streak = 0
    inventory_streak = longest_inventory_streak = inventory_entries = 0
    for line in (folder/'events.jsonl').open(encoding='utf-8'):
        event = json.loads(line)
        if event['event']=='stop_acknowledged':
            acknowledged = True
        if event['event']=='policy_dispatch_started':
            dispatches += 1
            after_ack += int(acknowledged)
            if task.completed or terminal_observed:
                errors.append('dispatch_after_completion_or_terminal')
        if event['event']!='transition':
            continue
        transitions += 1
        if transitions > len(rows):
            errors.append('missing_condition_row')
            break
        row = rows[transitions-1]
        key = condition_key(spec['mode'], phase)
        if (row['step']!=transitions or event['step']!=transitions
                or row['stage']!=selector.current or row['raw_stage']!=raw_stage
                or row['execution_phase']!=phase or row['goal_key']!=key
                or row['prompt']!=bank['prompts'][key] or row['context_sha256']!=digest(CONTEXT)
                or row['embedding_sha256']!=bank['embeddings'][key]['vector_sha256']):
            errors.append('condition_mismatch_'+str(transitions))
        if row['env_action']!=event['action']:
            errors.append('action_mismatch_'+str(transitions))
        if key!=last_key:
            switches.append(dict(step=transitions, key=key))
            last_key=key
        server = event['info']['workshop']
        terminal_observed = bool(event['done'])
        screens[str(server.get('screen'))] += 1
        for button in ('inventory','attack','drop','forward','jump'):
            buttons[button] += int(row['env_action'][button][0] != 0)
        mining_ticks += int(server.get('wood_server_mining') == 1)
        collect_sensor_ticks += int(server.get('wood_v3_kind') == 2)
        target = tuple(server.get('wood_mining_'+axis) for axis in 'xyz') if server.get('wood_server_mining')==1 else None
        mining_streak = mining_streak+1 if target is not None and target==mining_target else int(target is not None)
        mining_target = target
        longest_mining_streak = max(longest_mining_streak, mining_streak)
        if server.get('screen')==1:
            inventory_entries += int(inventory_streak==0)
            inventory_streak += 1
        else:
            inventory_streak = 0
        longest_inventory_streak = max(longest_inventory_streak, inventory_streak)
        reward = task.observe(server['server_inventory'])
        if row['reward']!=reward or event['reward']!=reward:
            errors.append('server_reward_mismatch_'+str(transitions))
        if server['server_tick']-previous['server_tick']!=1:
            errors.append('server_tick_mismatch_'+str(transitions))
        distance += math.hypot(server['server_x']-previous['server_x'],server['server_z']-previous['server_z'])
        cells.add((math.floor(server['server_x']),math.floor(server['server_z'])))
        min_y, max_y = min(min_y,server['server_y']), max(max_y,server['server_y'])
        raw_stage = 'complete' if task.completed else ShapedWoodRewardV3.state(server)[0]
        phase = next_phase(phase, selector.observe(raw_stage))
        if phase=='gathering' and first_target_step is None:
            first_target_step=transitions
        previous=server
    receipt_path = folder/'input-release-state.json'
    release = read_json(receipt_path) if receipt_path.exists() else {}
    release_verification = release_check(summary, release, terminal_observed)
    cleanup = summary.get('process_cleanup', {})
    if transitions!=len(rows) or dispatches!=transitions or transitions!=summary['steps']:
        errors.append('dispatch_row_count_mismatch')
    if not acknowledged or after_ack:
        errors.append('stop_contract')
    if not release_verification['valid']:
        errors.append('input_release')
    if cleanup.get('remaining_live_pids')!=[] or summary['worker_exit_code']!=0:
        errors.append('process_cleanup')
    if (summary.get('rm_conditioning')!=spec['mode'] or summary.get('optimizer_updates')!=0
            or summary.get('server_target_reached')!=task.completed):
        errors.append('summary_mismatch')
    if len(switches)> (2 if spec['mode']=='purpose-switch' else 1):
        errors.append('excess_condition_switch')
    return dict(out, classification='audit_failed' if errors else 'valid', audit_failures=errors,
        success=task.completed if not errors else None,
        initial_log_count=WoodReward.count(initial['server_inventory']),
        final_log_count=WoodReward.count(previous['server_inventory']),
        log_gain=WoodReward.count(previous['server_inventory'])-WoodReward.count(initial['server_inventory']),
        simulation_ticks=transitions, nominal_simulation_s=transitions/20,
        first_confirmed_target_step=first_target_step, actual_conditions=switches,
        no_dispatch_after_stop=acknowledged and not after_ack,
        release_verification=release_verification, held_key_count=release_verification['held_key_count'],
        remaining_live_pids=cleanup.get('remaining_live_pids'),
        exploratory_behavior=dict(screen_ticks=dict(screens),button_active_ticks=dict(buttons),
            accepted_mining_ticks=mining_ticks,collect_sensor_ticks=collect_sensor_ticks,
            longest_same_target_mining_streak=longest_mining_streak,
            inventory_entries=inventory_entries,longest_inventory_screen_streak=longest_inventory_streak,
            scope='Post-hoc diagnostics after observing first-seed failure; not preregistered causal endpoints'),
        initial_frame_sha256=hashlib.sha256((folder/'frames/000000.png').read_bytes()).hexdigest(),
        initial_pose={key:initial.get(key) for key in ('server_x','server_y','server_z','yaw','pitch')},
        motion=dict(unique_xz_cells=len(cells),path_xz_blocks=distance,min_y=min_y,max_y=max_y,
                    start_xyz=[initial['server_'+a] for a in 'xyz'],final_xyz=[previous['server_'+a] for a in 'xyz']),
        final_health={key:value for key,value in previous.items() if any(term in key for term in ('health','dead','alive'))},
        final_target={key:previous.get(key) for key in ('wood_v3_kind','wood_v3_distance','wood_v3_reachable')})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',type=Path,required=True)
    args=parser.parse_args()
    plan=read_json(args.batch/'pilot-plan.json')
    _,bank=load_bank(args.batch/'goal-bank')
    if hashlib.sha256((args.batch/'goal-bank/manifest.json').read_bytes()).hexdigest().upper()!=plan['bank_sha256']:
        raise ValueError('goal bank changed after plan lock')
    rows=[audit(args.batch/item['name'],item,bank) for item in plan['runs']]
    conditions={}
    for mode in MODES:
        selected=[r for r in rows if r['mode']==mode]
        valid=[r for r in selected if r['classification']=='valid']
        conditions[mode]=dict(planned=len(selected),valid=len(valid),success=sum(r['success'] for r in valid),
                              policy_failure=sum(not r['success'] for r in valid),excluded=len(selected)-len(valid))
    matched=[]
    for seed in plan['seeds']:
        group=[r for r in rows if r['seed']==seed and r['classification']=='valid']
        matched.append(dict(seed=seed,all_valid=len(group)==3,
            same_initial_pose=len({digest(r['initial_pose']) for r in group})==1 if len(group)==3 else None,
            same_initial_frame=len({r['initial_frame_sha256'] for r in group})==1 if len(group)==3 else None))
    result=dict(protocol='P2',optimizer_updates=0,conditions=conditions,runs=rows,matched=matched,
        interpretation='Development-only frozen policy; server-informed RM; no learning or generalization claim.')
    (args.batch/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(conditions,indent=2))


if __name__=='__main__':
    main()
