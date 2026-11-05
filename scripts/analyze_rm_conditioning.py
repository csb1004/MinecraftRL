"""Audit actual P1 dispatches and inventory; missing/invalid runs remain separate."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from minecraftrl.rm_conditioning import StageSelector
from minecraftrl.rl_core import WoodReward, ShapedWoodRewardV3


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def audit(folder, spec, bank):
    report = dict(seed=spec['seed'], mode=spec['mode'], name=spec['name'])
    if not (folder/'summary.json').exists():
        return dict(report, classification='unexecuted_or_incomplete', success=None)
    summary = read(folder/'summary.json')
    report.update(status=summary['status'], steps=summary.get('steps', 0),
                  control_s=summary.get('elapsed_control_s'), inference_ms=summary.get('inference_ms'),
                  cycle_ms=summary.get('control_cycle_ms'), device_vram=summary.get('device_vram'),
                  error=summary.get('error'), problems=summary.get('problems'))
    if summary.get('execution_valid') and 'worker_exit_code' not in summary:
        return dict(report, classification='unexecuted_or_incomplete', success=None)
    if not summary.get('execution_valid'):
        return dict(report, classification='cancelled' if summary['status']=='cancelled' else 'technical_error',
                    success=None, cleanup=summary.get('process_cleanup'))
    failures = []
    events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    transitions = [event for event in events if event['event']=='transition']
    rows = read(folder/'rm-rows.json')
    if len(rows) != len(transitions) or len(rows) != summary['steps']:
        failures.append('row_dispatch_count_mismatch')
    if sum(event['event']=='policy_dispatch_started' for event in events)!=len(rows):
        failures.append('dispatch_transition_count_mismatch')
    if summary.get('rm_conditioning')!=spec['mode'] or summary.get('optimizer_updates')!=0:
        failures.append('condition_or_training_mismatch')
    stop_indices = [i for i,event in enumerate(events) if event['event']=='stop_acknowledged']
    if not stop_indices or any(event['event']=='policy_dispatch_started' for event in events[stop_indices[0]+1:]):
        failures.append('dispatch_after_stop_or_missing_ack')
    release = read(folder/'input-release-state.json') if (folder/'input-release-state.json').exists() else {}
    if summary.get('release') != 'neutral_command_sent' or not release.get('verified'):
        failures.append('input_release_unverified')
    cleanup = summary.get('process_cleanup', {})
    if cleanup.get('remaining_live_pids') != []:
        failures.append('child_cleanup_unverified')
    selector = StageSelector()
    initial = read(folder/'initial-state.json')['server']
    task = WoodReward(initial['server_inventory'])
    selector.observe(ShapedWoodRewardV3.state(initial)[0])
    raw_stage = selector.current
    for expected_step,(row, event) in enumerate(zip(rows, transitions),1):
        if row['step']!=expected_step or event['step']!=expected_step:
            failures.append('step_sequence_mismatch')
        key = 'root' if spec['mode']=='fixed' else selector.current
        if (row['stage'] != selector.current or row['raw_stage'] != raw_stage or
                row['goal_key'] != key or row['embedding_sha256'] != bank['embeddings'][key]['vector_sha256']):
            failures.append('rm_condition_mismatch_at_'+str(row['step']))
        if row['env_action'] != event['action']:
            failures.append('action_mismatch_at_'+str(row['step']))
        server = event['info']['workshop']
        expected = task.observe(server['server_inventory'])
        if row['reward'] != expected or event['reward'] != expected:
            failures.append('reward_mismatch_at_'+str(row['step']))
        raw_stage = 'complete' if task.completed else ShapedWoodRewardV3.state(server)[0]
        selector.observe(raw_stage)
    final_count = WoodReward.count(transitions[-1]['info']['workshop']['server_inventory']) if transitions else 0
    if task.completed != summary.get('server_target_reached'):
        failures.append('server_goal_mismatch')
    image = folder/'frames/000000.png'
    states=[initial]+[event['info']['workshop'] for event in transitions]
    if any(b['server_tick']-a['server_tick']!=1 for a,b in zip(states,states[1:])):
        failures.append('server_tick_contract_mismatch')
    xyz=[tuple(state['server_'+axis] for axis in ('x','y','z')) for state in states]
    longest={}
    prior_stage=None
    length=0
    for row in rows:
        length=length+1 if row['stage']==prior_stage else 1
        longest[row['stage']]=max(longest.get(row['stage'],0),length)
        prior_stage=row['stage']
    final_server=states[-1]
    report.update(classification='audit_failed' if failures else 'valid',
                  success=task.completed if not failures else None, final_log_count=final_count,
                  stage_counts=summary.get('stage_counts'), goal_switches=summary.get('goal_switch_count'),
                  simulation_ticks=summary.get('simulation_ticks'),
                  nominal_simulation_s=summary.get('simulation_ticks',0)/20.,
                  audit_failures=failures, no_dispatch_after_stop=bool(stop_indices) and not any(
                      event['event']=='policy_dispatch_started' for event in events[stop_indices[0]+1:]),
                  held_key_count=release.get('held_key_count'), remaining_live_pids=cleanup.get('remaining_live_pids'),
                  initial_frame_sha256=hashlib.sha256(image.read_bytes()).hexdigest() if image.exists() else None,
                  initial_pose={k:initial.get(k) for k in ('server_x','server_y','server_z','yaw','pitch')},
                  longest_stage_ticks=longest,
                  motion=dict(start_xyz=xyz[0],final_xyz=xyz[-1],min_y=min(p[1] for p in xyz),max_y=max(p[1] for p in xyz),
                              unique_xz_cells=len({(math.floor(p[0]),math.floor(p[2])) for p in xyz}),
                              path_xz_blocks=sum(math.hypot(b[0]-a[0],b[2]-a[2]) for a,b in zip(xyz,xyz[1:]))),
                  final_target={k:final_server.get(k) for k in ('wood_v3_kind','wood_v3_distance','wood_v3_reachable',
                                                              'wood_v3_target_x','wood_v3_target_y','wood_v3_target_z')})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--plan', default='pilot-plan.json')
    args = parser.parse_args()
    plan = read(args.batch/args.plan)
    bank = read(args.batch/'goal-bank/manifest.json')
    rows = [audit(args.batch/item['name'], item, bank) for item in plan['runs']]
    conditions = {}
    for mode in ('fixed','staged'):
        selected = [row for row in rows if row['mode']==mode]
        valid = [row for row in selected if row['classification']=='valid']
        conditions[mode] = dict(planned=len(selected), valid=len(valid), success=sum(row['success'] for row in valid),
                                policy_failure=sum(not row['success'] for row in valid),
                                excluded=len(selected)-len(valid))
    pairs = []
    for seed in plan['seeds']:
        selected = {row['mode']:row for row in rows if row['seed']==seed}
        fixed, staged = selected['fixed'], selected['staged']
        valid = fixed['classification']==staged['classification']=='valid'
        pairs.append(dict(seed=seed, both_valid=valid,
                          same_initial_frame=(fixed['initial_frame_sha256']==staged['initial_frame_sha256']) if valid else None,
                          same_initial_pose=(fixed['initial_pose']==staged['initial_pose']) if valid else None))
    prior_attempts = [audit(args.batch/item['prior_attempt'],dict(item,name=item['prior_attempt']),bank)
                      for item in plan['runs'] if item.get('prior_attempt')]
    report = dict(schema=1, protocol=plan.get('protocol','P1'), optimizer_updates=0, conditions=conditions, runs=rows,
                  excluded_prior_attempts=prior_attempts,
                  pairs=pairs,
                  interpretation='Development only; server-informed RM; no RL training or generalization claim.')
    filename=args.plan.replace('pilot','audit',1).replace('-plan','')
    (args.batch/filename).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(conditions, indent=2))


if __name__ == '__main__':
    main()
