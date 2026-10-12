"""Actual policy weights, server inventory/route, input release and STOP evidence."""
import json
import math
from pathlib import Path

import torch

from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root/'results/coverage-bc-comparison-20261009'
out = base/'audit.json'
if out.exists():
    raise FileExistsError(out)
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
state = read(base/'experiment.json')
assert state['completed'] and not state['errors'] and state['active'] is None


def local(path):
    return root/str(path).replace('\\', '/').split('/MinecraftRL/', 1)[-1]


audits = []
for trial in state['runs']:
    run, checkpoint = local(trial['path']), local(trial['checkpoint'])
    report = read(run/'summary.json')
    assert trial['audited'] and digest(checkpoint) == trial['checkpoint_sha256']
    assert report['execution_valid'] and report['worker_exit_code'] == 0 and not report['rl_training_eligible']
    assert not report['process_cleanup']['remaining_live_pids']
    release = read(run/'input-release-state.json')
    assert release['verified'] and release['held_key_count'] == 0
    shaping = read(run/'shaping-audit.json')
    recovery_audit = read(run/'recovery-audit.json')
    assert shaping['verified'] and shaping['no_policy_action_after_stop'] and recovery_audit['verified']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    data = torch.load(run/'trajectory.pt', map_location='cpu', weights_only=False)
    assert data['split'] == 'eval' and len(data['rows']) == report['steps']
    for group in ('actor', 'value', 'backbone_tail', 'backbone_temporal'):
        assert state_fingerprint(saved[group]) == state_fingerprint(data['initial_heads'][group]), group
    del saved, data
    initial = read(run/'initial-state.json')
    previous = start = initial['server']
    wall = trial['world']['kind'] == 'long_wall'
    if wall:
        world = initial['initial_world']
        ox, oz = (16*v for v in world['chunk'])
        assert not world['curriculum']['mirror_z']
        wall_end = world['curriculum']['wall_end']
    distance = 0.
    steps = move_commands = attacks = closed_pre = gap_post = 0
    stopped = accepted = False
    first_crossing = first_entry = first_turn = first_displacement = None
    recovery_pose = None
    positions = [[start['server_x'], start['server_z']]]
    for line in (run/'events.jsonl').open():
        event = json.loads(line)
        if event['event'] in ('inference_started', 'policy_dispatch_started'):
            assert not stopped and not accepted
        if event['event'] == 'stop_acknowledged': stopped = True
        if event['event'] == 'target_reached': accepted = True
        if event['event'] != 'transition': continue
        steps += 1
        server = event['info']['workshop']
        pressed = lambda key: bool(event['action'][key][0])
        move_commands += any(pressed(k) for k in ('forward', 'back', 'left', 'right'))
        attacks += pressed('attack')
        distance += math.hypot(server['server_x']-previous['server_x'], server['server_z']-previous['server_z'])
        if wall:
            closed_pre += previous['server_z']-oz <= 1.35
            gap_post += server['server_z']-oz >= wall_end+1.3
            if first_crossing is None and server['server_x'] >= ox+8.3:
                first_crossing = steps
        recovery = event['info']['rl_reward']['recovery']
        if first_entry is None and recovery['event'] == 'entered':
            first_entry, recovery_pose = steps, server
        if recovery_pose is not None:
            angle = abs((server['yaw']-recovery_pose['yaw']+180)%360-180)
            displacement = math.hypot(server['server_x']-recovery_pose['server_x'], server['server_z']-recovery_pose['server_z'])
            if first_turn is None and angle >= 20: first_turn = steps
            if first_displacement is None and displacement >= .5: first_displacement = steps
        positions.append([server['server_x'], server['server_z']])
        previous = server
    assert stopped and steps == report['steps']
    delta = {k: previous['server_inventory'].get(k, 0)-start['server_inventory'].get(k, 0)
             for k in set(previous['server_inventory']) | set(start['server_inventory'])}
    delta = {k: v for k, v in delta.items() if v}
    success = any(delta.get(k+'_log', 0) > 0 for k in ('oak', 'birch', 'spruce', 'jungle', 'acacia', 'dark_oak'))
    assert success == report['server_target_reached']
    audits.append(dict(label=trial['label'], model=trial['model'], steps=steps, server_success=success,
                       inventory_delta=delta, horizontal_path_blocks=distance, movement_command_steps=move_commands,
                       attack_steps=attacks, first_crossing=first_crossing,
                       closed_corner_pre_action_steps=closed_pre if wall else None,
                       gap_post_action_steps=gap_post if wall else None,
                       first_recovery_entry=first_entry, first_20deg_turn_after_entry=first_turn,
                       first_half_block_displacement_after_entry=first_displacement,
                       x_range=[min(p[0] for p in positions), max(p[0] for p in positions)],
                       z_range=[min(p[1] for p in positions), max(p[1] for p in positions)],
                       elapsed_control_s=report['elapsed_control_s'], inference_ms=report['inference_ms'],
                       control_cycle_ms=report['control_cycle_ms'], gpu=report['gpu'], device_vram=report['device_vram'],
                       loaded_weights_verified=True, post_stop_policy_dispatches=0, released_held_keys=0,
                       remaining_owned_pids=[], source_sha256={name:digest(run/name) for name in
                           ('events.jsonl', 'summary.json', 'trajectory.pt', 'initial-state.json')}))
for path, sha in state['frozen'].items():
    assert digest(local(path)) == sha
for world in state['worlds']:
    assert digest(local(world['path'])) == world['sha256']
assert len(audits)+len(state['unrun']) == state['planned_conditions'] == 4
write_json(out, dict(verified=True, actual_games=len(audits), rows=audits, unrun_conditions=state['unrun'],
                     comparison_valid=state['comparison_valid'], gate_policy=state['gate_policy'],
                     strict_gate_passed=state['strict_gate_passed'],
                     scope=('Actual autonomous development games; no teacher assistance, not final evaluation'
                            if audits else 'Admission bookkeeping only; no actual games executed.')))
print(json.dumps(audits))
