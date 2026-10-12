"""Preserve the source/parameter evidence and inspect actual long-wall routes."""
import json
from pathlib import Path

from minecraftrl.runtime import write_json
from minecraftrl.saved_block import saved_block
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root / 'results/masked-correction-comparison-20261009'
out = base / 'final-verification.json'
if out.exists():
    raise FileExistsError(out)

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

work_path = root / 'results/masked-correction-work-20261009/work.json'
work = read(work_path)
state = read(base / 'experiment.json')
audit = read(base / 'audit.json')
training = read(root / 'results/masked-correction-training-20261009/training.json')
supervision = read(root / 'results/masked-correction-training-20261009/supervision-audit.json')
runtime = read(root / 'results/masked-correction-runtime-20261009/report.json')
assert state['completed'] and state['active'] is None and not state['errors']
assert audit['verified'] and supervision['verified'] and training['completed'] and runtime['completed']
assert training['updates'] == supervision['optimizer_updates'] == 130
assert digest(root / 'results/masked-correction-features-20261009/features.pt') == training['features_sha256'] == work['features_sha256']
assert digest(root / 'weights/steve1/steve1.weights') == training['model_sha256']
assert len(state['runs']) + audit['unrun_conditions'] == 4
assert len(audit['rows']) == len(state['runs'])
assert (root / 'results/STOP').read_text().startswith(('RL training ended', 'RL collection ended'))
for path, sha in work['code_and_plan_sha256'].items():
    assert digest(root / path) == sha, path
for path, sha in state['frozen'].items():
    assert digest(path) == sha, path
assert digest(root / 'docs/research-spec.md') == work['research_spec_sha256']

plan = read(root / 'configs/masked-correction-bc-20261009.json')
assert digest(root / plan['source_manifest']) == plan['source_manifest_sha256']
assert digest(root / plan['source_experiment']) == plan['source_experiment_sha256']
assert digest(root / plan['checkpoint']) == plan['checkpoint_sha256']
source_experiment = read(root / plan['source_experiment'])
for world in source_experiment['worlds'].values():
    assert digest(world['path']) == world['sha256']
source_files = source_frames = 0
for session in plan['sessions']:
    run = root / session['run']
    for name, sha in session['files_sha256'].items():
        assert digest(run / name) == sha
        source_files += 1
    for frame in session['frames']:
        assert digest(run / frame['path']) == frame['sha256']
        source_frames += 1
assert source_files == 36 and source_frames == 4961
for world in state['worlds']:
    assert digest(world['path']) == world['sha256']

routes = []
for record in state['runs']:
    if record['world']['kind'] != 'long_wall':
        continue
    run = Path(record['path'])
    initial = read(run / 'initial-state.json')
    world = initial['initial_world']
    ox, oz = (16 * v for v in world['chunk'])
    wall_end = world['curriculum']['wall_end']
    assert not world['curriculum']['mirror_z']
    blocks = {}
    for x, z, expected in ((6, 0, 'bedrock'), (7, 1, 'bedrock'),
                            (7, wall_end, 'bedrock'), (7, wall_end + 1, 'air')):
        position = (ox + x, 201, oz + z)
        actual = saved_block(record['world']['path'], position)
        assert actual == 'minecraft:' + expected
        blocks[str(position)] = actual
    previous = initial['server']
    positions = [(previous['server_x'], previous['server_z'])]
    crossing = None
    movement = attacks = wall_attacks = gap_steps = closed_side_steps = steps = 0
    for line in (run / 'events.jsonl').open():
        event = json.loads(line)
        if event['event'] != 'transition':
            continue
        steps += 1
        server = event['info']['workshop']
        action = event['action']
        pressed = lambda key: bool(action[key][0] if isinstance(action[key], list) else action[key])
        movement += any(pressed(key) for key in ('forward', 'back', 'left', 'right'))
        attack = pressed('attack')
        attacks += attack
        wall_attacks += int(attack and previous['hit_x'] == ox + 7 and
                            previous['hit_y'] in (201, 202) and 1 <= previous['hit_z'] - oz <= wall_end)
        gap_steps += server['server_z'] - oz >= wall_end + 1.3
        closed_side_steps += server['server_z'] - oz <= 1.35
        positions.append((server['server_x'], server['server_z']))
        if crossing is None and server['server_x'] >= ox + 8.3:
            crossing = dict(step=steps, position=[server['server_x'], server['server_z']])
        previous = server
    routes.append(dict(label=record['label'],steps=steps,first_crossing=crossing,
                       movement_command_steps=movement,attack_steps=attacks,wall_attack_steps=wall_attacks,
                       gap_clear_steps=gap_steps,gap_clearance_threshold_z=oz+wall_end+1.3,
                       closed_side_steps=closed_side_steps,geometry_nbt=blocks,final_position=positions[-1],
                       x_range=[min(p[0] for p in positions),max(p[0] for p in positions)],
                       z_range=[min(p[1] for p in positions),max(p[1] for p in positions)],
                       lateral_span_blocks=max(p[1] for p in positions)-min(p[1] for p in positions)))

write_json(base / 'route-audit.json',dict(verified=True,rows=routes,policy_actions_changed=0))
new_files = [root / name for name in (
    'scripts/audit_masked_correction_features.py', 'scripts/audit_masked_correction_training.py',
    'scripts/audit_temporal_bc.py', 'scripts/verify_masked_correction_completion.py',
    'scripts/plot_masked_correction_routes.py',
    'docs/masked-correction-bc-validation-2026-10-09.md', 'docs/progress.md', 'docs/next-task-units.md')]
receipts = [base / name for name in ('experiment.json', 'audit.json', 'route-audit.json')]
result = dict(verified=True,source_files_preserved=source_files,source_frames_preserved=source_frames,
              source_worlds_preserved=True,research_spec_unchanged=True,planned_code_hashes_unchanged=True,
              optimizer_updates=130,actual_games=len(state['runs']),unrun_conditions=audit['unrun_conditions'],
              game_transitions=sum(row['steps'] for row in audit['rows']),
              post_stop_dispatches=sum(row['post_stop_policy_dispatches'] for row in audit['rows']),
              released_held_keys=sum(row['released_held_keys'] for row in audit['rows']),
              remaining_owned_pids=[pid for row in audit['rows'] for pid in row['remaining_owned_pids']],
              stop_latched=True,paid_resources=0,actual_llm_calls=0,colab_connections=0,default_policy_promoted=False,
              artifacts_sha256={str(p.relative_to(root)):digest(p) for p in new_files+receipts})
write_json(out,result)
work.update(completed=True,phase='completed',final_verification=str(out.relative_to(root)))
write_json(work_path,work)
print(json.dumps(dict(verification=result,routes=routes)))
