"""One-use audit of actual supervised updates and preserved source weights."""
import json
from pathlib import Path

import torch

from minecraftrl.correction_bc import supervised_windows
from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.temporal_bc import episode_rows
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root / 'results/masked-correction-training-20261009'
out = base / 'supervision-audit.json'
if out.exists():
    raise FileExistsError(out)

report = json.loads((base / 'training.json').read_text())
features = root / 'results/masked-correction-features-20261009/features.pt'
source = json.loads((features.parent / 'source-replay-audit.json').read_text())
assert report['completed'] and report['error'] is None
assert source['verified'] and digest(features) == report['features_sha256'] == source['features_sha256']
data = torch.load(features, map_location='cpu', weights_only=False)
groups = episode_rows(data)
sessions = {s['id']: s for s in data['plan']['sessions']}
expected = {(name, start, end) for name, rows in groups.items()
            if sessions[name]['split'] == 'train'
            for start, end in supervised_windows(rows)}
actual = set()
covered = set()
context_frames = 0
updates = [json.loads(line) for line in (base / 'updates.jsonl').read_text().splitlines()]
assert len(updates) == report['updates'] == report['planned_updates'] == 130
for number, update in enumerate(updates, 1):
    key = (update['session'], update['start'], update['end'])
    assert update['update'] == number and key in expected and key not in actual
    actual.add(key)
    rows = groups[key[0]][key[1]:key[2]]
    targets = [r for r in rows if r['loss_eligible']]
    assert len(targets) == update['supervised_frames'] > 0
    assert len(rows) - len(targets) == update['context_frames']
    context_frames += update['context_frames']
    for row in targets:
        target = (row['session'], row['frame'])
        assert target not in covered and row['owner'] == 'teacher'
        covered.add(target)
assert actual == expected and len(covered) == 1005
assert report['label_counts'] == dict(train=1005, development=619)

parent_path = root / data['plan']['parent']
assert digest(parent_path) == report['parent_sha256']
assert digest(base / 'checkpoint.pt') == report['checkpoint_sha256']
assert digest(base / 'bc-optimizer.pt') == report['optimizer_sha256']
parent = torch.load(parent_path, map_location='cpu', weights_only=False)
candidate = torch.load(base / 'checkpoint.pt', map_location='cpu', weights_only=False)
assert candidate['scope'] == parent['scope'] == 'head_tail_last_recurrent'
assert candidate['bc']['method'] == 'masked_truncated_sequence_bc'
assert candidate['bc']['value_frozen'] and not candidate['bc']['ppo_optimizer_resumable']
assert candidate['condition'] == parent['condition']
assert state_fingerprint(candidate['value']) == state_fingerprint(parent['value'])
for group in ('actor', 'backbone_tail', 'backbone_temporal'):
    assert state_fingerprint(candidate[group]) != state_fingerprint(parent[group])
assert report['initial_replay_max_error'] == report['reload_max_error'] == report['future_to_past_max_error'] == 0
assert report['earlier_frame_gradient_l1'] > 0 and report['temporal_gradient_l1'] > 0
assert report['gradient_probe'] == dict(start=460, end=468, context_frames=7, target_frames=1, burnin_frames=460)

result = dict(verified=True, optimizer_updates=len(updates), unique_teacher_targets=len(covered),
              context_in_training_windows=context_frames, context_action_targets=0,
              development_optimizer_targets=0, context_only_optimizer_updates=0,
              value_weights_unchanged=True, condition_unchanged=True,
              actual_changed_groups=['actor', 'backbone_tail', 'backbone_temporal'],
              source_student_inputs_verified=source['original_student_inputs'],
              source_feature_error=max(r['max_feature_error'] for r in source['source_replays']),
              sources_sha256={str(p.relative_to(root)): digest(p) for p in
                             (base / 'training.json', base / 'updates.jsonl', features, parent_path,
                              base / 'checkpoint.pt', base / 'bc-optimizer.pt')}, game_actions=0)
write_json(out, result)
print(json.dumps(result))
