"""Check exact target exposure and checkpoint provenance after a completed onset pilot."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from minecraftrl.onset_bc import correction_schedule
from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.temporal_bc import episode_rows
from minecraftrl.world_store import digest


parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
base = args.run.resolve()
if not base.is_relative_to(root/'results'):
    raise ValueError('project-local result required')
out = base/'supervision-audit.json'
if out.exists():
    raise FileExistsError(out)
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
report = read(base/'training.json')
assert report['completed'] and report['error'] is None and report['updates'] == report['planned_updates']
assert report['mode'] == 'coverage_onset' and 260 < report['updates'] <= 2400
features = root/'results/onset-coverage-features-20261009/features.pt'
assert digest(features) == report['features_sha256']
data = torch.load(features, map_location='cpu', weights_only=False)
groups = episode_rows(data)
splits = {s['id']: s['split'] for s in data['plan']['sessions']}
schedule = correction_schedule(groups, splits, 'onset', seed=report['order_seed'])
assert schedule == read(base/'schedule.json') and digest(base/'schedule.json') == report['schedule_sha256']
updates = [json.loads(line) for line in (base/'updates.jsonl').read_text().splitlines()]
assert len(updates) == len(schedule)
exposures = Counter()
onset_context = 0
for index, (unit, update) in enumerate(zip(schedule, updates), 1):
    assert update['update'] == index and all(update[k] == v for k, v in unit.items())
    assert splits[unit['session']] == 'train'
    rows = groups[unit['session']][unit['start']:unit['end']]
    targets = [r for r in rows if r['loss_eligible']]
    assert update['supervised_frames'] == len(targets) and update['context_frames'] == len(rows)-len(targets)
    if unit['kind'] == 'onset':
        assert len(targets) == 1 and len(rows) == 8
        onset_context += 7
    for row in rows:
        if row['loss_eligible']:
            assert row['owner'] == 'teacher' and isinstance(row['action'], dict)
            exposures[(unit['session'], row['frame'])] += 1
        else:
            assert row['action'] is None
train_labels = sum(s['expert_labels'] for s in data['plan']['sessions'] if s['split'] == 'train')
train_sessions = sum(s['split'] == 'train' for s in data['plan']['sessions'])
assert len(exposures) == report['unique_teacher_targets'] == train_labels
assert sum(exposures.values()) == report['total_teacher_target_exposures'] == train_labels+len(schedule)//2
assert read(base/'target-exposures.json') == [dict(session=k[0], frame=k[1], count=v) for k, v in sorted(exposures.items())]
assert len([u for u in schedule if u['kind'] == 'onset']) == len(schedule)//2
for milestone in sorted({math.ceil(len(schedule)*i/5) for i in range(1, 6)}):
    diagnostic = read(base/f'train-onset-{milestone:03d}.json')
    assert set(diagnostic['splits']) == {'train'}
    assert set(diagnostic['first_actions']) == {name for name in groups if splits[name] == 'train'}
    assert diagnostic['splits']['train']['first']['nll']['count'] == train_sessions
parent_path = root/'results/masked-correction-training-20261009/checkpoint.pt'
assert digest(parent_path) == report['parent_sha256']
assert digest(base/'checkpoint.pt') == report['checkpoint_sha256']
assert digest(base/'bc-optimizer.pt') == report['optimizer_sha256']
parent = torch.load(parent_path, map_location='cpu', weights_only=False)
candidate = torch.load(base/'checkpoint.pt', map_location='cpu', weights_only=False)
assert candidate['scope'] == parent['scope'] == 'head_tail_last_recurrent'
assert candidate['bc']['value_frozen'] and not candidate['bc']['ppo_optimizer_resumable']
assert candidate['bc']['method'] == 'coverage_onset_bc'
assert candidate['bc']['burnin'] == report['burnin'] == 'exact_last_block_cache_current_weights'
assert candidate['condition'] == parent['condition']
assert state_fingerprint(candidate['value']) == state_fingerprint(parent['value'])
for group in ('actor', 'backbone_tail', 'backbone_temporal'):
    assert state_fingerprint(candidate[group]) != state_fingerprint(parent[group])
assert report['reload_max_error'] == 0
minimum = min(v['p_release_attack_and_turn'] for name, v in report['after']['first_actions'].items()
              if splits[name] == 'development')
a, b = report['before']['splits']['development'], report['after']['splits']['development']
assert report['diagnostic_min_development_onset_turn_probability'] == .005
assert minimum == report['development_onset_min_turn_probability']
expected_gate = (b['first']['nll']['mean'] <= a['first']['nll']['mean']-1 and
    b['first']['p_release_attack_and_turn']['mean'] > a['first']['p_release_attack_and_turn']['mean'] and
    b['continuation']['nll']['mean'] <= 1.1*a['continuation']['nll']['mean'] and minimum >= .005)
assert report['diagnostic_eligible'] == expected_gate
for name, sha in report['frozen_sha256'].items():
    assert digest(root/name) == sha, name
for split, count in (('train', train_labels), ('development', 619)):
    assert report['before']['splits'][split]['teacher']['reference_kl']['mean'] == 0
    assert report['after']['splits'][split]['teacher']['nll']['count'] == count
    assert 'nll' not in report['after']['splits'][split]['context']
result = dict(verified=True, mode=report['mode'], optimizer_updates=len(schedule), unique_teacher_targets=train_labels,
              total_teacher_target_exposures=sum(exposures.values()), onset_context_frames=onset_context,
              student_direct_targets=0, development_optimizer_targets=0, preserved_value=True,
              parent_reference_own_features=True, reload_max_error=0, sources_unchanged=True,
              receipts_sha256={name:digest(base/name) for name in ('training.json', 'schedule.json', 'updates.jsonl', 'target-exposures.json')})
write_json(out, result)
print(json.dumps(result))
