"""Audit paired latent/memory controls and summarize all recorded condition effects."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from minecraftrl.correction_diagnostics import aggregate
from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root/'results/goal-conditioning-diagnosis-20261009'
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
if (base/'audit.json').exists() or (base/'paired-effects.json').exists():
    raise FileExistsError('completed diagnosis must not be rerun')
report = read(base/'report.json')
assert report['completed'] and report['error'] is None
assert report['game_actions'] == report['optimizer_updates'] == report['actual_llm_calls'] == 0
assert report['seeds'] == [9711, 9712, 9713] and report['window'] == 8 and report['cond_scale'] == 6.
assert report['prompts'] == dict(wood='collect wood', explore='go explore')
for name, sha in report['frozen_sha256'].items():
    assert digest(root/name) == sha, name
manifest = read(base/'embeddings.json')
assert len(manifest) == 6
expected_conditions = {'original_wood'} | {f'{name}_{seed}' for name in ('wood','explore') for seed in report['seeds']}
assert {row['condition'] for row in manifest} == expected_conditions-{'original_wood'}
for row in manifest:
    condition, seed = row['condition'], row['seed']
    assert row['prompt'] == report['prompts'][condition.split('_')[0]]
    assert row['official_max_error'] == 0
    latent = np.load(base/f'latent-{seed}.npy', allow_pickle=False)
    assert np.isfinite(latent).all() and state_fingerprint(torch.from_numpy(latent)) == row['latent_sha256']
    path = base/(condition+'.npy')
    assert digest(path) == row['embedding_file_sha256']
    embed = np.load(path, allow_pickle=False)
    assert embed.shape == (1,512) and np.isfinite(embed).all()
plan = read(root/'configs/onset-coverage-bc-20261009.json')
assert digest(base/'original-wood.npy') == plan['embedding_sha256']
sessions = {s['id']:s for s in plan['sessions']}
assert len(report['rows']) == len(sessions)*7 == 91
lookup = {(row['session'],row['condition']):row for row in report['rows']}
assert len(lookup) == 91
training = read(root/'results/onset-coverage-training-20261009/training.json')
assert report['checkpoint_sha256'] == training['checkpoint_sha256']
baseline_error = 0.
effects = []
for name, session in sessions.items():
    group = [row for row in report['rows'] if row['session'] == name]
    assert {r['condition'] for r in group} == expected_conditions
    assert len({r['onset_memory_sha256'] for r in group}) == 1
    onset = next(i for i,r in enumerate(session['frames']) if r['loss_eligible'])
    for row in group:
        assert row['split'] == session['split'] and row['onset_frame'] == onset
        assert len(row['per_frame']) == 8 and row['first'] == row['per_frame'][0]
        assert row['first8'] == aggregate(row['per_frame'])
        assert all(math.isfinite(v) for frame in row['per_frame'] for v in frame.values())
    original = lookup[(name,'original_wood')]['first']
    expected = training['after']['first_actions'][name]
    error = max(abs(value-expected[key]) for key,value in original.items())
    baseline_error = max(baseline_error, error)
    assert error < 1e-4
    for seed in report['seeds']:
        wood, explore = lookup[(name,f'wood_{seed}')], lookup[(name,f'explore_{seed}')]
        delta = {key:explore['first'][key]-wood['first'][key] for key in
                 ('nll','p_attack','p_move','p_release_attack_and_turn','p_release_attack_and_target_yaw')}
        effects.append(dict(session=name, split=session['split'], seed=seed,
                            original_first=original, paired_wood_first=wood['first'],
                            paired_explore_first=explore['first'], explore_minus_wood_first=delta,
                            wood_first8=wood['first8'], explore_first8=explore['first8']))
assert baseline_error == report['original_baseline_max_metric_error']
assert report['prefix_frames'] == 7652 and report['branch_frames'] == 728
summary = {}
for split in ('train','development'):
    selected = [row for row in effects if row['split'] == split]
    summary[split] = dict(pairs=len(selected), unique_episodes=len({r['session'] for r in selected}),
                         first_delta_means=aggregate([r['explore_minus_wood_first'] for r in selected]),
                         turn_improved_pairs=sum(r['explore_minus_wood_first']['p_release_attack_and_turn']>0 for r in selected))
write_json(base/'paired-effects.json', dict(rows=effects, summary=summary,
    scope='All recorded pairs; correlated episodes/latent draws, not independent gameplay trials.'))
audit = dict(verified=True, episodes=13, conditions_per_episode=7, paired_latent_draws=3,
             paired_rows=len(effects), prefix_frames=7652, branch_frames=728,
             official_embedding_max_error=0, original_baseline_max_metric_error=baseline_error,
             source_hashes_preserved=True, same_branch_start_memory=True, optimizer_updates=0, game_actions=0,
             receipts_sha256={name:digest(base/name) for name in ('report.json','embeddings.json','paired-effects.json')})
write_json(base/'audit.json', audit)
print(json.dumps(audit))
