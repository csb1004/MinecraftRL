"""Descriptive strata of fixed OOF predictions; no fitting or model selection."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from audit_rl_pilot import local
from minecraftrl.rl_core import gae
from minecraftrl.rl_value import discounted_returns
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    parser.add_argument('--linear', type=Path, required=True)
    parser.add_argument('--mlp', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = json.loads((args.batch / 'experiment.json').read_text())
    assert state['completed'] and not state['errors'] and (ROOT / 'results/STOP').exists()
    parent = torch.load(local(state['parent_checkpoint'], ROOT), map_location='cpu')
    predictions = {}
    reports = {}
    for name, folder in (('linear', args.linear), ('mlp', args.mlp)):
        reports[name] = json.loads((folder / 'calibration.json').read_text())
        assert reports[name]['completed'] and reports[name]['parent_sha256'] == state['parent_sha256']
        rows = json.loads((folder / 'out-of-fold-predictions.json').read_text())
        predictions[name] = {r['episode_index']: r for r in rows}
        assert len(rows) == len(predictions[name]) == len(state['training']['trajectories'])
    groups = {}
    for index, item in enumerate(state['training']['trajectories']):
        path = local(item['path'], ROOT)
        assert digest(path) == item['sha256']
        data = torch.load(path, map_location='cpu')
        n = len(data['rows'])
        target = torch.tensor(discounted_returns([r['reward'] for r in data['rows']],
            data['bootstrap'], data['terminal']), dtype=torch.float64)
        features = torch.cat([r['features'][0:1] for r in data['rows']], 1).squeeze(0).double()
        values = {'inherited': torch.nn.functional.linear(features,
            parent['value']['weight'].double(), parent['value']['bias'].double()).flatten()}
        for name, lookup in predictions.items():
            row = lookup[index]
            assert row['seed'] == item['seed'] == data['seed'] and len(row['predictions']) == n
            values[name] = torch.tensor(row['predictions'], dtype=torch.float64)
            assert reports[name]['trajectories'][str(path)] == item['sha256']
        early = torch.arange(n-1, -1, -1) >= 100
        advantages = {name: torch.tensor(gae([r['reward'] for r in data['rows']], prediction.tolist(),
            data['bootstrap'], terminal=data['terminal'])[0], dtype=torch.float64)
            for name, prediction in values.items()}
        terminal = 'terminal' if data['terminal'] else 'truncated'
        origin = 'new_worlds' if data['seed'] in state['new_seeds'] else 'old_worlds'
        masks = {k: torch.ones(n, dtype=torch.bool) for k in ('all', terminal, origin)}
        masks.update(early=early, **{terminal+'_early': early, origin+'_early': early})
        for group, mask in masks.items():
            if not mask.any():
                continue
            bucket = groups.setdefault(group, dict(target=[],values={k:[] for k in values},
                advantages={k:[] for k in values},seeds=set(),episodes=0))
            bucket['target'].append(target[mask]); bucket['seeds'].add(data['seed']); bucket['episodes'] += 1
            for name, prediction in values.items():
                assert torch.isfinite(prediction).all()
                bucket['values'][name].append(prediction[mask])
                bucket['advantages'][name].append(advantages[name][mask])
    global_advantages = {k:torch.cat(v) for k,v in groups['all']['advantages'].items()}
    summary = {}
    for name, group in groups.items():
        target = torch.cat(group['target'])
        scores = {}
        for candidate, parts in group['values'].items():
            prediction = torch.cat(parts)
            scores[candidate] = dict(mse=(prediction-target).square().mean().item(),
                bias=(prediction-target).mean().item(),mean_prediction=prediction.mean().item(),
                min_prediction=prediction.min().item(),max_prediction=prediction.max().item())
            raw = torch.cat(group['advantages'][candidate])
            full = global_advantages[candidate]
            normalized = (raw-full.mean())/(full.std(unbiased=False)+1e-8)
            scores[candidate]['advantage'] = dict(raw_positive=int((raw>0).sum()),
                normalized_positive=int((normalized>0).sum()),samples=len(raw),raw_mean=raw.mean().item(),
                normalized_mean=normalized.mean().item(),observed_return_advantage_positive=int((target-prediction>0).sum()),
                gae_observed_return_sign_mismatches=int(((raw>0)!=(target-prediction>0)).sum()))
        summary[name] = dict(samples=len(target),episodes=group['episodes'],seeds=sorted(group['seeds']),
            mean_target=target.mean().item(),scores=scores)
    for name, report in reports.items():
        for group, metric in (('all', 'mse'), ('early', 'early_mse')):
            assert abs(summary[group]['scores'][name]['mse']-report['scores']['calibrated'][metric]) < 1e-6
    write_json(args.batch / 'value-strata.json', dict(verified=True,groups=summary,
        note='Descriptive only. Early means at least 100 actions before episode end. Truncated targets and all GAE candidates retain the recorded inherited endpoint bootstrap. Only interior OOF values change for the GAE diagnostic, gamma=.99 and lambda=.95; normalization uses the entire 24-episode batch. No actor update. Outcome-conditioned strata and positive-advantage counts do not identify true state value or causal action quality.'))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
