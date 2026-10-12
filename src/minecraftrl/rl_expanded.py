"""Eligibility boundary for the approved eight-seed, two-rollout batch."""
from pathlib import Path

TRAIN_SEEDS = (104729, 104743, 104759, 104773, 104779, 104789, 104801, 104803)
VALIDATION_SEEDS = (2000003, 2000029)
FINAL_SEEDS = (3000017, 3000029, 3000047, 3000061, 3000073)


def validate_rollout(report, split):
    if (not report['execution_valid'] or
            report['status'] not in ('target_reached', 'time_limit', 'environment_done') or
            report['release'] != 'neutral_command_sent' or
            report['process_cleanup']['remaining_live_pids'] or report['rl_split'] != split or
            report['rl_updates'] != 0 or report['steps'] <= 0 or
            report['simulation_ticks'] != report['steps'] or
            report['server_tick_deltas'] != [1] * report['steps']):
        raise ValueError('invalid rollout; no training or performance substitution')
    if (split == 'train' and report['status'] == 'time_limit' and
            report['condition'].get('recurrent_commit_mode') != 'dispatch_commit'):
        raise ValueError('unsafe truncation bootstrap')


def training_paths(runs, seeds=TRAIN_SEEDS):
    selected = []
    for seed in seeds:
        for repeat in (1, 2):
            matches = [r for r in runs if r['label'] == f'train-{seed}-{repeat}']
            if len(matches) != 1:
                raise ValueError('missing or duplicate training run')
            validate_rollout(matches[0]['report'], 'train')
            selected.append(Path(matches[0]['path']) / 'trajectory.pt')
    if len(set(p.resolve() for p in selected)) != len(selected):
        raise ValueError('duplicate trajectory path')
    return selected
