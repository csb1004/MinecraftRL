"""Replay existing P2 server traces, never score replay as new policy execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from minecraftrl.goal_contract import read_json
from minecraftrl.logo_validators import SuccessMonitor
from minecraftrl.logo_wood import admit, stage


def replay(folder, validators):
    summary = read_json(folder/'summary.json')
    if not summary['execution_valid'] or summary['worker_exit_code'] != 0:
        raise ValueError('replay accepts completed valid source episodes only')
    baseline = read_json(folder/'initial-state.json')['server']
    monitor = SuccessMonitor(validators, baseline)
    rows, counts = 0, Counter()
    first_true = {name:0 for name,value in monitor.values.items() if value}
    trace_hash = hashlib.sha256()
    harvest_without_acquisition = 0
    completion_step = None
    with (folder/'events.jsonl').open('rb') as handle:
        for raw in handle:
            trace_hash.update(raw)
            event = json.loads(raw)
            if event['event'] != 'transition':
                continue
            rows += 1
            server = event['info']['workshop']
            reward = monitor.advance(server)
            counts[stage(monitor.values, server)] += 1
            for name, value in monitor.values.items():
                if value and name not in first_true:
                    first_true[name] = rows
            harvest_without_acquisition += int(monitor.values['log_broken'] and not monitor.completed)
            if reward:
                completion_step = rows
    expected = bool(summary['server_target_reached'])
    if monitor.completed != expected or rows != summary['steps']:
        raise ValueError('replay disagrees with original authoritative success/count')
    return dict(run=folder.name, transitions=rows, success=monitor.completed, matches_original=True,
        completion_step=completion_step, first_true_steps=first_true, stage_counts=dict(counts),
        confirmed_break_without_acquisition_ticks=harvest_without_acquisition,
        events_sha256=trace_hash.hexdigest(), source_summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    validators, review = admit(read_json(args.proposal))
    runs = sorted(folder for folder in args.source.iterdir() if folder.is_dir() and folder.name[:8].isdigit())
    if len(runs) != 15:
        raise ValueError('expected the 15 completed P2 runs; do not silently omit runs')
    rows = []
    for folder in runs:
        rows.append(replay(folder, validators))
        print(folder.name+': replay agrees', flush=True)
    report = dict(schema='logo_replay_v1', source=str(args.source), validator_sha256=validators.sha256,
        review=review, runs=rows, total_transitions=sum(row['transitions'] for row in rows),
        source_successes=sum(row['success'] for row in rows), all_match=True,
        new_gameplay=0, API_calls=0, optimizer_updates=0, scope='offline replay of development evidence; not new policy performance')
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
