"""Audit fixed-actor data expansion against raw server events and saved heads."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from audit_rl_pilot import audit_run, local
from minecraftrl.rl_expanded import validate_rollout
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    state = json.loads((args.batch / 'experiment.json').read_text())
    assert state['completed'] and not state['errors'] and state['actor_updates'] == 0
    assert (ROOT / 'results/STOP').exists()
    parent = local(state['parent_checkpoint'], ROOT)
    assert digest(parent) == state['parent_sha256']
    heads = torch.load(parent, map_location='cpu')
    trajectories = {local(t['path'], ROOT).resolve(): t for t in state['training']['trajectories']}
    assert len(trajectories) == len(state['training']['trajectories']) == 24
    assert len(set(state['train_seeds'])) == 16
    assert set(state['train_seeds']).isdisjoint(state['validation_seeds'] + state['reserved_final_seeds'])
    for path, item in trajectories.items():
        assert digest(path) == item['sha256']
    results = []
    for run in state['runs']:
        if run['label'].startswith('prepare-'):
            continue
        print('audit ' + run['label'], flush=True)
        folder = local(run['path'], ROOT)
        report = json.loads((folder / 'summary.json').read_text())
        validate_rollout(report, 'train')
        result = audit_run(folder, heads['actor'], heads['value'], 'train')
        data = torch.load(folder / 'trajectory.pt', map_location='cpu')
        assert data['seed'] == run['seed'] and math.isfinite(data['bootstrap'])
        assert result['embedding'] == heads['condition']['embedding_sha256']
        assert digest(folder / 'goal-embed.npy') == state['embed_sha256']
        assert report['worker_exit_code'] == 0 and report['rl_updates'] == 0
        assert report['condition']['recurrent_commit_mode'] == 'dispatch_commit'
        path = (folder / 'trajectory.pt').resolve()
        assert path in trajectories and trajectories[path]['seed'] == run['seed']
        initial = json.loads((folder / 'initial-state.json').read_text())
        assert initial['seed'] == report['seed'] == run['seed']
        assert initial['initial_world']['sha256'] == state['worlds'][str(run['seed'])]['sha256']
        assert not any(initial['server']['server_inventory'].values())
        assert report['neutral_startup_steps'] == 60
        marker = json.loads((folder / 'rl-control-finished.json').read_text())
        assert marker['control_closed'] and marker['release'] == 'neutral_command_sent'
        previous = initial['server']
        started = ack = release = finished = None
        dispatches = 0
        first_inference_ms = None
        with (folder / 'events.jsonl').open() as stream:
            for line in stream:
                event = json.loads(line)
                kind = event['event']
                now = event['monotonic_s']
                if kind == 'episode_started':
                    started = now
                elif kind == 'inference_finished' and first_inference_ms is None:
                    first_inference_ms = (event['call_finished_s'] - event['call_started_s']) * 1000
                elif kind == 'policy_dispatch_started':
                    assert started is not None and now < started + state['limit_s'] and ack is None
                    dispatches += 1
                elif kind == 'stop_acknowledged':
                    ack = now
                elif kind == 'input_release' and event.get('controller') == 'cleanup':
                    release = now
                elif kind == 'episode_finished':
                    finished = now
                elif kind == 'transition':
                    current = event['info']['workshop']
                    assert current['server_tick'] - previous['server_tick'] == 1
                    assert current['world_time'] - previous['world_time'] == 1
                    previous = current
        assert ack <= release <= finished <= marker['monotonic_s']
        assert dispatches == report['steps']
        if report['status'] == 'time_limit':
            assert not data['terminal'] and ack - started >= state['limit_s']
        if data['terminal']:
            assert data['bootstrap'] == 0
        results.append(dict(seed=run['seed'],path=str(folder),terminal=data['terminal'],
            bootstrap=data['bootstrap'],dispatches_after_deadline=0,dispatches_after_ack=0,
            first_inference_ms=first_inference_ms,
            cleanup_seconds=finished-ack,remaining_live_pids=report['process_cleanup']['remaining_live_pids'],
            final_server_inventory={k:v for k,v in previous['server_inventory'].items() if v},**result))
    assert sorted(r['seed'] for r in results) == sorted(state['new_seeds'])
    for world in state['worlds'].values():
        assert digest(local(world['path'], ROOT)) == world['sha256']
    output = dict(verified=True,actor_updates=0,runs=results,new_rollouts=len(results),
        new_samples=sum(r['steps'] for r in results),successes=sum(r['success'] for r in results),
        time_limits=sum(r['status']=='time_limit' for r in results),
        total_trajectories=len(trajectories),total_samples=sum(t['samples'] for t in trajectories.values()),
        unrun_final_seeds=state['reserved_final_seeds'],evaluation_used_for_fitting=False,
        scope='training data collection only; not a before/after policy comparison')
    write_json(args.batch / 'audit.json', output)
    print(json.dumps({k:v for k,v in output.items() if k != 'runs'}))


if __name__ == '__main__':
    main()
