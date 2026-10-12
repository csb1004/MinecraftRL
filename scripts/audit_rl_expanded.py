"""Independent raw-event audit and descriptive evaluation of the expanded batch."""
import argparse
import json
import math
import statistics
from pathlib import Path
import torch
from audit_rl_pilot import audit_run, local
from run_rl_batch import compare_initial
from minecraftrl.rl_expanded import TRAIN_SEEDS, VALIDATION_SEEDS, FINAL_SEEDS, training_paths, validate_rollout
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('batch',type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];base=args.batch
    state=json.loads((base/'experiment.json').read_text())
    assert state['completed'] and not state['errors']
    assert state['train_seeds']==list(TRAIN_SEEDS) and state['validation_seeds']==list(VALIDATION_SEEDS)
    assert state['reserved_final_seeds']==list(FINAL_SEEDS)
    parent=local(state['parent_checkpoint'],root)
    assert digest(parent)==state['parent_sha256']
    before=torch.load(parent,map_location='cpu');after=torch.load(base/'training/checkpoint.pt',map_location='cpu')
    training=state['training']
    assert training['completed'] and training['parent_checkpoint_sha256']==state['parent_sha256']
    assert digest(base/'training/checkpoint.pt')==training['checkpoint_sha256']
    assert training['updates']==2*math.ceil(training['samples']/16)
    assert before['updates']+training['updates']==training['total_updates']==after['updates']
    assert all(v['step'].item()==after['updates'] for v in after['optimizer']['state'].values())
    trajectories={local(t['path'],root).resolve():t for t in training['trajectories']}
    expected={local(p,root).resolve() for p in training_paths(state['runs'])}
    assert len(trajectories)==16 and set(trajectories)==expected
    results=[];first={}
    for run in state['runs']:
        if run['label'].startswith('prepare-'):continue
        print('audit '+run['label'],flush=True)
        folder=local(run['path'],root);report=json.loads((folder/'summary.json').read_text())
        train=run['label'].startswith('train-');phase='after' if run['label'].endswith('-after') else 'before'
        validate_rollout(report,'train' if train else 'eval')
        heads=after if phase=='after' else before
        result=audit_run(folder,heads['actor'],heads['value'],'train' if train else 'eval')
        assert result['embedding']==before['condition']['embedding_sha256']
        path=(folder/'trajectory.pt').resolve();assert (path in trajectories)==train
        if train:assert digest(path)==trajectories[path]['sha256']
        seed=run['seed'];initial=json.loads((folder/'initial-state.json').read_text())
        assert report['seed']==initial['seed']==seed
        assert initial['initial_world']['sha256']==state['worlds'][str(seed)]['sha256']
        assert not any(initial['server']['server_inventory'].values()) and report['neutral_startup_steps']==60
        reference=first.setdefault(seed,folder);comparison=compare_initial(reference,folder)
        assert comparison['state_identical']
        previous=initial['server'];steps=0
        with (folder/'events.jsonl').open() as stream:
            for line in stream:
                event=json.loads(line)
                if event['event']!='transition':continue
                current=event['info']['workshop']
                assert current['server_tick']-previous['server_tick']==1 and current['world_time']-previous['world_time']==1
                previous=current;steps+=1
        assert steps==report['steps']
        results.append(dict(label=run['label'],seed=seed,split='train' if train else 'eval',phase=phase,
                            initial_pixels_match_first=comparison['pixel_identical'],
                            final_server_inventory={k:v for k,v in previous['server_inventory'].items() if v},**result))
    assert len(results)==28
    aggregates={}
    for seed in VALIDATION_SEEDS:
        aggregates[str(seed)]={}
        for phase in ('before','after'):
            group=[r for r in results if r['split']=='eval' and r['seed']==seed and r['phase']==phase]
            assert len(group)==3
            successes=[r for r in group if r['success']]
            aggregates[str(seed)][phase]=dict(attempted=3,successes=len(successes),
                time_limits=sum(r['status']=='time_limit' for r in group),
                steps=[r['steps'] for r in group],seconds=[r['seconds'] for r in group],
                successful_median_steps=statistics.median(r['steps'] for r in successes) if successes else None,
                successful_median_seconds=statistics.median(r['seconds'] for r in successes) if successes else None)
    for item in state['worlds'].values():assert digest(local(item['path'],root))==item['sha256']
    technical=[]
    for attempt in state.get('technical_attempts',[]):
        folder=local(attempt['path'],root);report=json.loads((folder/'summary.json').read_text())
        assert not report['execution_valid'] and (folder/'trajectory.pt').resolve() not in trajectories
        acknowledged=False;steps=0;observed_log=False
        with (folder/'events.jsonl').open() as stream:
            for line in stream:
                event=json.loads(line)
                if event['event']=='stop_acknowledged':acknowledged=True
                if event['event']=='policy_dispatch_started':assert not acknowledged
                if event['event']=='transition':
                    steps+=1
                    observed_log=observed_log or any(v for k,v in event['info']['workshop']['server_inventory'].items() if k.endswith('_log'))
        assert steps==report['steps'] and acknowledged and not report['process_cleanup']['remaining_live_pids']
        technical.append(dict(path=str(folder),status=report['status'],stop_reason=report['stop_reason'],
                              steps=steps,seconds=report['elapsed_control_s'],observed_server_log=observed_log,
                              included_in_training=False,worker_exit_code=report['worker_exit_code']))
    output=dict(verified=True,runs=results,validation=aggregates,training_samples=training['samples'],
                updates=training['updates'],total_updates=training['total_updates'],evaluation_used_for_training=False,
                technical_attempts=technical,
                unrun_final_seeds=FINAL_SEEDS,scope='descriptive small repeated validation; no general improvement claim')
    write_json(base/'audit.json',output)
    print(json.dumps(dict(verified=True,updates=training['updates'],validation=aggregates)))


if __name__=='__main__': main()
