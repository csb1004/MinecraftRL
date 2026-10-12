"""One approved 8x2 on-policy batch, then 2 seeds x 3 paired evaluations."""
import datetime
import argparse
import json
import sys
from pathlib import Path
from run_rl_batch import ROOT, LINUX, linux, rearm_allowed, wait_owned, compare_initial
from minecraftrl.rl_expanded import TRAIN_SEEDS, VALIDATION_SEEDS, FINAL_SEEDS, validate_rollout, training_paths
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',type=Path);args=parser.parse_args()
    base = args.resume.resolve() if args.resume else ROOT/'results'/('rl-expanded-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if not base.is_relative_to(ROOT/'results'):raise ValueError('project results required')
    if not args.resume:base.mkdir()
    previous = ROOT/'results/rl-batch-20261005T040215634778Z'
    old = json.loads((previous/'batch.json').read_text())
    checkpoint = previous/'training-retry-1/checkpoint.pt'
    embed = ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
    state = dict(completed=False, train_seeds=TRAIN_SEEDS, validation_seeds=VALIDATION_SEEDS,
                 reserved_final_seeds=FINAL_SEEDS, parent_checkpoint=str(checkpoint),
                 parent_sha256=digest(checkpoint), embed_sha256=digest(embed),
                 worlds=old['worlds'], runs=[], comparisons={}, errors=[], active=None,
                 training_repeats=2, validation_repeats=3, limit_s=180,
                 pixel_policy='record differences; require observed server and policy initial state equality')
    if args.resume:
        state=json.loads((base/'experiment.json').read_text())
        if state['completed'] or state.get('recovery_count') or not state['errors'] or (base/'training').exists():
            raise ValueError('only one pre-optimizer collection repair is supported')
        label=state['active']
        if not label.startswith('train-') or any(r['label']==label for r in state['runs']):
            raise ValueError('unexpected interrupted stage')
        log=base/(label+'.log');folder=Path(log.read_text().splitlines()[0])/'run'
        failed=json.loads((folder/'summary.json').read_text())
        if failed['status']!='worker_failed' or failed['stop_reason']!='time_limit' or failed['worker_exit_code']!=-15:
            raise ValueError('not the diagnosed finalization failure')
        with (base/'experiment-before-recovery.json').open('x',encoding='utf-8') as stream:
            stream.write((base/'experiment.json').read_text())
        archived=base/(label+'.attempt-1.log')
        if archived.exists():raise ValueError('attempt log already preserved')
        log.rename(archived)
        state['technical_attempts']=[dict(label=label,path=str(folder),report=failed,reason='supervisor finalization grace exhausted',included_in_training=False)]
        state['previous_errors']=state['errors'];state['errors']=[];state['recovery_count']=1
        state['preserved_trajectory_hashes']={str(Path(r['path'])/'trajectory.pt'):digest(Path(r['path'])/'trajectory.pt') for r in state['runs'] if r['label'].startswith('train-')}
    assert set(TRAIN_SEEDS).isdisjoint(VALIDATION_SEEDS+FINAL_SEEDS)
    def save(): write_json(base/'experiment.json', state)
    def trial(label, seed, *, prepare=False, weights=checkpoint, split='eval'):
        existing=[r for r in state['runs'] if r['label']==label]
        if existing:
            if len(existing)!=1:raise ValueError('duplicate completed stage')
            folder=Path(existing[0]['path'])
            if not prepare:
                validate_rollout(json.loads((folder/'summary.json').read_text()),split)
                path=folder/'trajectory.pt'
                if digest(path)!=state['preserved_trajectory_hashes'][str(path)]:raise ValueError('preserved data changed')
            return folder
        rearm_allowed(); state['active']=label; save(); print(label, flush=True)
        if digest(checkpoint)!=state['parent_sha256'] or digest(embed)!=state['embed_sha256']:
            raise ValueError('immutable parent or condition changed')
        argv=[sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(seed),'--split',split,'--limit-s','180']
        if prepare: argv+=['--prepare']
        else:
            item=state['worlds'][str(seed)]
            if digest(item['path'])!=item['sha256']: raise ValueError('initial world mutated')
            argv+=['--world',item['path'],'--checkpoint',str(weights),'--embed',str(embed)]
        log=base/(label+'.log'); wait_owned(argv,log)
        folder=Path(log.read_text().splitlines()[0])/'run'
        report=json.loads((folder/'summary.json').read_text())
        run=dict(label=label,seed=seed,path=str(folder),report=report)
        state['runs'].append(run); save()
        if prepare:
            if not report['execution_valid'] or report['process_cleanup']['remaining_live_pids']:
                raise ValueError('invalid preparation')
            archive=folder/'initial-world.zip'
            state['worlds'][str(seed)]=dict(path=str(archive),sha256=digest(archive));save()
        else:
            validate_rollout(report,split)
            references=[r for r in state['runs'][:-1] if r['seed']==seed and not r['label'].startswith('prepare-')]
            if references:
                comparison=compare_initial(Path(references[0]['path']),folder)
                state['comparisons'][label]=comparison;save()
                if not comparison['state_identical']:raise ValueError('observed initial state mismatch')
        return folder
    save(); (ROOT/'results/rl-expanded-current.txt').write_text(str(base)); print(base,flush=True)
    try:
        for seed in TRAIN_SEEDS:
            if str(seed) not in state['worlds']: trial('prepare-'+str(seed),seed,prepare=True)
        for repeat in (1,2):
            for seed in TRAIN_SEEDS: trial(f'train-{seed}-{repeat}',seed,split='train')
        paths=training_paths(state['runs']);rearm_allowed()
        state['active']='optimizer';save();print('optimizer',flush=True)
        trained=base/'training'
        argv=['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',
              LINUX+'/.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python','scripts/train_rl_heads.py',
              linux(paths[0]),'--additional',*[linux(p) for p in paths[1:]],'--resume',linux(checkpoint),'--output',linux(trained)]
        wait_owned(argv,base/'optimizer.log',timeout=3600)
        state['training']=json.loads((trained/'training.json').read_text());save()
        if not state['training']['completed']:raise ValueError('training did not complete')
        # Both policies are evaluated after training; fixed AB/BA/AB order.
        for repeat,order in ((1,('before','after')),(2,('after','before')),(3,('before','after'))):
            for seed in VALIDATION_SEEDS:
                for phase in order:
                    weights=checkpoint if phase=='before' else trained/'checkpoint.pt'
                    if phase=='after' and digest(weights)!=state['training']['checkpoint_sha256']:
                        raise ValueError('evaluation checkpoint changed')
                    trial(f'validation-{seed}-{repeat}-{phase}',seed,weights=weights)
        for item in state['worlds'].values():
            if digest(item['path'])!=item['sha256']:raise ValueError('world mutated')
        state['completed']=True;state['active']=None
    except BaseException as exc:
        state['errors'].append(repr(exc))
    finally: save()
    return 0 if state['completed'] else 2


if __name__=='__main__': raise SystemExit(main())
