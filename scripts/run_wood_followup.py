"""Single-use fixed checkpoint comparison. No training or automatic promotion."""
import json,sys
from pathlib import Path
from run_rl_batch import ROOT,rearm_allowed,wait_owned,compare_initial
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

base=ROOT/'results/wood-followup-20261007';base.mkdir(exist_ok=False)
natural=json.loads((ROOT/'results/rl-horizon-new-worlds-20261007/experiment.json').read_text())['worlds']
weights={'parent':ROOT/'results/rl-batch-20261005T040215634778Z/training-retry-1/checkpoint.pt',
         'v2':ROOT/'results/wood-shaping-v2-training-20261007/checkpoint.pt'}
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
plan=[]
for kind in ('break','detour'):
    for route in ('break','detour'):
        plan.append(dict(label=f'probe-{kind}-{route}',seed=104729,world=str(ROOT/f'results/wood-fixtures-20261007/{kind}.zip'),probe=route))
    for condition in weights:
        plan.append(dict(label=f'policy-{kind}-{condition}',seed=104729,world=str(ROOT/f'results/wood-fixtures-20261007/{kind}.zip'),condition=condition))
plan.append(dict(label='probe-death',seed=104729,world=str(ROOT/'results/wood-fixtures-20261007/death.zip'),probe='death'))
for i,(seed,world) in enumerate(natural.items()):
    for condition in (list(weights) if i%2==0 else list(reversed(weights))):
        plan.append(dict(label=f'natural-{seed}-{condition}',seed=int(seed),world=world['path'],condition=condition))
state=dict(completed=False,planned=plan,runs=[],unrun=list(plan),errors=[],active=None,optimizer_updates=0,
           checkpoints={k:dict(path=str(p),sha256=digest(p)) for k,p in weights.items()},
           source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src/minecraftrl').glob('*.py')},
           world_sha256={p['world']:digest(p['world']) for p in plan},embed_sha256=digest(embed),
           final_seeds_unrun=[3000017,3000029,3000047,3000061,3000073])
def save():write_json(base/'experiment.json',state)
save();print(base,flush=True)
try:
    for item in plan:
        state['active']=item['label'];save();rearm_allowed()
        for path,sha in state['source_sha256'].items():assert digest(ROOT/path)==sha,path
        for item_weight in state['checkpoints'].values():assert digest(item_weight['path'])==item_weight['sha256']
        assert digest(item['world'])==state['world_sha256'][item['world']]
        argv=[sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(item['seed']),'--split','eval',
              '--reward-mode','wood-potential-v2','--world',item['world']]
        if 'probe' in item:argv+=['--probe-route',item['probe']]
        else:argv+=['--checkpoint',str(weights[item['condition']]),'--embed',str(embed)]
        log=base/(item['label']+'.log');print(item['label'],flush=True)
        try:wait_owned(argv,log)
        finally:
            if log.exists() and log.read_text().splitlines():
                folder=Path(log.read_text().splitlines()[0])/'run'
                if (folder/'summary.json').exists():
                    report=json.loads((folder/'summary.json').read_text())
                    state['runs'].append(dict(item,path=str(folder),report=report))
                    state['unrun']=[p for p in plan if p['label'] not in {r['label'] for r in state['runs']}];save()
        assert report['execution_valid'] and not report['process_cleanup']['remaining_live_pids']
        if item.get('probe') in ('break','detour'):
            assert report['server_target_reached'], 'Calibration did not acquire wood: '+item['label']
        if 'probe' not in item:
            previous=[r for r in state['runs'][:-1] if r.get('world')==item['world'] and 'probe' not in r]
            if previous:
                state.setdefault('comparisons',{})[item['label']]=compare_initial(Path(previous[0]['path']),folder);save()
        print(json.dumps(dict(label=item['label'],status=report['status'],steps=report['steps'])),flush=True)
    state['completed']=True;state['active']=None
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
