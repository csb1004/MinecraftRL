"""Preregistered new-world comparison, no optimizer or policy selection during execution."""
import argparse,json,sys
from pathlib import Path
from run_rl_batch import ROOT,rearm_allowed,wait_owned,compare_initial
from minecraftrl.rl_expanded import validate_rollout
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

p=argparse.ArgumentParser();p.add_argument('--recover-initial-world-collision',action='store_true');args=p.parse_args()
base=ROOT/'results/rl-horizon-new-worlds-20261007'
if not args.recover_initial_world_collision:base.mkdir(exist_ok=False)
prior=ROOT/'results/rl-horizon-20261007'
old=json.loads((prior/'experiment.json').read_text())
verified=json.loads((prior/'final-verification.json').read_text());assert verified['verified']
seeds=[2100011,2100023,2100037,2100041,2100059];names=['parent','control','candidate']
assert set(seeds).isdisjoint(old['train_seeds']+old['validation_seeds']+old['final_seeds_unrun'])
checkpoints={'parent':dict(path=old['parent_checkpoint'],sha256=old['parent_sha256'])}
for name,source in [('control','g99-l95'),('candidate','g999-l95')]:
    checkpoints[name]=dict(path=str(prior/('training-'+source)/'checkpoint.pt'),sha256=old['training'][source]['checkpoint_sha256'])
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
plan=[]
for repeat in (1,2):
    for seed in seeds if repeat==1 else list(reversed(seeds)):
        offset=seeds.index(seed)%3;order=names[offset:]+names[:offset]
        if repeat==2:order=list(reversed(order))
        for name in order:plan.append(dict(label=f'eval-{repeat}-{name}-{seed}',seed=seed,repeat=repeat,condition=name))
assert len(plan)==30 and len({x['label'] for x in plan})==30
state=dict(completed=False,active=None,errors=[],worlds={},preparations=[],runs=[],comparisons={},technical_attempts=[],
    planned=plan,unrun=list(plan),seeds=seeds,checkpoints=checkpoints,embed_sha256=digest(embed),optimizer_updates=0,
    final_seeds_unrun=old['final_seeds_unrun'],protocol='docs/rl-horizon-new-worlds-protocol-2026-10-07.md')
if args.recover_initial_world_collision:
    previous=json.loads((base/'experiment.json').read_text())
    assert not previous['completed'] and not previous['worlds'] and not previous['runs'] and not previous['preparations']
    assert previous['planned']==plan and previous['checkpoints']==checkpoints
    assert previous['embed_sha256']==state['embed_sha256']
    assert len(previous['technical_attempts'])==1
    failure=previous['technical_attempts'][0]
    assert failure['label']=='prepare-2100011' and failure['report']['steps']==0
    assert failure['report']['error']=="RuntimeError('Cannot identify one owned Minecraft world')"
    assert not failure['report']['process_cleanup']['remaining_live_pids']
    rearm_allowed()
    with (base/'experiment-before-world-collision-recovery.json').open('x') as snapshot:json.dump(previous,snapshot,indent=2)
    state=previous;state['previous_errors']=state['errors'];state['errors']=[]
    state['recovery']='docs/rl-horizon-world-collision-2026-10-07.md'
def save():write_json(base/'experiment.json',state)
save()
def trial(label,seed,*,prepare=False,name='parent'):
    state['active']=label;save();print(label,flush=True)
    for item in checkpoints.values():assert digest(item['path'])==item['sha256']
    assert digest(embed)==state['embed_sha256']
    argv=[sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(seed),'--split','eval','--limit-s','180']
    if prepare:argv+=['--prepare']
    else:
        world=state['worlds'][str(seed)];assert digest(world['path'])==world['sha256']
        argv+=['--world',world['path'],'--checkpoint',checkpoints[name]['path'],'--embed',str(embed)]
    first_attempt=1+sum(x['label']==label for x in state['technical_attempts'])
    for attempt in (first_attempt,first_attempt+1):
        rearm_allowed();log=base/(label+f'-attempt-{attempt}.log')
        assert not log.exists(),'Refusing to overwrite trial evidence'
        try:wait_owned(argv,log)
        except RuntimeError:
            folder=Path(log.read_text().splitlines()[0])/'run'
            report=json.loads((folder/'summary.json').read_text())
            state['technical_attempts'].append(dict(label=label,attempt=attempt,path=str(folder),report=report));save()
            transient=set(report.get('problems',[]))
            if attempt==first_attempt and report['status']=='blocked' and report['steps']==0 and transient and transient<={'Java unavailable','nvidia-smi unavailable or failed'}:continue
            raise
        folder=Path(log.read_text().splitlines()[0])/'run'
        return folder,json.loads((folder/'summary.json').read_text())
try:
    for seed in seeds:
        folder,report=trial('prepare-'+str(seed),seed,prepare=True)
        state['preparations'].append(dict(seed=seed,path=str(folder),report=report));save()
        assert report['status']=='world_prepared' and report['execution_valid'] and not report['process_cleanup']['remaining_live_pids']
        path=folder/'initial-world.zip';state['worlds'][str(seed)]=dict(path=str(path),sha256=digest(path));save()
    for item in plan:
        folder,report=trial(item['label'],item['seed'],name=item['condition'])
        state['runs'].append(dict(item,path=str(folder),report=report))
        state['unrun']=[x for x in plan if x['label'] not in {r['label'] for r in state['runs']}];save()
        validate_rollout(report,'eval');assert not report['rl_training_eligible']
        references=[r for r in state['runs'][:-1] if r['seed']==item['seed']]
        if references:
            comparison=compare_initial(Path(references[0]['path']),folder)
            state['comparisons'][item['label']]=comparison;save();assert comparison['state_identical']
    state['completed']=True;state['active']=None
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
