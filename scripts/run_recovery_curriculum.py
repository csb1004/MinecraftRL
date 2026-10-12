"""Bounded two-stage on-policy curriculum followed by held-out paired trials."""
import json,sys
from pathlib import Path
from run_rl_batch import ROOT,LINUX,linux,rearm_allowed,wait_owned,compare_initial
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

base=ROOT/'results/recovery-curriculum-20261007';base.mkdir(exist_ok=False)
worlds=json.loads((ROOT/'results/recovery-curriculum-worlds-20261007/worlds.json').read_text())
assert worlds['completed'];worlds=worlds['worlds']
parent=ROOT/'results/wood-shaping-v2-training-20261007/checkpoint.pt'
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
train_seeds=[104743,104759];eval_seeds=[2000003,2000029];final_seeds=[3000017,3000029,3000047,3000061,3000073]
assert set(train_seeds).isdisjoint(eval_seeds+final_seeds)
state=dict(completed=False,active=None,runs=[],training=[],errors=[],worlds=worlds,comparisons={},
           parent=str(parent),parent_sha256=digest(parent),embed_sha256=digest(embed),
           research_spec_sha256=digest(ROOT/'docs/research-spec.md'),final_seeds_unrun=final_seeds,
           source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src/minecraftrl').iterdir() if p.suffix in ('.py','.java')},
           trainer_sha256=digest(ROOT/'scripts/train_rl_heads.py'),accepted_checkpoint=None)
def save():write_json(base/'experiment.json',state)
def frozen():
    for p,sha in state['source_sha256'].items():assert digest(ROOT/p)==sha,p
    assert digest(parent)==state['parent_sha256'] and digest(embed)==state['embed_sha256']
    assert digest(ROOT/'scripts/train_rl_heads.py')==state['trainer_sha256']
def wsl():
    return ['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src','LD_LIBRARY_PATH=/usr/lib/wsl/lib',
            'MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba','.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python']
def trial(label,world,checkpoint):
    frozen();rearm_allowed();state['active']=label;save();print(label,flush=True)
    assert digest(world['path'])==world['sha256']
    log=base/(label+'.log')
    argv=[sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(world['seed']),'--split',world['split'],
          '--reward-mode','wood-potential-v4','--world',world['path'],'--checkpoint',str(checkpoint),'--embed',str(embed)]
    try:wait_owned(argv,log)
    finally:
        if log.exists() and log.read_text().splitlines():
            folder=Path(log.read_text().splitlines()[0])/'run'
            if (folder/'summary.json').exists():
                report=json.loads((folder/'summary.json').read_text())
                state['runs'].append(dict(label=label,path=str(folder),world=world,checkpoint=str(checkpoint),
                                          checkpoint_sha256=digest(checkpoint),report=report,audited=False));save()
    assert report['execution_valid'] and report['worker_exit_code']==0 and not report['process_cleanup']['remaining_live_pids']
    assert report['rl_split']==world['split'] and report['rl_training_eligible']==(world['split']=='train')
    wait_owned(wsl()+['scripts/audit_wood_shaping.py',linux(folder)],base/(label+'-reward-audit.log'))
    wait_owned([sys.executable,str(ROOT/'scripts/audit_approach_recovery.py'),str(folder)],base/(label+'-recovery-audit.log'))
    state['runs'][-1]['audited']=True;save()
    print(json.dumps(dict(label=label,status=report['status'],steps=report['steps'],success=report['server_target_reached'])),flush=True)
    return state['runs'][-1]
def train(stage,runs):
    assert runs and all(r['audited'] and r['world']['split']=='train' for r in runs)
    assert len({r['checkpoint_sha256'] for r in runs})==1
    assert any(r['report']['server_target_reached'] for r in runs)
    paths=[Path(r['path'])/'trajectory.pt' for r in runs]
    for epochs in (2,1):
        frozen();rearm_allowed();label=f'train-{stage}-epochs{epochs}'
        state['active']=label;save();print(label,flush=True);out=base/label
        argv=wsl()+['scripts/train_rl_heads.py',linux(paths[0]),'--additional',*[linux(p) for p in paths[1:]],
                    '--output',linux(out),'--epochs',str(epochs)]
        try:wait_owned(argv,base/(label+'.log'),timeout=3600)
        finally:
            if (out/'training.json').exists():
                report=json.loads((out/'training.json').read_text());state['training'].append(dict(stage=stage,path=str(out),report=report));save()
        assert report['completed']
        if report['runtime_comparison_eligible']:
            checkpoint=out/'checkpoint.pt';assert digest(checkpoint)==report['checkpoint_sha256']
            return checkpoint
    return None

save();print(base,flush=True)
try:
    checkpoint=parent
    for stage in ('short','medium'):
        runs=[]
        for repeat in (1,2):
            for seed in train_seeds:runs.append(trial(f'collect-{stage}-{seed}-{repeat}',worlds[f'train-{stage}-{seed}'],checkpoint))
            if any(r['report']['server_target_reached'] for r in runs):break
        if not any(r['report']['server_target_reached'] for r in runs):
            state['curriculum_stop']=dict(stage=stage,reason='no_acquisition_in_four_training_trials');save();break
        candidate=train(stage,runs)
        if candidate is None:
            state['curriculum_stop']=dict(stage=stage,reason='both_epoch_budgets_exceeded_training_KL_limit');save();break
        checkpoint=candidate;state['accepted_checkpoint']=dict(path=str(checkpoint),sha256=digest(checkpoint));save()
    if checkpoint!=parent:
        for i,seed in enumerate(eval_seeds):
            pair={}
            for name in (('parent','candidate') if i==0 else ('candidate','parent')):
                pair[name]=trial(f'eval-{seed}-{name}',worlds[f'eval-long-{seed}'],parent if name=='parent' else checkpoint)
            comparison=compare_initial(Path(pair['parent']['path']),Path(pair['candidate']['path']))
            state['comparisons'][str(seed)]=comparison;save()
            assert comparison['state_identical'],'initial state mismatch'
    else:state['evaluation_unrun_reason']='no trained checkpoint passed acquisition and KL gates'
    frozen()
    for world in worlds.values():assert digest(world['path'])==world['sha256']
    assert digest(ROOT/'docs/research-spec.md')==state['research_spec_sha256']
    state['completed']=True;state['active']=None
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
