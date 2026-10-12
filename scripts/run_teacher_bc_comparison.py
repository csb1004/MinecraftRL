"""Single-use teacher BC comparison on existing development worlds, not final evaluation."""
import json,sys
from pathlib import Path
from run_rl_batch import ROOT,LINUX,linux,rearm_allowed,wait_owned,compare_initial
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

base=ROOT/'results/teacher-bc-comparison-20261008';base.mkdir(exist_ok=False)
training_dir=ROOT/'results/teacher-bc-training-20261008'
training=json.loads((training_dir/'training.json').read_text())
candidate=training_dir/'checkpoint.pt'
parent=ROOT/'results/recovery-gap-20261007/train-gap-epochs2/checkpoint.pt'
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
old=json.loads((ROOT/'results/recovery-curriculum-worlds-20261007/worlds.json').read_text())['worlds']
natural=json.loads((ROOT/'results/rl-batch-20261005T040215634778Z/batch.json').read_text())['worlds']['2000029']
worlds=[dict(old['eval-long-2000003'],kind='long_wall'),dict(natural,seed=2000029,split='eval',kind='natural')]
state=dict(completed=False,active=None,runs=[],errors=[],comparisons={},training_report=str(training_dir/'training.json'),
           parent=str(parent),candidate=str(candidate),worlds=worlds,optimizer_updates=0)
frozen={str(p):digest(p) for p in (parent,candidate,embed,ROOT/'docs/research-spec.md')}
state['frozen']=frozen
def save():write_json(base/'experiment.json',state)
def trial(label,world,checkpoint):
    rearm_allowed()
    for path,sha in frozen.items():assert digest(path)==sha
    assert digest(world['path'])==world['sha256']
    state['active']=label;save();print(label,flush=True);log=base/(label+'.log')
    try:
        wait_owned([sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(world['seed']),'--split','eval',
            '--sampling-seed',str(world['seed']),'--reward-mode','wood-potential-v4','--world',world['path'],
            '--checkpoint',str(checkpoint),'--embed',str(embed)],log)
    finally:
        lines=log.read_text().splitlines() if log.exists() else []
        if lines:
            run=Path(lines[0])/'run'
            if (run/'summary.json').exists():
                report=json.loads((run/'summary.json').read_text())
                state['runs'].append(dict(label=label,path=str(run),world=world,checkpoint_sha256=digest(checkpoint),
                                         report=report,audited=False));save()
    assert report['execution_valid'] and report['worker_exit_code']==0 and report['rl_split']=='eval'
    assert not report['rl_training_eligible'] and not report['process_cleanup']['remaining_live_pids']
    release=json.loads((run/'input-release-state.json').read_text());assert release['verified'] and release['held_key_count']==0
    wsl=['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src','LD_LIBRARY_PATH=/usr/lib/wsl/lib',
         'MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba','.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python']
    wait_owned(wsl+['scripts/audit_wood_shaping.py',linux(run)],base/(label+'-reward-audit.log'))
    wait_owned([sys.executable,str(ROOT/'scripts/audit_approach_recovery.py'),str(run)],base/(label+'-recovery-audit.log'))
    state['runs'][-1].update(audited=True,release_evidence=release);save()
    print(json.dumps(dict(label=label,steps=report['steps'],success=report['server_target_reached'])),flush=True)
    return run

save()
try:
    assert training['completed'] and digest(parent)==training['parent_sha256']
    assert digest(candidate)==training['checkpoint_sha256']
    if not training['runtime_comparison_eligible']:
        state['unrun_reason']='BC offline comparison gate not met; no gameplay sample counted'
    else:
        for i,world in enumerate(worlds):
            pair={}
            for name in (('parent','candidate') if i==0 else ('candidate','parent')):
                pair[name]=trial(world['kind']+'-'+name,world,parent if name=='parent' else candidate)
            comparison=compare_initial(pair['parent'],pair['candidate'])
            state['comparisons'][world['kind']]=comparison;save()
            assert comparison['state_identical'],'initial state mismatch'
    for path,sha in frozen.items():assert digest(path)==sha
    for w in worlds:assert digest(w['path'])==w['sha256']
    state['completed']=True;state['active']=None
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
