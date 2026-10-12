"""Single-use two-run collection; no optimizer or held-out evaluation."""
import json,sys
from pathlib import Path
from run_rl_batch import ROOT,LINUX,linux,rearm_allowed,wait_owned
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

base=ROOT/'results/blocked-entry-20261008';base.mkdir(exist_ok=False)
prepared=json.loads((ROOT/'results/blocked-entry-worlds-20261008/worlds.json').read_text())
assert prepared['completed']
checkpoint=ROOT/'results/recovery-gap-20261007/train-gap-epochs2/checkpoint.pt'
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
frozen={str(p):digest(p) for p in (checkpoint,embed,ROOT/'docs/research-spec.md')}
state=dict(completed=False,active=None,runs=[],errors=[],worlds=prepared['worlds'],optimizer_updates=0,
           frozen=frozen,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src/minecraftrl').iterdir() if p.suffix in ('.py','.java')})
def save():write_json(base/'experiment.json',state)
save()
try:
    for seed,world in prepared['worlds'].items():
        rearm_allowed()
        for path,sha in frozen.items():assert digest(path)==sha
        assert digest(world['path'])==world['sha256']
        label='collect-blocked-'+seed;state['active']=label;save();print(label,flush=True)
        log=base/(label+'.log')
        try:
            wait_owned([sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',seed,'--split','train',
                '--sampling-seed',str(world['sampling_seed']),'--reward-mode','wood-potential-v4',
                '--world',world['path'],'--checkpoint',str(checkpoint),'--embed',str(embed)],log)
        finally:
            lines=log.read_text().splitlines() if log.exists() else []
            if lines:
                run=Path(lines[0])/'run'
                if (run/'summary.json').exists():
                    report=json.loads((run/'summary.json').read_text())
                    state['runs'].append(dict(label=label,path=str(run),world=world,report=report,audited=False));save()
        assert report['execution_valid'] and report['worker_exit_code']==0
        assert report['rl_training_eligible'] and report['condition']['sampling_seed']==world['sampling_seed']
        assert not report['process_cleanup']['remaining_live_pids']
        release=json.loads((run/'input-release-state.json').read_text())
        assert release['verified'] and release['held_key_count']==0
        wsl=['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src',
             'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',
             '.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python']
        wait_owned(wsl+['scripts/audit_wood_shaping.py',linux(run)],base/(label+'-reward-audit.log'))
        wait_owned([sys.executable,str(ROOT/'scripts/audit_approach_recovery.py'),str(run)],base/(label+'-recovery-audit.log'))
        state['runs'][-1].update(audited=True,release_evidence=release);save()
        print(json.dumps(dict(label=label,status=report['status'],steps=report['steps'],success=report['server_target_reached'])),flush=True)
    for path,sha in frozen.items():assert digest(path)==sha
    for path,sha in state['source_sha256'].items():assert digest(ROOT/path)==sha
    for world in state['worlds'].values():assert digest(world['path'])==world['sha256']
    state['completed']=True;state['active']=None
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
