"""Four matched updates and twenty preregistered local development rollouts."""
import datetime,json,sys
from pathlib import Path
from run_rl_batch import ROOT,LINUX,linux,rearm_allowed,wait_owned,compare_initial
from minecraftrl.rl_expanded import validate_rollout
from minecraftrl.rl_control import publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

base=ROOT/'results/rl-horizon-20261007';record=base/'experiment.json'
if record.exists():raise FileExistsError('batch is single-use; preserve any prior attempt')
source=json.loads((ROOT/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
old=json.loads((ROOT/'results/rl-batch-20261005T040215634778Z/batch.json').read_text())
diagnostic=json.loads((base/'diagnostic/comparison.json').read_text());assert diagnostic['completed']
checkpoint=Path(source['parent_checkpoint']);embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
names=['g99-l95','g999-l95','g99-l1','g999-l1']
state=dict(completed=False,active=None,errors=[],runs=[],training={},unrun=[],technical_attempts=[],
    train_seeds=source['train_seeds'],validation_seeds=[2000003,2000029],final_seeds_unrun=source['reserved_final_seeds'],
    parent_checkpoint=str(checkpoint),parent_sha256=source['parent_sha256'],embed_sha256=digest(embed),
    worlds={str(s):old['worlds'][str(s)] for s in (2000003,2000029)},comparisons={},protocol='docs/rl-horizon-protocol-2026-10-07.md')
def save():write_json(record,state)
save();stop=ROOT/'results/STOP'
# This exact latch was created by our previous deliberate cancellation test.
# The latest user explicitly authorized resuming all work; never accept any other manual STOP.
if stop.read_text()=='Backend preparation cancellation probe; keep STOP latched.':
    stop.rename(base/'STOP-before-user-resume.txt')
    if stop.exists():raise InterruptedError('new STOP while resuming')
    publish_completion(stop,'RL collection ended; user explicitly resumed horizon experiments.')
rearm_allowed()
paths=[Path(x['path'].replace('/mnt/d/DATA/Desktop/MinecraftRL/',str(ROOT)+'/')) for x in source['training']['trajectories']]
def native(p):return Path(str(p).replace('/mnt/d/DATA/Desktop/MinecraftRL/',str(ROOT)+'/'))
try:
    for name in names:
        rearm_allowed();state['active']='train-'+name;save()
        bundle=native(diagnostic['bundles'][name]['path'])
        assert digest(bundle)==diagnostic['bundles'][name]['sha256']
        assert digest(checkpoint)==state['parent_sha256']
        trained=base/('training-'+name)
        argv=['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib',
            'MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',LINUX+'/.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1',
            'python','scripts/train_rl_heads.py',linux(paths[0]),'--additional',*[linux(p) for p in paths[1:]],
            '--output',linux(trained),'--epochs','1','--target-bundle',linux(bundle)]
        wait_owned(argv,base/('train-'+name+'.log'),timeout=3600)
        report=json.loads((trained/'training.json').read_text());state['training'][name]=report;save()
        assert report['completed'] and report['updates']==(diagnostic['samples']+15)//16
    conditions=['parent']+names
    for repeat in (1,2):
        order=conditions if repeat==1 else list(reversed(conditions))
        for name in order:
            for seed in state['validation_seeds']:
                label=f'eval-{repeat}-{name}-{seed}'
                if name!='parent' and not state['training'][name]['runtime_comparison_eligible']:
                    state['unrun'].append(dict(label=label,reason='final mean reference KL exceeds .03'));save();continue
                state['active']=label;save();print(label,flush=True)
                weights=checkpoint if name=='parent' else base/('training-'+name)/'checkpoint.pt'
                expected=state['parent_sha256'] if name=='parent' else state['training'][name]['checkpoint_sha256']
                assert digest(weights)==expected and digest(embed)==state['embed_sha256']
                world=state['worlds'][str(seed)];assert digest(world['path'])==world['sha256']
                argv=[sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(seed),'--split','eval','--limit-s','180',
                    '--world',world['path'],'--checkpoint',str(weights),'--embed',str(embed)]
                folder=None
                for attempt in (1,2):
                    rearm_allowed();log=base/(label+f'-attempt-{attempt}.log')
                    try:wait_owned(argv,log)
                    except RuntimeError:
                        trial=Path(log.read_text().splitlines()[0])/'run'
                        failed=json.loads((trial/'summary.json').read_text())
                        state['technical_attempts'].append(dict(label=label,attempt=attempt,path=str(trial),report=failed));save()
                        if attempt==1 and failed['status']=='blocked' and failed['steps']==0:
                            continue
                        raise
                    folder=Path(log.read_text().splitlines()[0])/'run';break
                report=json.loads((folder/'summary.json').read_text())
                state['runs'].append(dict(label=label,condition=name,repeat=repeat,seed=seed,path=str(folder),report=report));save()
                validate_rollout(report,'eval');assert not report['rl_training_eligible']
                references=[r for r in state['runs'][:-1] if r['seed']==seed]
                if references:
                    comparison=compare_initial(Path(references[0]['path']),folder);state['comparisons'][label]=comparison;save()
                    assert comparison['state_identical']
    state['completed']=True;state['active']=None
except BaseException as exc:
    state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
