"""Single-use local diagnostic comparison; no training or policy promotion."""
import json,sys
from pathlib import Path
from run_rl_batch import ROOT,LINUX,linux,rearm_allowed,wait_owned,compare_initial
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

base=ROOT/'results/approach-recovery-20261007';base.mkdir(exist_ok=False)
checkpoint=ROOT/'results/wood-shaping-v2-training-20261007/checkpoint.pt'
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
arena=ROOT/'results/approach-recovery-fixture-20261007/recovery.zip'
death=ROOT/'results/wood-fixtures-20261007/death.zip'
plan=[dict(label='calibration-detour',mode='wood-potential-v2',world=str(arena),probe='detour'),
      dict(label='baseline-v3',mode='wood-potential-v3',world=str(arena)),
      dict(label='recovery-v4',mode='wood-potential-v4',world=str(arena)),
      dict(label='death-v4',mode='wood-potential-v4',world=str(death))]
preserved=[checkpoint,embed,arena,death,ROOT/'docs/research-spec.md']
state=dict(completed=False,planned=plan,runs=[],errors=[],active=None,optimizer_updates=0,
           preserved={str(p.relative_to(ROOT)):digest(p) for p in preserved},
           source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src/minecraftrl').iterdir() if p.suffix in ('.py','.java')},
           final_seeds_unrun=[3000017,3000029,3000047,3000061,3000073])
def save():write_json(base/'experiment.json',state)
save();print(base,flush=True)
try:
    for item in plan:
        state['active']=item['label'];save();rearm_allowed()
        for path,sha in {**state['preserved'],**state['source_sha256']}.items():assert digest(ROOT/path)==sha,path
        argv=[sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed','104729','--split','eval',
              '--reward-mode',item['mode'],'--world',item['world']]
        if 'probe' in item:argv+=['--probe-route',item['probe']]
        else:argv+=['--checkpoint',str(checkpoint),'--embed',str(embed)]
        log=base/(item['label']+'.log');print(item['label'],flush=True)
        try:wait_owned(argv,log)
        finally:
            if log.exists() and log.read_text().splitlines():
                folder=Path(log.read_text().splitlines()[0])/'run'
                if (folder/'summary.json').exists():
                    report=json.loads((folder/'summary.json').read_text())
                    state['runs'].append(dict(item,path=str(folder),report=report));save()
        assert report['execution_valid'] and not report['process_cleanup']['remaining_live_pids']
        if 'probe' in item:
            wait_owned([sys.executable,str(ROOT/'scripts/audit_fixture_probe.py'),str(folder)],base/(item['label']+'-audit.log'))
            assert report['server_target_reached'],'detour calibration did not acquire wood'
        else:
            wsl=['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src',
                 'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',
                 '.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python']
            wait_owned(wsl+['scripts/audit_wood_shaping.py',linux(folder)],base/(item['label']+'-reward-audit.log'))
            wait_owned([sys.executable,str(ROOT/'scripts/audit_approach_recovery.py'),str(folder)],base/(item['label']+'-recovery-audit.log'))
            if item['label']=='recovery-v4':state['initial_comparison']=compare_initial(Path(state['runs'][1]['path']),folder)
            if item['label']=='death-v4':assert report['status']=='environment_done'
        print(json.dumps(dict(label=item['label'],status=report['status'],steps=report['steps'],audited=True)),flush=True);save()
    for path,sha in state['preserved'].items():assert digest(ROOT/path)==sha,path
    state['completed']=True;state['active']=None
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
