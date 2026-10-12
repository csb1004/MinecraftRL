"""Single-use, three-run sensor correction gate. Evaluation only, no optimizer."""
import json,sys
from pathlib import Path
from run_rl_batch import ROOT,LINUX,linux,rearm_allowed,wait_owned,compare_initial
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

base=ROOT/'results/wood-target-20261007';base.mkdir(exist_ok=False)
checkpoint=ROOT/'results/wood-shaping-v2-training-20261007/checkpoint.pt'
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
plan=[dict(label='legacy-break',mode='wood-potential-v2',world='break'),
      dict(label='corrected-break',mode='wood-potential-v3',world='break'),
      dict(label='corrected-death',mode='wood-potential-v3',world='death')]
preserved=[checkpoint,embed,ROOT/'docs/research-spec.md',*[ROOT/f'results/wood-fixtures-20261007/{k}.zip' for k in ('break','death')]]
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
        log=base/(item['label']+'.log')
        argv=[sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed','104729','--split','eval',
              '--reward-mode',item['mode'],'--world',str(ROOT/f"results/wood-fixtures-20261007/{item['world']}.zip"),
              '--checkpoint',str(checkpoint),'--embed',str(embed)]
        print(item['label'],flush=True)
        try:wait_owned(argv,log)
        finally:
            if log.exists() and log.read_text().splitlines():
                folder=Path(log.read_text().splitlines()[0])/'run'
                if (folder/'summary.json').exists():
                    report=json.loads((folder/'summary.json').read_text())
                    state['runs'].append(dict(item,path=str(folder),report=report));save()
        assert report['execution_valid'] and not report['process_cleanup']['remaining_live_pids']
        wsl=['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src',
             'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',
             '.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python']
        wait_owned(wsl+['scripts/audit_wood_shaping.py',linux(folder)],base/(item['label']+'-reward-audit.log'))
        wait_owned([sys.executable,str(ROOT/'scripts/audit_wood_target.py'),str(folder)],base/(item['label']+'-target-audit.log'))
        audit=json.loads((folder/'wood-target-audit.json').read_text())
        if item['world']=='break':
            assert report['server_target_reached'] and audit['accepted_server_log_mining_snapshots']>0
            if item['mode']=='wood-potential-v2':assert audit['legacy_target_or_stage_mismatches']>0
            else:
                state['initial_comparison']=compare_initial(Path(state['runs'][0]['path']),folder)
        else:assert report['status']=='environment_done'
        print(json.dumps(dict(label=item['label'],status=report['status'],steps=report['steps'],audited=True)),flush=True)
        save()
    for path,sha in state['preserved'].items():assert digest(ROOT/path)==sha,path
    state['completed']=True;state['active']=None
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
