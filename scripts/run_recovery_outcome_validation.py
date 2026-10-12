"""One new local fixed-proposal trial of the corrected recovery boundary; no API."""
import json
import sys
from pathlib import Path

from run_rl_batch import ROOT, LINUX, linux, rearm_allowed, wait_owned
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


base=ROOT/'results/recovery-outcome-game-20261009'
base.mkdir(exist_ok=False)
checkpoint=ROOT/'results/onset-coverage-training-20261009/checkpoint.pt'
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
explore=ROOT/'results/goal-conditioning-diagnosis-20261009/explore_9711.npy'
world=ROOT/'results/diverse-teacher-worlds-20261008/development-corner.zip'
assert digest(checkpoint)=='473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd'
assert digest(world)=='2851b755a45623f97ef800ac844b66e0c22964481352de66f1fae4220a0d7eef'
frozen={str(p.relative_to(ROOT)):digest(p) for p in (checkpoint,embed,explore,world,Path(__file__),
    ROOT/'scripts/audit_recovery_outcome.py',ROOT/'docs/research-spec.md',
    ROOT/'docs/recovery-outcome-fix-protocol-2026-10-09.md',
    ROOT/'src/minecraftrl/planning_recovery.py',ROOT/'src/minecraftrl/recovery_evidence.py',
    ROOT/'src/minecraftrl/goal_recovery.py',ROOT/'src/minecraftrl/runtime.py',ROOT/'src/minecraftrl/cli.py')}
state=dict(completed=False,active=None,error=None,actual_games=0,api_calls=0,optimizer_updates=0,
    seed=104761,sampling_seed=9721,limit_s=180,frozen_sha256=frozen)
write_json(base/'work.json',state)
try:
    # GPU sequencing is explicit: the preceding experiment must have returned its latch.
    rearm_allowed()
    state['active']='game';write_json(base/'work.json',state)
    log=base/'game.log';launch_error=None
    try:
        wait_owned([sys.executable,str(ROOT/'scripts/run_goal_recovery_pilot.py'),
            '--seed','104761','--sampling-seed','9721','--limit-s','180',
            '--checkpoint',str(checkpoint),'--embed',str(embed),'--recovery-embed',str(explore),
            '--world',str(world),'--planner-choice','explore_then_resume','--planner-delay','5'],log)
    except RuntimeError as exc:
        launch_error=str(exc)  # A clean planning_paused diagnostic deliberately exits 2.
    lines=log.read_text(encoding='utf-8').splitlines()
    run=Path(lines[0])/'run'
    report=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    state.update(run=str(run),actual_games=int(report.get('steps',0)>0),status=report['status'],
        worker_exit_code=report['worker_exit_code'],launch_nonzero=launch_error)
    write_json(base/'work.json',state)
    if report['status']=='planning_paused':
        assert report['worker_exit_code']==2 and launch_error is not None
        assert not report['execution_valid'] and 'error' not in report
    else:
        assert report['execution_valid'] and report['worker_exit_code']==0 and launch_error is None
    assert report['release']=='neutral_command_sent' and not report['process_cleanup']['remaining_live_pids']
    assert report['actual_llm_calls']==report['rl_updates']==0 and report['planner_calls']<=1
    state['active']='audit';write_json(base/'work.json',state)
    argv=['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src',
        'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',
        '.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python',
        'scripts/audit_recovery_outcome.py',linux(run),linux(checkpoint)]
    wait_owned(argv,base/'audit.log')
    audit=json.loads((run/'recovery-outcome-audit.json').read_text())
    assert audit['verified']
    for name,sha in frozen.items():assert digest(ROOT/name)==sha,name
    state.update(completed=True,handoff_exercised=audit['handoff_exercised'],
        log_gain=audit['log_gain'],recovery_attempts=audit['recovery_attempts'],
        audit=str(run/'recovery-outcome-audit.json'),sources_unchanged=True)
except BaseException as exc:
    state['error']=repr(exc)
finally:
    state['active']=None;write_json(base/'work.json',state)
print(json.dumps(state))
raise SystemExit(0 if state['completed'] else 2)
