"""One new STOP trial after the interrupted-artifact fix. Preserve the failed run."""
import json
import subprocess
import sys
import time

from run_rl_batch import ROOT,LINUX,linux,rearm_allowed,wait_owned
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


base=ROOT/'results/rm-planner-stop-fix-20261010';base.mkdir(exist_ok=False)
checkpoint=ROOT/'results/onset-coverage-training-20261009/checkpoint.pt'
sources=list((ROOT/'src/minecraftrl').glob('*.py'))+[ROOT/'scripts'/name for name in
    ('run_rm_planner_stop_validation.py','run_goal_recovery_pilot.py','audit_fresh_target_handoff.py','audit_rm_schedule.py')]
frozen={str(p.relative_to(ROOT)):digest(p) for p in sources}
for p in sources:
    dest=base/'source-at-game'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(p.read_bytes())
state=dict(completed=False,active=None,error=None,api_calls=0,optimizer_updates=0,frozen_sha256=frozen)
def save():write_json(base/'work.json',state)
def stop(message):
    try:
        with (ROOT/'results/STOP').open('x',encoding='utf-8') as f:f.write(message)
    except FileExistsError:return False
    return True
save()
try:
    rearm_allowed();state['active']='stop-in-plan-wait';save()
    argv=[sys.executable,str(ROOT/'scripts/run_goal_recovery_pilot.py'),'--seed','104761',
        '--sampling-seed','9721','--limit-s','180','--checkpoint',str(checkpoint),
        '--embed',str(ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'),
        '--recovery-embed',str(ROOT/'results/goal-conditioning-diagnosis-20261009/explore_9711.npy'),
        '--world',str(ROOT/'results/diverse-teacher-worlds-20261008/development-corner.zip'),
        '--rm-plan-fixture','explore','--planner-max-requests','2','--planner-delay','5']
    with (base/'game.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        state['owned_parent_pid']=proc.pid;save();started=time.monotonic();stream=None;run=None;requested=False
        try:
            while proc.poll() is None:
                if run is None:
                    lines=(base/'game.log').read_text(encoding='utf-8').splitlines()
                    if lines:
                        from pathlib import Path
                        run=Path(lines[0])/'run'
                if stream is None and run and (run/'events.jsonl').exists():
                    stream=(run/'events.jsonl').open(encoding='utf-8')
                if stream is not None and not requested:
                    while True:
                        position=stream.tell();line=stream.readline()
                        if not line.endswith('\n'):stream.seek(position);break
                        event=json.loads(line)
                        if event['event']=='planner_wait_started':
                            requested=stop('RL collection ended; controlled RM planner STOP after archive fix.')
                            write_json(base/'controlled-stop.json',dict(requested=requested,trigger=event,run=str(run)))
                            break
                if time.monotonic()-started>870:raise TimeoutError('STOP verification timed out')
                time.sleep(.05)
        finally:
            if stream is not None:stream.close()
            if proc.poll() is None:
                stop('RM planner STOP verification timeout; inspect before rearming');proc.wait(timeout=100)
    report=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    state.update(run=str(run),status=report['status'],steps=report['steps'],worker_exit_code=report['worker_exit_code'])
    save()
    assert requested and report['status']=='cancelled' and report['planner_calls']==1
    assert report['worker_exit_code']==2 and not report['rl_archive_timed_out'] and report['rl_cancelled_archive']
    assert report['release']=='neutral_command_sent' and not report['process_cleanup']['remaining_live_pids']
    assert report['actual_llm_calls']==report['rl_updates']==0 and report['switches']==[]
    assert not any(k in report for k in ('error','policy_close_error','environment_close_error','finalization_marker_error'))
    state['active']='audit';save()
    wait_owned(['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src:scripts',
        'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',
        '.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python',
        'scripts/audit_fresh_target_handoff.py',linux(run),linux(checkpoint)],base/'audit.log')
    audit=json.loads((run/'rm-plan-audit.json').read_text());assert audit['verified']
    for name,sha in frozen.items():assert digest(ROOT/name)==sha,name
    state.update(completed=True,sources_unchanged=True,audit=str(run/'rm-plan-audit.json'))
except BaseException as exc:state['error']=repr(exc)
finally:state['active']=None;save()
print(json.dumps(state))
raise SystemExit(0 if state['completed'] else 2)
