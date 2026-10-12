"""One-use real fixed-plan integration and separate STOP trial; no paid API."""
import json
import subprocess
import sys
import time
from pathlib import Path

from run_rl_batch import ROOT,LINUX,linux,rearm_allowed,wait_owned
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


base=ROOT/'results/rm-planner-game-20261010';base.mkdir(exist_ok=False)
checkpoint=ROOT/'results/onset-coverage-training-20261009/checkpoint.pt'
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
explore=ROOT/'results/goal-conditioning-diagnosis-20261009/explore_9711.npy'
world=ROOT/'results/diverse-teacher-worlds-20261008/development-corner.zip'
assert digest(checkpoint)=='473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd'
assert digest(world)=='2851b755a45623f97ef800ac844b66e0c22964481352de66f1fae4220a0d7eef'
assert digest(ROOT/'docs/research-spec.md')=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
sources=[p for p in (ROOT/'src/minecraftrl').glob('*.py')]
sources += [ROOT/'scripts'/s for s in ('run_goal_recovery_pilot.py','audit_fresh_target_handoff.py',
                                    'audit_rm_schedule.py','run_rm_planner_validation.py')]
frozen={str(p.relative_to(ROOT)):digest(p) for p in sources+[checkpoint,embed,explore,world,ROOT/'docs/research-spec.md']}
for source in sources:
    dest=base/'source-at-game'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(source.read_bytes())
state=dict(completed=False,active=None,error=None,api_calls=0,optimizer_updates=0,runs=[],frozen_sha256=frozen)
def save():write_json(base/'work.json',state)
save()


def trial(label,controlled_stop=False):
    rearm_allowed();state['active']=label;save()
    argv=[sys.executable,str(ROOT/'scripts/run_goal_recovery_pilot.py'),'--seed','104761',
          '--sampling-seed','9721','--limit-s','180','--checkpoint',str(checkpoint),'--embed',str(embed),
          '--recovery-embed',str(explore),'--world',str(world),'--rm-plan-fixture','explore',
          '--planner-max-requests','2','--planner-delay','5']
    logpath=base/(label+'.log');requested=False;run=None
    with logpath.open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        state['owned_parent_pid']=proc.pid;save();started=time.monotonic()
        try:
            while proc.poll() is None:
                lines=logpath.read_text(encoding='utf-8').splitlines()
                if lines:run=Path(lines[0])/'run'
                if controlled_stop and run and not requested and (run/'events.jsonl').exists():
                    for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines():
                        try:event=json.loads(line)
                        except json.JSONDecodeError:continue
                        if event.get('event')=='planner_wait_started':
                            (ROOT/'results/STOP').write_text('RL collection ended; controlled RM planner wait STOP.',encoding='utf-8')
                            write_json(base/'controlled-stop.json',dict(parent_pid=proc.pid,trigger=event,run=str(run)))
                            requested=True;break
                if time.monotonic()-started>870:raise TimeoutError('owned RM planner trial timed out')
                time.sleep(.1)
        finally:
            if proc.poll() is None:
                (ROOT/'results/STOP').write_text('RM planner validation timeout; investigate before rearming')
                proc.wait(timeout=100)
    if run is None:raise RuntimeError('worker did not publish output path')
    report=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    row=dict(label=label,path=str(run),status=report['status'],worker_exit_code=report['worker_exit_code'],
             steps=report['steps'],planner_calls=report['planner_calls'],stop_requested=requested)
    state['runs'].append(row);save()
    assert report['release']=='neutral_command_sent' and not report['process_cleanup']['remaining_live_pids']
    assert report['actual_llm_calls']==report['rl_updates']==0 and report['planner_calls']<=2
    assert 'error' not in report
    if controlled_stop:
        assert requested and report['status']=='cancelled'
    elif report['status']=='planning_paused':assert report['worker_exit_code']==2
    else:assert report['execution_valid'] and report['worker_exit_code']==0
    state['active']=label+'-audit';save()
    wait_owned(['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src:scripts',
        'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',
        '.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python',
        'scripts/audit_fresh_target_handoff.py',linux(run),linux(checkpoint)],base/(label+'-audit.log'))
    audit=json.loads((run/'rm-plan-audit.json').read_text())
    assert audit['verified'];row['audit']=str(run/'rm-plan-audit.json');save()


try:
    trial('fixed-plan')
    trial('stop-in-plan-wait',True)
    for name,sha in frozen.items():assert digest(ROOT/name)==sha,name
    state.update(completed=True,sources_unchanged=True)
except BaseException as exc:state['error']=repr(exc)
finally:state['active']=None;save()
print(json.dumps(state))
raise SystemExit(0 if state['completed'] else 2)
