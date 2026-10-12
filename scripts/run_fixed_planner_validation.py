"""Two one-use local integration trials; zero API calls, sequential GPU use."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from run_rl_batch import ROOT,rearm_allowed


def main():
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group()
    mode.add_argument('--after-no-stall',action='store_true');mode.add_argument('--after-key-count-fix',action='store_true');args=p.parse_args()
    followup=args.after_no_stall or args.after_key_count_fix
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    previous=read(ROOT/'results/goal-recovery-comparison-20261009/experiment.json')
    assert previous['completed'] and not previous['errors']
    rearm_allowed()
    if args.after_no_stall:
        old=read(ROOT/'results/planning-recovery-validation-20261009/work.json')
        first=old['runs'][0]['summary']
        assert len(old['runs'])==1 and first['status']=='target_reached' and first['planner_calls']==0
        assert first['neutral_control_transitions']==0 and first['steps']==430
    if args.after_key_count_fix:
        old=read(ROOT/'results/planning-recovery-validation-20261009-handoff/work.json')
        first=old['runs'][0]['summary']
        assert len(old['runs'])==1 and first['error']=='ValueError: invalid release evidence'
        assert first['planner_calls']==0 and first['neutral_control_transitions']==1
    suffix='-key-count-fix' if args.after_key_count_fix else '-handoff' if args.after_no_stall else ''
    base=ROOT/('results/planning-recovery-validation-20261009'+suffix);base.mkdir(exist_ok=False)
    stop=ROOT/'results/STOP';world=previous['worlds']['corner']
    checkpoint=ROOT/'results/onset-coverage-training-20261009/checkpoint.pt'
    state=dict(completed=False,actual_api_calls=0,optimizer_updates=0,active=None,runs=[],errors=[],
        world=world,checkpoint_sha256=digest(checkpoint),
        sources={str(p.relative_to(ROOT)):digest(p) for p in (
            Path(__file__),ROOT/'src/minecraftrl/planning_recovery.py',ROOT/'src/minecraftrl/runtime.py',
            ROOT/'src/minecraftrl/replanning.py',ROOT/'src/minecraftrl/goal_recovery.py',ROOT/'src/minecraftrl/cli.py',
            ROOT/'scripts/run_goal_recovery_pilot.py',ROOT/'docs/planning-recovery-protocol-2026-10-09.md')})
    def save():write_json(base/'work.json',state)
    if followup:
        state['sources']['docs/planning-recovery-protocol-amendment-2026-10-09.md']=digest(ROOT/'docs/planning-recovery-protocol-amendment-2026-10-09.md')
        state['prior_technical_error_run' if args.after_key_count_fix else 'prior_unexercised_run']=old['runs'][0]['run']
    save()
    cases=[('fixed-explore',9721,5,False),('stop-in-wait',9721,30,True)] if followup else [
        ('fixed-explore',9731,5,False),('stop-in-wait',9732,30,True)]
    for label,sampling,delay,cancel in cases:
        rearm_allowed()
        argv=[sys.executable,str(ROOT/'scripts/run_goal_recovery_pilot.py'),'--seed',str(world['seed']),
            '--sampling-seed',str(sampling),'--checkpoint',str(checkpoint),
            '--embed',str(ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'),
            '--recovery-embed',str(ROOT/'results/goal-conditioning-diagnosis-20261009/explore_9711.npy'),
            '--world',world['path'],'--planner-choice','explore_then_resume','--planner-delay',str(delay)]
        row=dict(label=label,sampling_seed=sampling,response_delay_s=delay,performance_sample=False,
                 scope='Runtime integration diagnostic, not LLM planning performance',run=None,cancel_requested=False)
        state['runs'].append(row);state['active']=label;save()
        logpath=base/(label+'-launch.log');process=None
        try:
            with logpath.open('w',encoding='utf-8') as log:
                process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                deadline=time.monotonic()+900;offset=0;pending=b'';run=None;cancel_at=None
                while process.poll() is None:
                    if time.monotonic()>=deadline:raise TimeoutError('owned trial deadline exceeded')
                    if run is None:
                        with logpath.open(encoding='utf-8') as source:first=source.readline().strip()
                        if first:
                            run=Path(first)/'run';row['run']=str(run);save();print(label,str(run),flush=True)
                    if cancel and run and (run/'events.jsonl').exists() and not row['cancel_requested']:
                        with (run/'events.jsonl').open('rb') as source:
                            source.seek(offset);block=source.read(2**21);offset+=len(block)
                        lines=(pending+block).split(b'\n');pending=lines.pop()
                        for line in lines:
                            event=json.loads(line)
                            if event['event']=='planner_wait_started' and cancel_at is None:
                                cancel_at=time.monotonic()+1.
                                row['wait_event']=event;save()
                        if cancel_at is not None and time.monotonic()>=cancel_at:
                            with stop.open('x',encoding='utf-8') as stream:
                                stream.write('RL collection ended; owned planning-wait STOP audit 20261009 request.')
                            row.update(cancel_requested=True,stop_requested_windows_monotonic_s=time.monotonic())
                            save();print('STOP requested during planning wait',flush=True)
                    time.sleep(.05)
            assert run is not None
            report=read(run/'summary.json');row['summary']=report;row['launcher_exit']=process.returncode;save()
            assert report['release']=='neutral_command_sent' and not report['process_cleanup']['remaining_live_pids']
            assert report.get('actual_llm_calls')==0 and report.get('rl_updates')==0
            if report['planner_calls']==0 and report['execution_valid']:
                row.update(completed=True,handoff_exercised=False,classification='no_planning_request_observed')
                state['active']=None;save();continue
            if cancel:
                assert row['cancel_requested'] and report['status']=='cancelled'
            else:
                assert report['execution_valid'] and report['worker_exit_code']==0,report.get('error',report['status'])
                assert report['neutral_control_transitions']==2 and report['planner_calls']==1
                assert report['intervention_committed'], 'No actual accepted recovery handoff'
            row['completed']=True;row['handoff_exercised']=True;state['active']=None;save();print(label,'completed',report['status'],flush=True)
        except BaseException as exc:
            row['error']=repr(exc);state['errors'].append(dict(label=label,error=repr(exc)));save()
            if process is not None and process.poll() is None:
                try:
                    with stop.open('x',encoding='utf-8') as stream:stream.write('RL collection ended; owned planning validation error STOP.')
                except FileExistsError:pass
                try:process.wait(timeout=90)
                except subprocess.TimeoutExpired:row['cleanup_timeout']=True
            state['active']=None;save();raise
    state['completed']=True;save()
    print(json.dumps(dict(completed=True,trials=len(state['runs']),actual_api_calls=0)),flush=True)


if __name__=='__main__':main()
