"""One-use local game with Windows broker; paid branch requires separate approval."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from run_rl_batch import ROOT,rearm_allowed


def verify_prepared(root):
    manifest=root/'results/planning-recovery-live-prepared-20261009/manifest.json'
    plan=json.loads(manifest.read_text(encoding='utf-8'))
    if plan.get('maximum_api_calls')!=1 or plan.get('api_retries')!=0 or plan.get('approved') is not False:
        raise ValueError('Invalid prepared one-call scope')
    for relative,expected in plan['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or digest(path)!=expected:
            raise ValueError('Prepared source/artifact changed: '+relative)
    return dict(path=str(manifest),sha256=digest(manifest))


def main():
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fixture',action='store_true');mode.add_argument('--allow-paid',action='store_true')
    a=p.parse_args()
    preparation=verify_prepared(ROOT) if a.allow_paid else None
    rearm_allowed()
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    world=read(ROOT/'results/goal-recovery-comparison-20261009/experiment.json')['worlds']['corner']
    checkpoint=ROOT/'results/onset-coverage-training-20261009/checkpoint.pt'
    assert digest(checkpoint)=='473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd'
    base=ROOT/('results/planning-recovery-live-20261009' if a.allow_paid else 'results/planning-recovery-mailbox-game-20261009')
    base.mkdir(exist_ok=False);exchange=base/'exchange'
    state=dict(completed=False,mode='live_api' if a.allow_paid else 'fixed_fixture',maximum_api_calls=1 if a.allow_paid else 0,
        api_retries=0,actual_api_calls=0,sampling_seed=9721,world=world,checkpoint_sha256=digest(checkpoint),
        run=None,error=None,preparation=preparation,performance_sample=False,
        scope='One bounded experimental recovery handoff; not final evaluation',
        sources={str(p.relative_to(ROOT)):digest(p) for p in (
            Path(__file__),ROOT/'scripts/run_recovery_broker.py',ROOT/'src/minecraftrl/recovery_live.py',
            ROOT/'src/minecraftrl/recovery_broker.py',ROOT/'src/minecraftrl/recovery_exchange.py',
            ROOT/'src/minecraftrl/planning_recovery.py',ROOT/'src/minecraftrl/replanning.py',
            ROOT/'src/minecraftrl/cli.py',ROOT/'src/minecraftrl/runtime.py',ROOT/'src/minecraftrl/goal_recovery.py',
            ROOT/'scripts/run_goal_recovery_pilot.py')})
    def save():write_json(base/'work.json',state)
    save()
    game_argv=[sys.executable,str(ROOT/'scripts/run_goal_recovery_pilot.py'),'--seed',str(world['seed']),
        '--sampling-seed','9721','--checkpoint',str(checkpoint),
        '--embed',str(ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'),
        '--recovery-embed',str(ROOT/'results/goal-conditioning-diagnosis-20261009/explore_9711.npy'),
        '--world',world['path'],'--planner-exchange',str(exchange)]
    broker_argv=[sys.executable,str(ROOT/'scripts/run_recovery_broker.py'),'--exchange',str(exchange)]
    if a.allow_paid:
        game_argv.append('--planner-exchange-live');broker_argv.append('--allow-paid')
    else:broker_argv+=['--fixture-choice','explore_then_resume','--fixture-delay','5']
    game=broker=None
    try:
        with (base/'game-launch.log').open('w',encoding='utf-8') as game_log,(base/'broker-console.log').open('w',encoding='utf-8') as broker_log:
            game=subprocess.Popen(game_argv,stdout=game_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            # The isolated launcher must consume only its automatic STOP before the broker starts.
            import time
            deadline=time.monotonic()+15
            while (ROOT/'results/STOP').exists():
                if game.poll() is not None or time.monotonic()>=deadline:raise RuntimeError('Game did not claim automatic STOP')
                time.sleep(.02)
            broker=subprocess.Popen(broker_argv,stdout=broker_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            state['game_launcher_pid']=game.pid;state['broker_pid']=broker.pid
            lines=(base/'game-launch.log').read_text(encoding='utf-8').splitlines()
            if lines:state['run']=str(Path(lines[0])/'run')
            save()
            game.wait(timeout=900)
            try:broker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                broker.terminate();broker.wait(timeout=10);state['broker_forced_termination']=True
        first=(base/'game-launch.log').read_text(encoding='utf-8').splitlines()[0]
        run=Path(first)/'run';state['run']=str(run)
        summary=read(run/'summary.json');state['summary']=summary
        assert summary['release']=='neutral_command_sent' and not summary['process_cleanup']['remaining_live_pids']
        if (exchange/'api-receipt.json').exists():
            state['api_receipt']=read(exchange/'api-receipt.json');state['actual_api_calls']=state['api_receipt']['network_calls']
        assert state['actual_api_calls']<=state['maximum_api_calls']
        if (exchange/'broker-result.json').exists():state['broker_result']=read(exchange/'broker-result.json')
        state.update(completed=True,game_launcher_exit=game.returncode,broker_exit=broker.returncode)
    except BaseException as exc:
        state['error']=repr(exc)
        if game is not None and game.poll() is None:
            from minecraftrl.rl_control import publish_completion
            publish_completion(ROOT/'results/STOP','RL collection ended; owned mailbox pilot error STOP.')
            try:game.wait(timeout=90)
            except subprocess.TimeoutExpired:state['game_cleanup_timeout']=True
        if broker is not None and broker.poll() is None:
            broker.terminate();broker.wait(timeout=10)
        raise
    finally:
        # Preserve call accounting even when game startup/finalization fails.
        if (exchange/'api-receipt.json').exists():
            state['api_receipt']=read(exchange/'api-receipt.json')
            state['actual_api_calls']=state['api_receipt']['network_calls']
        if (exchange/'broker-result.json').exists():state['broker_result']=read(exchange/'broker-result.json')
        if game is not None:state['game_launcher_exit']=game.poll()
        if broker is not None:state['broker_exit']=broker.poll()
        save()
    print(json.dumps(dict(completed=state['completed'],mode=state['mode'],actual_api_calls=state['actual_api_calls'],
        run=state['run'],status=state['summary']['status'],planner_calls=state['summary'].get('planner_calls'))))


if __name__=='__main__':main()
