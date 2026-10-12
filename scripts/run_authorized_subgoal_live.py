"""One new live RM plan, requiring approval bound to a prepared manifest."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--authorization',type=Path,required=True);p.add_argument('--allow-paid',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];prepared=a.prepared.resolve()
    if not a.allow_paid:raise SystemExit('Separate one-call approval required; no game or network started.')
    if not prepared.is_relative_to(root/'results'):raise ValueError('project-local preparation required')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(prepared/'manifest.json');approval=read(a.authorization)
    if (approval.get('approved') is not True or approval.get('maximum_calls')!=1
            or approval.get('manifest_sha256')!=digest(prepared/'manifest.json')):
        raise ValueError('approval must bind this exact one-call preparation')
    for name,sha in manifest['source_sha256'].items():
        if digest(root/name)!=sha:raise ValueError('prepared source/input changed: '+name)
    if (manifest['maximum_calls']!=1 or manifest['maximum_games']!=1 or manifest['control_limit_s']!=180
            or manifest['sampling_seed']!=9821 or manifest['world_seed']!=104761):raise ValueError('scope changed')
    base=prepared/'live-run';base.mkdir(exist_ok=False);exchange=base/'exchange'
    argv=[sys.executable,str(root/'scripts/run_goal_recovery_pilot.py'),'--seed','104761',
        '--sampling-seed','9821','--limit-s','180','--rm-plan','--planner-max-requests','1',
        '--planner-exchange',str(exchange),'--planner-exchange-live']
    for k,v in manifest['assets'].items():argv+=['--'+k.replace('_','-'),str(root/v)]
    state=dict(completed=False,error=None,run=None,broker_started=False,manifest_sha256=digest(prepared/'manifest.json'))
    game=broker=None;started=time.monotonic()
    with (base/'game.log').open('w',encoding='utf-8') as game_log,(base/'broker.log').open('w',encoding='utf-8') as broker_log:
        try:
            game=subprocess.Popen(argv,stdout=game_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            state['owned_game_launcher_pid']=game.pid;write_json(base/'work.json',state)
            while game.poll() is None:
                if not broker and (exchange/'request-001/request.json').exists() and not (root/'results/STOP').exists():
                    broker=subprocess.Popen([sys.executable,str(root/'scripts/run_recovery_broker.py'),
                        '--exchange',str(exchange/'request-001'),'--rm-plan','--allow-paid'],stdout=broker_log,
                        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                    state.update(broker_started=True,owned_broker_pid=broker.pid);write_json(base/'work.json',state)
                if time.monotonic()-started>970:raise TimeoutError('live validation timeout')
                time.sleep(.05)
            if broker:broker.wait(timeout=25)
            lines=(base/'game.log').read_text(encoding='utf-8').splitlines();run=Path(lines[0])/'run'
            state['run']=str(run);report=read(run/'summary.json')
            receipt=exchange/'request-001/api-receipt.json'
            state.update(status=report['status'],api_calls=read(receipt)['network_calls'] if receipt.exists() else 0)
            assert state['api_calls']<=1 and report['rl_updates']==0
            assert not report['process_cleanup']['remaining_live_pids']
            assert report['release']=='neutral_command_sent' and 'error' not in report
            state['completed']=True
        except BaseException as exc:state['error']=repr(exc)
        finally:
            if game and game.poll() is None:
                (root/'results/STOP').write_text('Live subgoal validation interrupted; inspect before rearming.')
                try:game.wait(timeout=100)
                except subprocess.TimeoutExpired:state['error']='owned launcher did not finish after STOP'
            if broker and broker.poll() is None:
                broker.terminate();broker.wait(timeout=10)
            write_json(base/'work.json',state)
    print(json.dumps(state));return 0 if state['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
