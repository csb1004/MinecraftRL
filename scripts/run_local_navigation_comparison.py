"""One bounded, user-authorized development game; preserve each attempt and STOP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('label');p.add_argument('--strategy',choices=['coverage_motion','local_passability'],required=True)
    p.add_argument('--sampling-seed',type=int,default=14031)
    p.add_argument('--world',type=Path)
    p.add_argument('--output-root',type=Path,default=Path('results/local-navigation-detour-20261019'))
    p.add_argument('--diagnostic-initial-goal-wait',type=int,choices=[32],help='Targeted integration probe only: shorten initial goal-stall wait in the isolated source copy')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if not a.label.replace('-','').isalnum():raise ValueError('simple attempt label required')
    out=(root/a.output_root).resolve()
    if not out.is_relative_to(root/'results'):raise ValueError('output must stay inside project results')
    out.mkdir(exist_ok=True)
    attempt=out/a.label;attempt.mkdir(exist_ok=False)
    old=json.loads((root/'results/local-wall-diagnosis-20261017/protocol.json').read_text(encoding='utf-8'))
    for path,sha in old['frozen_assets'].items():
        with Path(path).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=sha:raise ValueError('frozen input changed: '+path)
    source=attempt/'execution-source';shutil.copytree(root/'src',source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if a.diagnostic_initial_goal_wait is not None:
        if a.strategy!='local_passability' or a.world is None:raise ValueError('diagnostic requires explicit local-route fixture')
        module=source/'minecraftrl/goal_recovery.py';text=module.read_text(encoding='utf-8')
        marker='self.detector = StallDetector()';metadata='direct_action_override=False))'
        assert text.count(marker)==text.count(metadata)==1
        text=text.replace(marker,marker+'\n        self.detector.goal_budget_ticks=32')
        text=text.replace(metadata,metadata+"\n        self.condition_metadata['intervention']['diagnostic_initial_goal_budget_ticks']=32")
        module.write_text(text,encoding='utf-8')
    linux='/mnt/c/MinecraftRL';rel=attempt.relative_to(root).as_posix()
    argv=list(old['argv'])
    argv[argv.index('--output')+1]=linux+'/'+rel+'/run'
    argv[argv.index('--rl-navigation-strategy')+1]=a.strategy
    argv[argv.index('--rl-sampling-seed')+1]=str(a.sampling_seed)
    for i,v in enumerate(argv):
        if v.startswith('PYTHONPATH='):argv[i]='PYTHONPATH='+linux+'/'+rel+'/execution-source'
    if a.world:
        world=a.world.resolve();assert world.is_relative_to(root/'results')
        argv[argv.index('--rl-world')+1]=linux+'/'+world.relative_to(root).as_posix()
    protocol=dict(scope='Development comparison; no training or final-test claim',strategy=a.strategy,
        sampling_seed=a.sampling_seed,argv=argv,frozen_assets=old['frozen_assets'],control_limit_s=180,
        observation='Both conditions record bounded local passability; only local_passability consumes it',
        motor='Frozen neural actors; no scripted movement/mining',paid_api_calls=0,optimizer_updates=0,
        evaluation='Actual server log gain; waypoint and routed approach activation reported separately',
        source_sha256={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()})
    if a.diagnostic_initial_goal_wait is not None:
        protocol.update(scope='Targeted real-game recovery integration probe; NOT default-policy performance or training success',
            diagnostic_initial_goal_budget_ticks=32,original_initial_goal_budget_ticks=800,
            diagnostic_source_change='Isolated goal_recovery.py initial detector instance only; production source, resumed detector, actors, actions and rewards unchanged')
    (attempt/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    stop=root/'results/STOP'
    if stop.exists():
        contents=stop.read_text(encoding='utf-8')
        if not (contents.startswith('Local wall diagnosis ended;') or contents.startswith('Local navigation comparison ended;')):
            raise ValueError('New/user STOP requires inspection before rearming')
        stop.rename(attempt/'STOP-before.txt')
    result=dict(completed=False,error=None);proc=None
    try:
        with (attempt/'launcher.log').open('x',encoding='utf-8') as log:
            proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            (attempt/'launcher-pid.json').write_text(json.dumps(dict(pid=proc.pid)))
            try:proc.wait(timeout=970)
            except subprocess.TimeoutExpired:
                with stop.open('a',encoding='utf-8') as stream:stream.write('Local navigation startup/runtime deadline; STOP.\n')
                proc.wait(timeout=100)
                raise TimeoutError('comparison launcher limit')
            result.update(completed=True,exit_code=proc.returncode)
    except BaseException as exc:result['error']=repr(exc)
    finally:
        try:
            with stop.open('x',encoding='utf-8') as stream:stream.write('Local navigation comparison ended; inspect '+out.relative_to(root).as_posix()+' before re-arming.\n')
        except FileExistsError:pass
        result['stop_latched']=stop.exists()
        (attempt/'launcher-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result));return 0 if result['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
