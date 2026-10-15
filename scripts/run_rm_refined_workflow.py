"""One bounded, sequential RM integration batch, with no LLM/API calls."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from minecraftrl.companion import worker_command
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/rm-refined-navigation-20261015'


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--coverage',action='store_true');group.add_argument('--motion',action='store_true');args=parser.parse_args()
    base=(ROOT/'results/exploration-motion-20261015' if args.motion else
          ROOT/'results/exploration-coverage-20261015' if args.coverage else BASE)
    ledger=base/'workflow.json'
    if ledger.exists():raise FileExistsError('existing batch preserved; inspect instead of repeating')
    status=dict(completed=False,active=None,steps=[],error=None,api_calls=0);proc=None
    try:
        for world in (('corner',) if args.motion else ('far','corner')):
            prefix=worker_command('/mnt/d/DATA/Desktop/MinecraftRL','Ubuntu-24.04','unused',104761,180,'unused')
            audit=prefix[:prefix.index('python')+1]+['scripts/audit_fresh_target_handoff.py',
                'results/rm-refined-navigation-20261015/game-'+world+'/run',
                'results/onset-coverage-training-20261009/checkpoint.pt']
            game=[sys.executable,'scripts/run_rm_navigation_validation.py','--revision','3','--world',world]
            if args.coverage or args.motion:
                game+=['--strategy','coverage_motion' if args.motion else 'coverage']
                audit=[sys.executable,'scripts/run_saved_rm_audit.py',
                    str((base/('game-'+world)/'run').relative_to(ROOT)),str(base.relative_to(ROOT)),
                    '--label','audit-'+world+'-detail']
            commands=[('game-'+world,game),
                ('audit-'+world,audit)]
            for label,argv in commands:
                status['active']=label;write_json(ledger,status);print('START '+label,flush=True)
                log=base/(label+'.log');started=time.monotonic()
                with log.open('x',encoding='utf-8') as stream:
                    proc=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                    proc.wait(timeout=1100)
                if proc.returncode:raise RuntimeError(label+' exited '+str(proc.returncode))
                status['steps'].append(dict(label=label,elapsed_s=time.monotonic()-started,log_sha256=digest(log)))
                print('DONE '+label,flush=True)
        status.update(completed=True,active=None)
    except BaseException as exc:
        status['error']=repr(exc)
        if proc and proc.poll() is None:
            with (ROOT/'results/STOP').open('a') as f:f.write('\nRM refined workflow interrupted; inspect before rearming.')
            try:proc.wait(timeout=110)
            except subprocess.TimeoutExpired:status['error']+='; owned child still active'
    finally:write_json(ledger,status)
    print(json.dumps(status));return 0 if status['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
