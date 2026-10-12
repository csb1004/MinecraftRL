"""Fixed, paired before/after evaluation. No evaluation trajectory is trained on."""
import argparse,datetime,json,subprocess,sys
from pathlib import Path
from minecraftrl.world_store import digest
from minecraftrl.runtime import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--embed',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1];stop=root/'results/STOP'
    seeds=[1000003,1000033,1000037,1000039,1000081]
    base=root/'results'/('rl-evaluation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));base.mkdir()
    checkpoint=args.checkpoint.resolve();embedding=args.embed.resolve();before=digest(checkpoint)
    state=dict(completed=False,seeds=seeds,planned=10,limit_s=180,checkpoint=str(checkpoint),checkpoint_sha256=before,
               embedding=str(embedding),embedding_sha256=digest(embedding),reports=[],errors=[])
    def save():write_json(base/'evaluation.json',state)
    save();print(base,flush=True)
    try:
        for seed in seeds:
            for phase in ('before','after'):
                if not stop.exists() or not stop.read_text(encoding='utf-8').startswith(('RL training ended','RL collection ended')):
                    raise RuntimeError('unexpected STOP state; refusing automatic re-arm')
                argv=[sys.executable,str(root/'scripts/run_rl_pilot.py'),'--seed',str(seed),'--split','eval','--embed',str(embedding)]
                if phase=='after':argv+=['--checkpoint',str(checkpoint)]
                logpath=base/(str(seed)+'-'+phase+'.log');print(json.dumps(dict(seed=seed,phase=phase,starting=True)),flush=True)
                with logpath.open('w',encoding='utf-8') as log:
                    proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    try:proc.wait(timeout=960)
                    finally:
                        if proc.poll() is None:stop.write_text('evaluation timeout');proc.wait(timeout=100)
                lines=logpath.read_text(encoding='utf-8').splitlines();trial=Path(lines[0]);r=json.loads((trial/'run/summary.json').read_text(encoding='utf-8'))
                state['reports'].append(dict(seed=seed,phase=phase,path=str(trial),report=r));save()
                if proc.returncode or not r['execution_valid']:raise RuntimeError('evaluation technical failure: '+str(trial))
                assert r['release']=='neutral_command_sent' and not r['process_cleanup']['remaining_live_pids']
                assert digest(checkpoint)==before,'checkpoint changed during evaluation'
                print(json.dumps(dict(seed=seed,phase=phase,status=r['status'],success=r['server_target_reached'],seconds=r['elapsed_control_s'])),flush=True)
        state['completed']=True
    except Exception as exc:state['errors'].append(repr(exc))
    finally:state['unrun']=10-len(state['reports']);save()
    return 0 if state['completed'] else 2

if __name__=='__main__':raise SystemExit(main())
