"""Resume-aware bounded BC comparison, PPO and validation; never reruns completed games."""
import json,subprocess,sys,time
from pathlib import Path
from minecraftrl.companion import worker_command
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/navigation-refinement-20261014'
LINUX='/mnt/d/DATA/Desktop/MinecraftRL'
PRIOR='results/waypoint-training-goal-focused-20261010/checkpoint.pt'


def linux(script,*args):
    prefix=worker_command(LINUX,'Ubuntu-24.04','unused',104743,180,'unused')
    return prefix[:prefix.index('python')+1]+[script,*map(str,args)]


def main():
    ledger=BASE/'workflow.json';status=dict(completed=False,active=None,steps=[],error=None,api_calls=0)
    if ledger.exists():
        status=json.loads(ledger.read_text())
        if status['completed']:raise ValueError('workflow already completed; do not repeat')
    steps=BASE/'workflow-logs';steps.mkdir(exist_ok=True);proc=None
    def execute(label,argv):
        nonlocal proc
        log=steps/(label+'.log')
        if log.exists():raise FileExistsError('unfinished step log preserved; inspect before any retry: '+str(log))
        status['active']=label;write_json(ledger,status);print('START '+label,flush=True)
        with log.open('w',encoding='utf-8') as stream:
            proc=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            began=time.monotonic()
            while proc.poll() is None:
                if time.monotonic()-began>1100:raise TimeoutError(label)
                time.sleep(.2)
        if proc.returncode:raise RuntimeError(label+' exited '+str(proc.returncode)+'; see '+str(log))
        status['steps'].append(dict(label=label,elapsed_s=time.monotonic()-began,log_sha256=digest(log)))
        status['active']=None;write_json(ledger,status);print('DONE '+label,flush=True)
    def game(label,world,mode,actor,sampling):
        proof=BASE/label/'work.json'
        if proof.exists():assert json.loads(proof.read_text())['completed']
        else:execute(label,[sys.executable,str(ROOT/'scripts/run_navigation_refinement_trial.py'),label,
            '--world',world,'--mode',mode,'--actor',actor,'--sampling-seed',str(sampling)])
        check=BASE/label/'run/navigation-probability-audit.json'
        if not check.exists():execute(label+'-probability',linux('scripts/audit_navigation_refinement.py',str((BASE/label/'run').relative_to(ROOT)).replace('\\','/')))
        assert json.loads(check.read_text())['verified']
    try:
        for label in ('correction-train-104743','correction-train-104759','correction-development'):
            assert json.loads((BASE/label/'run/navigation-probability-audit.json').read_text())['verified']
        bc=BASE/'bc'
        if not (bc/'training.json').exists():
            execute('bc',linux('scripts/train_navigation_refinement.py','bc','results/navigation-refinement-20261014/bc',
                '--actor',PRIOR,'--train','results/navigation-refinement-20261014/correction-train-104743/run',
                'results/navigation-refinement-20261014/correction-train-104759/run',
                '--development','results/navigation-refinement-20261014/correction-development/run'))
        assert json.loads((bc/'training.json').read_text())['completed']
        actors=dict(prior=PRIOR,linear='results/navigation-refinement-20261014/bc/linear/checkpoint.pt',
            mlp='results/navigation-refinement-20261014/bc/mlp/checkpoint.pt')
        for kind,actor in actors.items():
            for world in ('far','corner'):game('evaluate-'+kind+'-'+world,'development-'+world,'evaluate',actor,14131)
        scores=[]
        for kind in ('linear','mlp'):
            reports=[json.loads((BASE/('evaluate-'+kind+'-'+w)/'run/summary.json').read_text()) for w in ('far','corner')]
            scores.append((sum(r['learned_arrivals'] for r in reports),-sum(r['mode_steps']['student'] for r in reports),kind=='linear',kind))
        chosen=max(scores)[-1]
        write_json(BASE/'ppo-selection.json',dict(kind=chosen,scores=scores,
            rule='most observed arrivals, then fewer student actions, then smaller linear head; development selection, no promotion'))
        for seed in (104743,104759):game('ppo-train-'+str(seed),'train-'+str(seed)+'-far','ppo',actors[chosen],14141)
        ppo=BASE/'ppo'
        if not (ppo/'training.json').exists():
            execute('ppo',linux('scripts/train_navigation_refinement.py','ppo','results/navigation-refinement-20261014/ppo',
                '--actor',actors[chosen],'--train','results/navigation-refinement-20261014/ppo-train-104743/run',
                'results/navigation-refinement-20261014/ppo-train-104759/run'))
        assert json.loads((ppo/'training.json').read_text())['completed']
        actor='results/navigation-refinement-20261014/ppo/'+chosen+'/checkpoint.pt'
        for world in ('far','corner'):game('evaluate-ppo-'+world,'development-'+world,'evaluate',actor,14131)
        execute('report',[sys.executable,str(ROOT/'scripts/report_navigation_refinement.py')])
        status.update(completed=True,active=None)
    except BaseException as exc:
        status['error']=repr(exc)
        if proc and proc.poll() is None:
            with (ROOT/'results/STOP').open('a',encoding='utf-8') as f:f.write('\nNavigation workflow interrupted; inspect before rearming.')
            try:proc.wait(timeout=110)
            except subprocess.TimeoutExpired:status['error']+='; owned step still active'
    finally:write_json(ledger,status)
    print(json.dumps(status));return 0 if status['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
