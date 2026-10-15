"""One bounded correction/BC/development batch; never rerun an existing ledger."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/near-navigation-20261015'
PRIOR='results/navigation-refinement-20261014/ppo/mlp/checkpoint.pt'


def linux(script,*args):
    return ['wsl.exe','-d','Ubuntu-24.04','--cd','/mnt/d/DATA/Desktop/MinecraftRL','--','env',
        'PYTHONPATH=src','LD_LIBRARY_PATH=/usr/lib/wsl/lib','/usr/bin/timeout','--kill-after=15','900',
        '/mnt/d/DATA/Desktop/MinecraftRL/.runtime/steve1/bin/python','-u',script,*map(str,args)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--low-lr',action='store_true');args=parser.parse_args()
    base=ROOT/'results/near-navigation-low-lr-20261015' if args.low_lr else BASE
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    assert read(ROOT/'results/exploration-coverage-20261015/workflow.json')['completed']
    assert read(ROOT/'results/exploration-motion-20261015/workflow.json')['completed']
    assert read(ROOT/'results/rm-refined-navigation-20261015/final-verification.json')['verified']
    if args.low_lr:
        previous=read(BASE/'verification.json')
        assert previous['verified'] and not previous['training']['models'][0]['actor_weights_changed']
    ledger=base/'workflow.json'
    if ledger.exists():raise FileExistsError('inspect preserved batch; never repeat automatically')
    logs=base/'workflow-logs';logs.mkdir(parents=True,exist_ok=False)
    status=dict(completed=False,active=None,steps=[],error=None,api_calls=0);proc=None
    def execute(label,argv):
        nonlocal proc
        status['active']=label;write_json(ledger,status);print('START '+label,flush=True)
        started=time.monotonic();log=logs/(label+'.log')
        with log.open('x',encoding='utf-8') as f:
            proc=subprocess.Popen(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW)
            proc.wait(timeout=1100)
        if proc.returncode:raise RuntimeError(label+' exited '+str(proc.returncode))
        status['steps'].append(dict(label=label,elapsed_s=time.monotonic()-started,log_sha256=digest(log)))
        write_json(ledger,status);print('DONE '+label,flush=True)
    def game(label,world,mode,actor,sampling):
        execute(label,[sys.executable,'scripts/run_navigation_refinement_trial.py',label,'--near-low-lr' if args.low_lr else '--near',
            '--world',world,'--mode',mode,'--actor',actor,'--sampling-seed',str(sampling)])
        run=(base/label/'run').relative_to(ROOT).as_posix()
        execute(label+'-audit',linux('scripts/audit_navigation_refinement.py',run))
        assert read(ROOT/run/'navigation-probability-audit.json')['verified']
    try:
        if not args.low_lr:
            for seed in (104743,104759):game('correction-'+str(seed),'train-'+str(seed)+'-far','correction',PRIOR,14111)
        extra=['--learning-rate','0.00002'] if args.low_lr else []
        execute('bc',linux('scripts/train_navigation_refinement.py','bc_continue',(base/'bc').relative_to(ROOT).as_posix(),
            '--actor',PRIOR,'--train',
            'results/navigation-refinement-20261014/correction-train-104743/run',
            'results/navigation-refinement-20261014/correction-train-104759/run',
            'results/near-navigation-20261015/correction-104743/run',
            'results/near-navigation-20261015/correction-104759/run',
            '--development','results/navigation-refinement-20261014/correction-development/run',*extra))
        training=read(base/'bc/training.json');assert training['completed'] and training['root_unchanged']
        if not training['models'][0]['actor_weights_changed']:
            status.update(completed=True,active=None,development=None,rm_followup_run=False,
                unrun_reason='KL guard retained no changed actor; no duplicate evaluations')
            print(json.dumps(status));return 0
        actor=(base/'bc/mlp/checkpoint.pt').relative_to(ROOT).as_posix()
        for world in ('far','corner'):game('evaluate-'+world,'development-'+world,'evaluate',actor,14131)
        reports=[read(base/('evaluate-'+w)/'run/summary.json') for w in ('far','corner')]
        score=dict(arrivals=sum(r['learned_arrivals'] for r in reports),actions=sum(r['mode_steps']['student'] for r in reports))
        improved=score['arrivals']>10 or (score['arrivals']==10 and score['actions']<649)
        write_json(base/'development-selection.json',dict(baseline=dict(arrivals=10,actions=649),candidate=score,
            improved=improved,scope='Reused development tasks; not final evaluation or default promotion'))
        if improved:
            execute('rm-far',[sys.executable,'scripts/run_rm_navigation_validation.py','--revision','3','--near-low-lr' if args.low_lr else '--near-candidate'])
            execute('rm-far-audit',[sys.executable,'scripts/run_saved_rm_audit.py',
                (base/'game-far/run').relative_to(ROOT).as_posix(),base.relative_to(ROOT).as_posix(),'--label','rm-far-audit-detail'])
        status.update(completed=True,active=None,development=score,rm_followup_run=improved)
    except BaseException as exc:
        status['error']=repr(exc)
        if proc and proc.poll() is None:
            with (ROOT/'results/STOP').open('a',encoding='utf-8') as f:f.write('\nNear navigation workflow interrupted; inspect before rearming.')
            try:proc.wait(timeout=110)
            except subprocess.TimeoutExpired:status['error']+='; owned child still active'
    finally:write_json(ledger,status)
    print(json.dumps(status));return 0 if status['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
