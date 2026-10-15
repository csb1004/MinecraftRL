"""One-use development comparison; bounded Windows candidate brokers, no motor scripts."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from minecraftrl.companion import worker_command
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=['far','corner'])
    p.add_argument('planner',choices=['visit_memory','adaptive_rule','adaptive_llm']);p.add_argument('--attempt',type=int,choices=[1,2],default=1);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];batch=root/'results/adaptive-recovery-20261014'
    index_path=batch/'completed-trials.json'
    completed=json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
    trial_key=a.case+'/'+a.planner
    if trial_key in completed:raise ValueError('completed game cannot be rerun')
    if a.attempt==2:
        old=json.loads((batch/trial_key/'run/summary.json').read_text(encoding='utf-8'))
        if old['status']!='blocked' or old['steps']!=0:raise ValueError('retry allowed only for preflight failure before gameplay')
    out=batch/a.case/(a.planner if a.attempt==1 else a.planner+'-attempt2');out.mkdir(parents=True,exist_ok=False)
    stop=root/'results/STOP';linux='/mnt/d/DATA/Desktop/MinecraftRL';seed=14021 if a.case=='far' else 14022
    paths=dict(checkpoint='results/onset-coverage-training-20261009/checkpoint.pt',embed='results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy',
        world='results/blind-recovery-20261012/worlds/'+a.case+'.zip',waypoint_actor='results/waypoint-training-goal-focused-20261010/checkpoint.pt')
    expected=dict(checkpoint='473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd',
        embed='810512d05a7f228d77aed45f29a21277156f934408e2387f2ae3fd6f4217be40',
        waypoint_actor='f1dee8013938deca68d87b6eda2c7db1bffa511eaf23528745bbc60bec2c282a',
        world={'far':'5239e0ac8e769f75cbc064fd5dc0ecf42da07422b353d899dfd2f74478d5b798','corner':'cbd7890bc992accf4093d6d156f7187f0b03a0af484a40e822ec467f751698a3'}[a.case])
    for key,path in paths.items():assert digest(root/path)==expected[key],key
    frozen={path:expected[key] for key,path in paths.items()}
    for folder in ('src/minecraftrl','scripts'):
        for path in (root/folder).glob('*.py'):
            dest=out/'source-at-game'/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    argv=worker_command(linux,'Ubuntu-24.04','unused',104761,180,(out/'run').relative_to(root).as_posix())
    index=argv.index('--world-slot');del argv[index:index+2];argv[argv.index('session')]='episode'
    argv+=['--condition','wood','--rl-collect','--rl-split','eval','--rl-reward-mode','wood-potential-v4','--rl-sampling-seed',str(seed),
        '--rl-waypoint-start','stall','--rl-waypoint-planner',a.planner]
    for name,path in paths.items():argv+=['--rl-'+name.replace('_','-'),linux+'/'+path]
    result=dict(completed=False,error=None,case=a.case,planner=a.planner,sampling_seed=seed,api_calls=0,optimizer_updates=0,frozen_assets=frozen)
    game=None;brokers=[];seen=set();claim_automatic_stop(stop,out/'STOP-before.txt')
    try:
        with (out/'launcher.log').open('w',encoding='utf-8') as log:
            game=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            started=time.monotonic()
            while game.poll() is None:
                for request in sorted((out/'run/candidate-exchange').glob('request-*/request.json')):
                    if request in seen:continue
                    if a.planner!='adaptive_llm' or len(seen)>=2:raise RuntimeError('unexpected paid request')
                    seen.add(request)
                    # Explicit experiment authorization is validated again inside every broker.
                    cmd=[sys.executable,str(root/'scripts/run_navigation_choice_broker.py'),'--exchange',str(request.parent),'--allow-paid']
                    brokers.append(subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW))
                if time.monotonic()-started>970:raise TimeoutError('adaptive trial timeout')
                time.sleep(.05)
            for broker in brokers:broker.wait(timeout=25)
        summary=json.loads((out/'run/summary.json').read_text(encoding='utf-8'));assert 'error' not in summary,summary.get('error')
        assert game.returncode==0 or (game.returncode==2 and summary['status']=='planning_paused')
        assert summary['release']=='neutral_command_sent' and not summary['process_cleanup']['remaining_live_pids']
        for path,sha in frozen.items():assert digest(root/path)==sha,path
        receipts=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'run/candidate-exchange').glob('request-*/api-receipt.json')]
        calls=sum(r['network_calls'] for r in receipts);assert calls<=2 and calls==summary['actual_llm_calls']
        result.update(completed=True,status=summary['status'],steps=summary['steps'],recovery_called=summary['mode_steps']['waypoint']>0,
            route_receipts=summary['route_receipts'],root_success=summary['root_inventory_success'],api_calls=calls,
            reserved_usd=sum(r.get('reserved_usd',0) for r in receipts),estimated_usd=sum(r.get('estimated_usd') or 0 for r in receipts))
        from minecraftrl.memory import atomic_json
        completed[trial_key]=(out/'run').relative_to(batch).as_posix();atomic_json(index_path,completed)
    except BaseException as exc:result['error']=repr(exc)
    finally:
        if game and game.poll() is None:
            stop.write_text('Adaptive validation interrupted; inspect before rearming.',encoding='utf-8')
            try:game.wait(timeout=100)
            except subprocess.TimeoutExpired:result['error']='owned game did not finish after STOP'
        for broker in brokers:
            if broker.poll() is None:broker.terminate();broker.wait(timeout=10)
        if game is None or game.poll() is not None:publish_completion(stop,'RL collection ended; adaptive recovery trial finished.')
        write_json(out/'work.json',result)
    print(json.dumps(result));return 0 if result['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
