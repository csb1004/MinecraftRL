"""One local development game, fixed plan provider, no network/API calls."""
import argparse,json,subprocess,time
from pathlib import Path
from minecraftrl.companion import worker_command
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--world',choices=['far','corner'],default='far')
    parser.add_argument('--strategy',choices=['visited','coverage','coverage_motion'],default='visited')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--near-candidate',action='store_true')
    group.add_argument('--near-low-lr',action='store_true');args=parser.parse_args()
    near=args.near_candidate or args.near_low_lr
    if near and (args.revision!=3 or args.world!='far' or args.strategy!='visited'):
        parser.error('near candidate permits one separate far visited-strategy trial')
    if args.strategy!='visited' and args.revision!=3:parser.error('coverage requires explicit refined integration revision')
    if args.revision!=3 and args.world!='far':parser.error('older revisions are preserved far-only protocols')
    root=Path(__file__).resolve().parents[1];batch=root/'results/rm-navigation-integration-20261014'
    if args.revision==3:batch=root/'results/rm-refined-navigation-20261015'
    if args.strategy=='coverage':batch=root/'results/exploration-coverage-20261015'
    if args.strategy=='coverage_motion':batch=root/'results/exploration-motion-20261015'
    if near:
        batch=root/('results/near-navigation-low-lr-20261015' if args.near_low_lr else 'results/near-navigation-20261015')
        selection=json.loads((batch/'development-selection.json').read_text())
        assert selection['improved'] and selection['baseline']==dict(arrivals=10,actions=649)
    if args.revision==2:
        previous=json.loads((batch/'game-v1/work.json').read_text())
        events=json.loads((batch/'game-v1/run/planner-events.json').read_text())
        assert previous['completed'] and any(e.get('reason')=='approach_requires_known_target' for e in events)
    out=batch/(('game-'+args.world) if args.revision==3 else ('game-v'+str(args.revision)))
    out.mkdir(parents=True,exist_ok=False)
    stop=root/'results/STOP';linux='/mnt/d/DATA/Desktop/MinecraftRL'
    paths=dict(checkpoint='results/onset-coverage-training-20261009/checkpoint.pt',
        embed='results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy',
        world='results/blind-recovery-20261012/worlds/far.zip',
        rm_navigation_actor='results/waypoint-training-goal-focused-20261010/checkpoint.pt',
        recovery_embed='results/goal-conditioning-diagnosis-20261009/explore_9711.npy',
        skill_catalogue='results/subgoal-skills-20261010/embeddings/catalogue.json')
    expected=dict(checkpoint='473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd',
        embed='810512d05a7f228d77aed45f29a21277156f934408e2387f2ae3fd6f4217be40',
        rm_navigation_actor='f1dee8013938deca68d87b6eda2c7db1bffa511eaf23528745bbc60bec2c282a',
        world='5239e0ac8e769f75cbc064fd5dc0ecf42da07422b353d899dfd2f74478d5b798')
    if args.revision==3:
        paths['rm_navigation_actor']='results/navigation-refinement-20261014/ppo/mlp/checkpoint.pt'
        expected['rm_navigation_actor']='e033f1ef7902102569d01459d85cdd80252c141f4e1346bdb40832502e8acaea'
        paths['world']='results/blind-recovery-20261012/worlds/'+args.world+'.zip'
        if args.world=='corner':expected['world']='cbd7890bc992accf4093d6d156f7187f0b03a0af484a40e822ec467f751698a3'
    if near:
        paths['rm_navigation_actor']=(batch/'bc/mlp/checkpoint.pt').relative_to(root).as_posix()
        training=json.loads((batch/'bc/training.json').read_text())
        assert training['completed'] and not training['error'] and training['method']=='bc_continue'
        expected['rm_navigation_actor']=training['models'][0]['checkpoint_sha256']
    frozen={path:digest(root/path) for path in paths.values()}
    frozen['docs/research-spec.md']=digest(root/'docs/research-spec.md')
    assert frozen['docs/research-spec.md']=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
    for key,sha in expected.items():assert frozen[paths[key]]==sha,key
    for folder in ('src/minecraftrl','scripts'):
        for path in (root/folder).glob('*.py'):
            dest=out/'source-at-game'/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    argv=worker_command(linux,'Ubuntu-24.04','unused',104761,180,(out/'run').relative_to(root).as_posix())
    index=argv.index('--world-slot');del argv[index:index+2];argv[argv.index('session')]='episode'
    maximum_requests=3 if args.strategy=='coverage_motion' else 2
    argv+=['--condition','wood','--rl-collect','--rl-split','eval','--rl-reward-mode','wood-potential-v4',
        '--rl-sampling-seed','14031','--rl-rm-plan-fixture','subgoals','--rl-planner-max-requests',str(maximum_requests),'--rl-planner-delay','1']
    for name,path in paths.items():argv+=['--rl-'+name.replace('_','-'),linux+'/'+path]
    argv+=['--rl-navigation-strategy',args.strategy]
    result=dict(completed=False,error=None,api_calls=0,optimizer_updates=0,frozen_assets=frozen,
        scope='Fixed response integration, development '+args.world+' arena; not LLM effectiveness')
    write_json(out/'protocol.json',dict(world_seed=104761,sampling_seed=14031,control_limit_s=180,
        planner='fixed_rm_navigation_fixture',maximum_requests=maximum_requests,skill_ticks=100,skill_seconds=20,
        inputs='privileged server pose, visits, actual motion and existing root wood sensor; no wall map',
        rule='Select lowest-cost admissible candidate, then locate_wood (skip if known), approach_wood, original root',
        revision=args.revision,navigation_strategy=args.strategy,
        no_repeats=True,uncalled_policy='exclude from recovery denominator',argv=argv,frozen_assets=frozen))
    game=None;claim_automatic_stop(stop,out/'STOP-before.txt')
    try:
        with (out/'launcher.log').open('w',encoding='utf-8') as log:
            game=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            write_json(out/'launcher-pid.json',dict(pid=game.pid));started=time.monotonic()
            while game.poll() is None:
                if time.monotonic()-started>970:raise TimeoutError('integration game timeout')
                time.sleep(.1)
        summary=json.loads((out/'run/summary.json').read_text(encoding='utf-8'))
        assert not any(k in summary for k in ('error','policy_close_error','environment_close_error')),summary.get('error')
        assert game.returncode==0 or (game.returncode==2 and summary['status']=='planning_paused')
        assert summary['actual_llm_calls']==0 and summary['rl_updates']==0
        assert summary['release']=='neutral_command_sent' and not summary['process_cleanup']['remaining_live_pids']
        for path,sha in frozen.items():assert digest(root/path)==sha,path
        receipts=json.loads((out/'run/navigation-receipts.json').read_text()) if (out/'run/navigation-receipts.json').exists() else []
        result.update(completed=True,status=summary['status'],steps=summary['steps'],planner_calls=summary['planner_calls'],
            root_success=summary['server_target_reached'],navigation_called=bool(receipts),
            navigation_outcomes=[r['outcome'] for r in receipts])
    except BaseException as exc:result['error']=repr(exc)
    finally:
        if game and game.poll() is None:
            with stop.open('a',encoding='utf-8') as stream:stream.write('\nRM navigation validation interrupted; inspect before rearming.')
            try:game.wait(timeout=100)
            except subprocess.TimeoutExpired:result['error']='owned game did not finish after STOP'
        if game is None or game.poll() is not None:publish_completion(stop,'RL collection ended; RM navigation integration trial finished.')
        write_json(out/'work.json',result)
    print(json.dumps(result));return 0 if result['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
