"""Windows launcher for an isolated, eval-only neural goal-conditioning probe."""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

from minecraftrl.companion import worker_command
from minecraftrl.rl_control import claim_automatic_stop, publish_completion


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    from minecraftrl.cli import sampling_seed
    p.add_argument('--sampling-seed',type=sampling_seed,required=True)
    for name in ('checkpoint','embed','recovery-embed','world'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit-s',type=float,default=180)
    planner=p.add_mutually_exclusive_group()
    planner.add_argument('--planner-choice',choices=['explore_then_resume','remain_paused'])
    planner.add_argument('--planner-exchange',type=Path)
    planner.add_argument('--rm-plan-fixture',choices=['explore','unsupported','subgoals'])
    p.add_argument('--skill-catalogue',type=Path)
    p.add_argument('--rm-plan',action='store_true')
    p.add_argument('--planner-max-requests',type=int,choices=[1,2,3],default=1)
    p.add_argument('--planner-exchange-live',action='store_true')
    p.add_argument('--planner-delay',type=float,default=5.)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.skill_catalogue:
        from minecraftrl.subgoal_skills import load_catalogue
        args.skill_catalogue=args.skill_catalogue.resolve()
        load_catalogue(args.skill_catalogue,root)
        if not (args.rm_plan or args.rm_plan_fixture):raise SystemExit('Catalogue requires RM planning')
    if args.rm_plan_fixture=='subgoals' and not args.skill_catalogue:
        raise SystemExit('Subgoal fixture requires catalogue')
    if args.planner_exchange_live and not args.planner_exchange:
        raise SystemExit('Live scope requires a mailbox')
    if (args.planner_choice or args.rm_plan_fixture) and not 0<=args.planner_delay<=60:
        raise SystemExit('Fixed response delay must be finite and in [0,60]')
    if args.planner_exchange:
        args.planner_exchange=args.planner_exchange.resolve()
        if not args.planner_exchange.is_relative_to(root/'results') or args.planner_exchange.exists():
            raise SystemExit('Mailbox must be a new project-local results directory')
    stop = root/'results/STOP'
    if not stop.exists():
        raise SystemExit('STOP must be latched before isolated trial')
    if not 0 < args.limit_s <= 180:
        raise ValueError('control limit must be in (0,180]')
    paths = {name:getattr(args,name).resolve().relative_to(root).as_posix()
             for name in ('checkpoint','embed','recovery_embed','world')}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base = root/'results'/f'goal-recovery-eval-{args.seed}-{stamp}'
    base.mkdir()
    print(base,flush=True)
    linux = '/mnt/d/DATA/Desktop/MinecraftRL'
    argv = worker_command(linux,'Ubuntu-24.04','unused',args.seed,args.limit_s,(base/'run').relative_to(root).as_posix())
    index = argv.index('--world-slot')
    del argv[index:index+2]
    argv[argv.index('session')] = 'episode'
    argv += ['--condition','wood','--rl-collect','--rl-split','eval','--rl-reward-mode','wood-potential-v4',
             '--rl-sampling-seed',str(args.sampling_seed)]
    if args.planner_choice:
        argv += ['--rl-planner-choice',args.planner_choice,'--rl-planner-delay',str(args.planner_delay)]
    if args.rm_plan_fixture:
        argv += ['--rl-rm-plan-fixture',args.rm_plan_fixture,'--rl-planner-delay',str(args.planner_delay)]
    if args.rm_plan:argv.append('--rl-rm-plan')
    if args.skill_catalogue:
        argv += ['--rl-skill-catalogue',linux+'/'+args.skill_catalogue.relative_to(root).as_posix()]
    argv += ['--rl-planner-max-requests',str(args.planner_max_requests)]
    if args.planner_exchange:
        argv += ['--rl-planner-exchange',linux+'/'+args.planner_exchange.relative_to(root).as_posix()]
        if args.planner_exchange_live:argv.append('--rl-planner-exchange-live')
    for name,path in paths.items():
        argv += ['--rl-'+name.replace('_','-'),linux+'/'+path]
    claim_automatic_stop(stop,base/'STOP-before.txt')
    try:
        with (base/'supervisor.log').open('w',encoding='utf-8') as log:
            process = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                process.wait(timeout=870)
            finally:
                if process.poll() is None:
                    stop.touch(exist_ok=True)
                    process.wait(timeout=90)
        report = json.loads((base/'run/summary.json').read_text(encoding='utf-8'))
        print(json.dumps(report),flush=True)
        return process.returncode
    finally:
        publish_completion(stop,'RL collection ended; goal conditioning diagnostic finished.')


if __name__ == '__main__':
    raise SystemExit(main())
