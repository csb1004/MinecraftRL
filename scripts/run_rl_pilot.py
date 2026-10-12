"""Windows launcher for isolated RL collection, using the existing owned-process supervisor."""
import argparse,datetime,json,subprocess
from pathlib import Path
from minecraftrl.companion import worker_command
from minecraftrl.rl_control import claim_automatic_stop,publish_completion

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=104729)
    p.add_argument('--split',choices=['train','eval'],default='train')
    from minecraftrl.cli import sampling_seed
    p.add_argument('--sampling-seed',type=sampling_seed)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--embed',type=Path)
    p.add_argument('--world',type=Path);p.add_argument('--prepare',action='store_true')
    p.add_argument('--limit-s',type=float,default=180)
    p.add_argument('--reward-mode',choices=['sparse','wood-potential-v1','wood-potential-v2','wood-potential-v3','wood-potential-v4'],default='sparse')
    p.add_argument('--probe-route',choices=['break','detour','death'])
    p.add_argument('--teacher',action='store_true')
    p.add_argument('--student-correction',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parents[1];stop=root/'results/STOP'
    if not stop.exists():raise SystemExit('STOP must be latched before isolated RL trial')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base=root/'results'/('rl-'+args.split+'-'+str(args.seed)+'-'+stamp);base.mkdir();print(base,flush=True)
    linux='/mnt/d/DATA/Desktop/MinecraftRL'
    if not 0<args.limit_s<=180:raise ValueError('limit must be in (0,180]')
    argv=worker_command(linux,'Ubuntu-24.04','unused',args.seed,args.limit_s,(base/'run').relative_to(root).as_posix())
    i=argv.index('--world-slot');del argv[i:i+2];argv[argv.index('session')]='episode'
    argv+=['--condition','wood','--rl-collect','--rl-split',args.split]
    argv+=['--rl-reward-mode',args.reward_mode]
    if args.sampling_seed is not None:argv+=['--rl-sampling-seed',str(args.sampling_seed)]
    if args.probe_route:argv+=['--rl-probe-route',args.probe_route]
    if args.teacher:argv.append('--rl-teacher')
    if args.student_correction:argv.append('--rl-student-correction')
    if args.prepare:argv.append('--rl-prepare')
    for flag,path in (('--rl-checkpoint',args.checkpoint),('--rl-embed',args.embed),('--rl-world',args.world)):
        if path:
            path=path.resolve();relative=path.relative_to(root);argv +=[flag,linux+'/'+relative.as_posix()]
    claim_automatic_stop(stop,base/'STOP-before.txt')
    try:
        with (base/'supervisor.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            try:proc.wait(timeout=870)
            finally:
                if proc.poll() is None:stop.touch();proc.wait(timeout=90)
        report=json.loads((base/'run/summary.json').read_text(encoding='utf-8'))
        print(json.dumps(report),flush=True);return proc.returncode
    finally:publish_completion(stop,'RL collection ended; controlled re-arm required.')

if __name__=='__main__':raise SystemExit(main())
