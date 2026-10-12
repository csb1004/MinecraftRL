"""Bounded local snapshot/reproducibility gate, three-seed update and held-out validation."""
import argparse,datetime,json,subprocess,sys
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1]
LINUX='/mnt/d/DATA/Desktop/MinecraftRL'
STOP=ROOT/'results/STOP'

def linux(path):return LINUX+'/'+Path(path).resolve().relative_to(ROOT).as_posix()

def rearm_allowed():
    if not STOP.exists() or not STOP.read_text(encoding='utf-8').startswith(('RL training ended','RL collection ended')):
        raise RuntimeError('Unexpected STOP; batch will not override manual stop')

def wait_owned(argv,logpath,timeout=960):
    with logpath.open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        try:proc.wait(timeout=timeout)
        finally:
            if proc.poll() is None:STOP.write_text('batch timeout');proc.wait(timeout=100)
    if proc.returncode:raise RuntimeError('Subprocess failed: '+str(logpath))

def compare_initial(a,b):
    from PIL import Image,ImageChops,ImageStat
    states=[json.loads((p/'initial-state.json').read_text()) for p in (a,b)]
    selected=[]
    for state in states:
        server=state['server'];obs=state['observation']
        selected.append(dict(archive=state['initial_world']['sha256'],inventory=server['server_inventory'],
            position=[server[k] for k in ('server_x','server_y','server_z','yaw','pitch')],world_time=server['world_time'],
            life=obs['life_stats']))
    imgs=[Image.open(p/'frames/000000.png') for p in (a,b)]
    conditions=[json.loads((p/'rl-condition.json').read_text()) for p in (a,b)]
    memory_identical=all(conditions[0][k]==conditions[1][k] for k in ('initial_recurrent_sha256','sampling_seed','embedding_sha256'))
    diff=ImageChops.difference(*imgs)
    return dict(state_identical=selected[0]==selected[1] and memory_identical,memory_identical=memory_identical,states=selected,pixel_identical=diff.getbbox() is None,
                pixel_difference_bbox=diff.getbbox(),mean_abs_pixel=ImageStat.Stat(diff).mean)

def main():
    p=argparse.ArgumentParser();entry=p.add_mutually_exclusive_group(required=True)
    entry.add_argument('--prepared',type=Path);entry.add_argument('--resume-optimizer',type=Path)
    args=p.parse_args()
    if args.resume_optimizer:
        base=args.resume_optimizer.resolve()
        if not base.is_relative_to(ROOT/'results'):raise ValueError('batch must be inside results')
        state=json.loads((base/'batch.json').read_text())
        failed=json.loads((base/'training/training.json').read_text())
        if state['completed'] or failed['updates'] or failed['completed'] or any(r['label'].startswith('validation-') for r in state['runs']):
            raise ValueError('only a pre-update failure before validation may resume here')
        with (base/'batch-before-optimizer-retry.json').open('x',encoding='utf-8') as snapshot:snapshot.write((base/'batch.json').read_text())
        state['previous_errors']=state['errors'];state['errors']=[]
    else:
        base=ROOT/'results'/('rl-batch-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));base.mkdir()
        state=dict(completed=False,train_seeds=[104729,104743,104759],validation_seeds=[2000003,2000029],
            reserved_final_seeds=[3000017,3000029,3000047,3000061,3000073],runs=[],errors=[],worlds={})
    checkpoint=ROOT/'results/rl-head-training-20261005/checkpoint.pt'
    embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
    def save():write_json(base/'batch.json',state)
    def trial(label,seed,*,prepare=False,world=None,weights=None,split='eval',limit=180):
        rearm_allowed();print(label,flush=True)
        argv=[sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(seed),'--split',split,'--limit-s',str(limit)]
        if prepare:argv.append('--prepare')
        else:argv+=['--embed',str(embed)]
        if world:argv+=['--world',str(world)]
        if weights:argv+=['--checkpoint',str(weights)]
        wait_owned(argv,base/(label+'.log'))
        folder=Path((base/(label+'.log')).read_text().splitlines()[0])/'run'
        report=json.loads((folder/'summary.json').read_text());state['runs'].append(dict(label=label,path=str(folder),report=report));save()
        if not report['execution_valid'] or report['release']!='neutral_command_sent' or report['process_cleanup']['remaining_live_pids']:
            raise RuntimeError('invalid collection: '+str(folder))
        return folder
    print(base,flush=True);save()
    try:
        if args.resume_optimizer:
            worlds={int(k):Path(v['path']) for k,v in state['worlds'].items()}
            for item in state['worlds'].values():
                if digest(item['path'])!=item['sha256']:raise ValueError('initial archive changed before retry')
            training=[]
            for seed in state['train_seeds']:
                matches=[r for r in state['runs'] if r['label']=='train-'+str(seed)]
                if len(matches)!=1 or not matches[0]['report']['execution_valid']:raise ValueError('incomplete collection')
                training.append(Path(matches[0]['path'])/'trajectory.pt')
            if not state['reproducibility']['state_identical'] or not state['reproducibility']['pixel_identical']:raise ValueError('missing reproducibility gate')
        else:
            worlds={104729:args.prepared.resolve()}
            # Diagnostic starts are never trained on.
            a=trial('repeat-a',104729,world=worlds[104729],weights=checkpoint,limit=45)
            b=trial('repeat-b',104729,world=worlds[104729],weights=checkpoint,limit=45)
            state['reproducibility']=compare_initial(a,b);save()
            if not state['reproducibility']['state_identical'] or not state['reproducibility']['pixel_identical']:
                raise RuntimeError('Initial state/pixels differ; diagnose before batch training')
            for seed in state['train_seeds'][1:]+state['validation_seeds']:
                folder=trial('prepare-'+str(seed),seed,prepare=True)
                worlds[seed]=folder/'initial-world.zip'
            state['worlds']={str(k):dict(path=str(v),sha256=digest(v)) for k,v in worlds.items()};save()
            training=[]
            for seed in state['train_seeds']:
                training.append(trial('train-'+str(seed),seed,world=worlds[seed],weights=checkpoint,split='train')/'trajectory.pt')
        rearm_allowed();print('optimizer',flush=True)
        trained=base/('training-retry-1' if args.resume_optimizer else 'training')
        state['training_directory']=str(trained);save()
        argv=['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',
            LINUX+'/.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python','scripts/train_rl_heads.py',
            linux(training[0]),'--additional',*[linux(x) for x in training[1:]],'--resume',linux(checkpoint),'--output',linux(trained)]
        wait_owned(argv,base/('optimizer-retry-1.log' if args.resume_optimizer else 'optimizer.log'),timeout=1200)
        state['training']=json.loads((trained/'training.json').read_text());save()
        if not state['training']['completed']:raise RuntimeError('training incomplete')
        for seed in state['validation_seeds']:
            pair=[]
            for phase,weights in [('before',checkpoint),('after',trained/'checkpoint.pt')]:
                pair.append(trial('validation-'+str(seed)+'-'+phase,seed,world=worlds[seed],weights=weights))
            comparison=compare_initial(*pair);state.setdefault('validation_initial',{})[str(seed)]=comparison;save()
            if not comparison['state_identical'] or not comparison['pixel_identical']:raise RuntimeError('validation start mismatch')
        for item in state['worlds'].values():
            if digest(item['path'])!=item['sha256']:raise RuntimeError('initial archive mutated')
        state['completed']=True
    except Exception as exc:state['errors'].append(repr(exc))
    finally:save()
    return 0 if state['completed'] else 2

if __name__=='__main__':raise SystemExit(main())
