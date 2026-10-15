"""One-use Windows launcher; all gameplay stays inside the existing WSL supervisor."""
import argparse,json,subprocess,time
from pathlib import Path
from minecraftrl.companion import worker_command
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/navigation-refinement-20261014'
PARENT='results/onset-coverage-training-20261009/checkpoint.pt'
EMBED='results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
PRIOR='results/waypoint-training-goal-focused-20261010/checkpoint.pt'
POINTS=[[4.5,6.5],[4.5,3.5],[2.5,3.5],[2.5,2.7],[4.5,2.7],[4.5,5.5]]
NEAR_POINTS=[[2.7,3.5],[3.5,3.5],[2.7,4.2],[3.5,4.2],[2.7,3.5],[4.5,5.5]]


def audit(run):
    report=json.loads((run/'summary.json').read_text());rows=json.loads((run/'navigation-rows.json').read_text())
    assert report['execution_valid'] and report['worker_exit_code']==0,report.get('error',report['status'])
    assert report['root_actor_preserved'] and report['actual_llm_calls']==0 and report['rl_updates']==0
    assert not report['process_cleanup']['remaining_live_pids']
    release=json.loads((run/'input-release-state.json').read_text());assert release['verified'] and release['held_key_count']==0
    stopped=False;count=0
    before=json.loads((run/'initial-state.json').read_text())['server']
    for line in (run/'events.jsonl').open(encoding='utf-8'):
        e=json.loads(line)
        if e['event']=='stop_acknowledged':stopped=True
        if e['event'] in ('policy_dispatch_started','inference_started'):assert not stopped
        if e['event']!='transition':continue
        r=rows[count];assert e['action']==r['action'] and e['controller']=='navigation_'+r['owner']
        assert r['server_before']==before and r['server_after']==e['info']['workshop']
        assert r['server_after']['server_tick']-before['server_tick']==1
        before=r['server_after'];count+=1
    assert stopped and count==len(rows)==report['steps']
    result=dict(verified=True,transitions=count,post_stop_actions=0,released_keys=0,remaining_owned_pids=[],
        learned_arrivals=report['learned_arrivals'],teacher_arrivals=report['teacher_arrivals'],
        files={n:digest(run/n) for n in ('summary.json','navigation-rows.json','navigation-trajectory.pt','events.jsonl')})
    write_json(run/'navigation-audit.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--world',required=True)
    p.add_argument('--mode',choices=['correction','evaluate','ppo'],required=True);p.add_argument('--actor',default=PRIOR)
    p.add_argument('--sampling-seed',type=int,default=14111)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--near',action='store_true',help='Separate near/reverse correction batch; evaluation keeps original fixed goals')
    group.add_argument('--near-low-lr',action='store_true',help='Separate low-rate follow-up evaluation; no repeated collection')
    a=p.parse_args()
    if not a.label.replace('-','').isalnum():raise ValueError('simple unique label required')
    base=ROOT/'results/near-navigation-20261015' if a.near else BASE
    if a.near_low_lr:
        if a.mode!='evaluate':p.error('low-rate follow-up only permits evaluation')
        base=ROOT/'results/near-navigation-low-lr-20261015'
    out=base/a.label;out.mkdir(parents=True,exist_ok=False)
    world=json.loads((ROOT/'results/diverse-teacher-worlds-20261008/worlds.json').read_text())['worlds'][a.world]
    path=Path(world['path']);assert digest(path)==world['sha256']
    frozen={p:digest(ROOT/p) for p in (PARENT,EMBED,PRIOR,'docs/research-spec.md')}
    assert frozen[PARENT]=='473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd'
    near=a.near and a.mode=='correction'
    config=dict(schema='navigation_trial_config_v1',mode=a.mode,split=world['split'],points=NEAR_POINTS if near else POINTS,
        actor=a.actor,actor_sha256=digest(ROOT/a.actor),world_sha256=world['sha256'],sampling_seed=a.sampling_seed)
    if near:config['student_ticks']=8
    write_json(out/'protocol.json',config)
    for folder in ('src/minecraftrl','scripts'):
        for source in (ROOT/folder).glob('*.py'):
            dest=out/'source'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    linux='/mnt/d/DATA/Desktop/MinecraftRL';stop=ROOT/'results/STOP';proc=None
    argv=worker_command(linux,'Ubuntu-24.04','unused',world['seed'],180,(out/'run').relative_to(ROOT).as_posix())
    idx=argv.index('--world-slot');del argv[idx:idx+2];argv[argv.index('session')]='episode'
    argv+=['--condition','wood','--rl-collect','--rl-split','train' if world['split']=='train' else 'eval',
        '--rl-reward-mode','wood-potential-v4','--rl-checkpoint',linux+'/'+PARENT,'--rl-embed',linux+'/'+EMBED,
        '--rl-world',linux+'/'+path.relative_to(ROOT).as_posix(),'--rl-sampling-seed',str(a.sampling_seed),
        '--rl-navigation-trial',linux+'/'+(out/'protocol.json').relative_to(ROOT).as_posix()]
    result=dict(completed=False,api_calls=0,argv=argv,frozen=frozen,error=None)
    claim_automatic_stop(stop,out/'STOP-before.txt')
    try:
        with (out/'launcher.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            started=time.monotonic();write_json(out/'launcher-pid.json',dict(pid=proc.pid))
            while proc.poll() is None:
                if time.monotonic()-started>970:raise TimeoutError('navigation trial exceeded supervisor allowance')
                time.sleep(.1)
        result['audit']=audit(out/'run')
        for path,sha in frozen.items():assert digest(ROOT/path)==sha
        result.update(completed=True,elapsed_s=time.monotonic()-started)
    except BaseException as exc:result['error']=repr(exc)
    finally:
        if proc and proc.poll() is None:
            with stop.open('a',encoding='utf-8') as f:f.write('\nNavigation trial interrupted. Inspect before rearming.')
            try:proc.wait(timeout=100)
            except subprocess.TimeoutExpired:result['error']='owned trial did not finish after STOP'
        if proc is None or proc.poll() is not None:publish_completion(stop,'RL collection ended; navigation refinement trial finished.')
        write_json(out/'work.json',result)
    print(json.dumps(result));return 0 if result['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
