"""One-use, 180-second known-route or map-free recovery trial after a real stall."""
import argparse,json,subprocess
from pathlib import Path
from minecraftrl.companion import worker_command
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=['far','corner']);p.add_argument('planner',choices=['fixture','visit_memory'])
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/'results/blind-recovery-20261012'/a.case/a.planner;out.mkdir(parents=True,exist_ok=False)
    stop=root/'results/STOP';linux='/mnt/d/DATA/Desktop/MinecraftRL';seed=12021 if a.case=='far' else 12022
    paths=dict(checkpoint='results/onset-coverage-training-20261009/checkpoint.pt',
        embed='results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy',
        world='results/blind-recovery-20261012/worlds/'+a.case+'.zip',
        waypoint_actor='results/waypoint-training-goal-focused-20261010/checkpoint.pt')
    training=json.loads((root/'results/waypoint-training-goal-focused-20261010/training.json').read_text())
    assert training['completed'] and digest(root/paths['waypoint_actor'])==training['checkpoint_sha256']
    frozen={path:digest(root/path) for path in paths.values()}
    for folder in ('src/minecraftrl','scripts'):
        for path in (root/folder).glob('*.py'):
            dest=out/'source-at-game'/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    argv=worker_command(linux,'Ubuntu-24.04','unused',104761,180,(out/'run').relative_to(root).as_posix())
    index=argv.index('--world-slot');del argv[index:index+2];argv[argv.index('session')]='episode'
    argv+=['--condition','wood','--rl-collect','--rl-split','eval','--rl-reward-mode','wood-potential-v4',
        '--rl-sampling-seed',str(seed),'--rl-waypoint-start','stall','--rl-waypoint-planner',a.planner]
    for name,path in paths.items():argv+=['--rl-'+name.replace('_','-'),linux+'/'+path]
    result=dict(completed=False,error=None,case=a.case,planner=a.planner,sampling_seed=seed,api_calls=0,optimizer_updates=0,frozen_assets=frozen)
    claim_automatic_stop(stop,out/'STOP-before.txt')
    try:
        with (out/'launcher.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            try:proc.wait(timeout=970)
            finally:
                if proc.poll() is None:stop.touch(exist_ok=True);proc.wait(timeout=100)
        summary=json.loads((out/'run/summary.json').read_text());assert 'error' not in summary,summary.get('error')
        assert summary['release']=='neutral_command_sent' and not summary['process_cleanup']['remaining_live_pids']
        for path,sha in frozen.items():assert digest(root/path)==sha,path
        result.update(completed=True,status=summary['status'],steps=summary['steps'],recovery_called=summary['mode_steps']['waypoint']>0,
            route_completed=summary['route_completed'],route_receipts=summary['route_receipts'],root_success=summary['root_inventory_success'])
    except BaseException as exc:result['error']=repr(exc)
    finally:
        publish_completion(stop,'RL collection ended; blind recovery development trial finished.');write_json(out/'work.json',result)
    print(json.dumps(result));return 0 if result['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
