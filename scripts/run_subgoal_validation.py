"""One-use development game for three neural options. No API or weight update."""
import json
import argparse
import subprocess
import sys
from pathlib import Path

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.output.resolve()
    if not base.is_relative_to(root/'results/subgoal-skills-20261010'):raise ValueError('unit-local output required')
    base.mkdir(exist_ok=False)
    assets=dict(checkpoint='results/onset-coverage-training-20261009/checkpoint.pt',
        embed='results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy',
        recovery_embed='results/goal-conditioning-diagnosis-20261009/explore_9711.npy',
        world='results/diverse-teacher-worlds-20261008/development-corner.zip',
        skill_catalogue='results/subgoal-skills-20261010/embeddings/catalogue.json')
    expected=dict(checkpoint='473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd',
        world='2851b755a45623f97ef800ac844b66e0c22964481352de66f1fae4220a0d7eef')
    for name,sha in expected.items():assert digest(root/assets[name])==sha
    assert digest(root/'docs/research-spec.md')=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
    sources=list((root/'src/minecraftrl').glob('*.py'))+[Path(__file__),root/'scripts/run_goal_recovery_pilot.py']
    frozen={p.relative_to(root).as_posix():digest(p) for p in sources+[root/p for p in assets.values()]}
    for p in sources:
        dest=base/'source-at-game'/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes())
    state=dict(completed=False,error=None,api_calls=0,optimizer_updates=0,sampling_seed=9813,
        world_seed=104761,frozen_sha256=frozen,run=None)
    write_json(base/'work.json',state)
    argv=[sys.executable,str(root/'scripts/run_goal_recovery_pilot.py'),'--seed','104761',
        '--sampling-seed','9813','--limit-s','180','--rm-plan-fixture','subgoals',
        '--planner-max-requests','3','--planner-delay','5']
    for k,v in assets.items():argv+=['--'+k.replace('_','-'),str(root/v)]
    try:
        with (base/'game.log').open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            state['owned_launcher_pid']=process.pid;write_json(base/'work.json',state)
            try:process.wait(timeout=970)
            finally:
                if process.poll() is None:
                    (root/'results/STOP').write_text('Subgoal validation timeout; inspect before rearming.')
                    process.wait(timeout=100)
        lines=(base/'game.log').read_text(encoding='utf-8').splitlines()
        run=Path(lines[0])/'run';state['run']=str(run)
        report=json.loads((run/'summary.json').read_text(encoding='utf-8'))
        state.update(status=report['status'],steps=report['steps'],planner_calls=report['planner_calls'])
        assert 'error' not in report,report.get('error')
        assert report['release']=='neutral_command_sent' and not report['process_cleanup']['remaining_live_pids']
        assert report['actual_llm_calls']==report['rl_updates']==0
        for name,sha in frozen.items():assert digest(root/name)==sha,name
        state['completed']=True
    except BaseException as exc:state['error']=repr(exc)
    finally:write_json(base/'work.json',state)
    print(json.dumps(state));return 0 if state['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
