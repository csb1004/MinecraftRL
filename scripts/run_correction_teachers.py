"""Single-use two training-world teacher demonstrations; no optimization."""
import json,sys
from pathlib import Path
from run_rl_batch import ROOT,rearm_allowed,wait_owned
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

base=ROOT/'results/correction-teacher-20261008';base.mkdir(exist_ok=False)
worlds=json.loads((ROOT/'results/blocked-entry-worlds-20261008/worlds.json').read_text())['worlds']
frozen={str(p):digest(p) for p in (ROOT/'docs/research-spec.md',ROOT/'results/recovery-gap-20261007/train-gap-epochs2/checkpoint.pt')}
state=dict(completed=False,active=None,runs=[],errors=[],optimizer_updates=0,learned_policy_episodes=0,frozen=frozen)
def save():write_json(base/'experiment.json',state)
save()
try:
    for seed,world in worlds.items():
        assert world['split']=='train' and digest(world['path'])==world['sha256']
        rearm_allowed();state['active']=seed;save()
        log=base/(seed+'.log');print('teacher '+seed,flush=True)
        try:
            wait_owned([sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',seed,'--split','train',
                        '--teacher','--world',world['path'],'--limit-s','180'],log)
        finally:
            lines=log.read_text().splitlines() if log.exists() else []
            if lines:
                run=Path(lines[0])/'run'
                if (run/'summary.json').exists():
                    report=json.loads((run/'summary.json').read_text())
                    state['runs'].append(dict(path=str(run),seed=int(seed),report=report,audited=False));save()
        assert report['execution_valid'] and report['worker_exit_code']==0
        assert not report['rl_training_eligible'] and not report['learned_policy_loaded']
        assert not report['process_cleanup']['remaining_live_pids']
        release=json.loads((run/'input-release-state.json').read_text())
        assert release['verified'] and release['held_key_count']==0
        rows=json.loads((run/'teacher-rows.json').read_text())
        events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
        transitions=[e for e in events if e['event']=='transition']
        assert len(rows)==len(transitions)==report['steps']
        stopped=False
        for event in events:
            if event['event']=='stop_acknowledged':stopped=True
            if stopped:assert event['event']!='policy_dispatch_started'
        assert stopped
        for i,(row,event) in enumerate(zip(rows,transitions)):
            assert row['action']==event['action'] and (run/row['frame']).is_file()
            if i:assert row['server_before']==rows[i-1]['server_after']
        final=rows[-1]['server_after']['server_inventory']
        state['runs'][-1].update(audited=True,release=release,server_inventory=final,
                                 rows_sha256=digest(run/'teacher-rows.json'),events_sha256=digest(run/'events.jsonl'))
        save();print(json.dumps(dict(seed=seed,status=report['status'],steps=report['steps'],inventory=final)),flush=True)
        assert digest(world['path'])==world['sha256']
    for path,sha in frozen.items():assert digest(path)==sha
    state.update(completed=True,active=None)
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
