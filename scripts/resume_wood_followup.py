"""Continue only the ten unrun natural-world trials after the preserved death diagnostic error."""
import json,sys
from pathlib import Path
from run_rl_batch import ROOT,rearm_allowed,wait_owned,compare_initial
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

base=ROOT/'results/wood-followup-20261007'
path=base/'experiment.json';state=json.loads(path.read_text())
assert not state['completed'] and len(state['runs'])==9 and len(state['unrun'])==10
assert state['runs'][-1]['label']=='probe-death' and not state['runs'][-1]['report']['execution_valid']
assert all(p['label'].startswith('natural-') for p in state['unrun'])
with (base/'experiment-before-natural-resume.json').open('x') as f:f.write(path.read_text())
state['resume_reason']='Preserve death diagnostic timing error; execute independent unrun natural trials with unchanged source/checkpoints.'
plan=list(state['unrun'])
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
def save():write_json(path,state)
try:
    for item in plan:
        state['active']=item['label'];save();rearm_allowed()
        for source,sha in state['source_sha256'].items():assert digest(ROOT/source)==sha,source
        for weight in state['checkpoints'].values():assert digest(weight['path'])==weight['sha256']
        assert digest(item['world'])==state['world_sha256'][item['world']]
        assert digest(embed)==state['embed_sha256']
        log=base/(item['label']+'.log');assert not log.exists()
        argv=[sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(item['seed']),'--split','eval',
              '--reward-mode','wood-potential-v2','--world',item['world'],
              '--checkpoint',state['checkpoints'][item['condition']]['path'],'--embed',str(embed)]
        print(item['label'],flush=True)
        try:wait_owned(argv,log)
        finally:
            folder=Path(log.read_text().splitlines()[0])/'run'
            if (folder/'summary.json').exists():
                report=json.loads((folder/'summary.json').read_text())
                state['runs'].append(dict(item,path=str(folder),report=report))
                state['unrun']=[p for p in plan if p['label'] not in {r['label'] for r in state['runs']}];save()
        assert report['execution_valid'] and not report['process_cleanup']['remaining_live_pids']
        previous=[r for r in state['runs'][:-1] if r.get('world')==item['world'] and 'probe' not in r]
        if previous:state.setdefault('comparisons',{})[item['label']]=compare_initial(Path(previous[0]['path']),folder);save()
        print(json.dumps(dict(label=item['label'],status=report['status'],steps=report['steps'])),flush=True)
    state['completed']=True;state['active']=None
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
