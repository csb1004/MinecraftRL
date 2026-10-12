"""Two fixed runtime probes then one deliberate startup cancellation; no training."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_rl_batch import ROOT,STOP,rearm_allowed,wait_owned
from minecraftrl.rl_expanded import validate_rollout
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

parser=argparse.ArgumentParser()
parser.add_argument('--recover-java-preflight',action='store_true')
args=parser.parse_args()
base=ROOT/'results/rl-backend-preparation-20261007'
if args.recover_java_preflight:
    previous=json.loads((base/'experiment.json').read_text())
    failed_folder=Path((base/'prepared-104869.log').read_text().splitlines()[0])/'run'
    failed=json.loads((failed_folder/'summary.json').read_text())
    assert not previous['completed'] and not previous['runs']
    assert failed['status']=='blocked' and failed['steps']==0 and failed['problems']==['Java unavailable']
    base=ROOT/'results/rl-backend-preparation-recovery-20261007'
base.mkdir(exist_ok=False)
original=json.loads((ROOT/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
replay=json.loads((ROOT/'results/policy-startup-prepared-20261007/diagnosis.json').read_text())
assert replay['completed'] and all(r['logp_error']==r['feature_max_error']==0 for r in replay['replayed_actions'])
checkpoint=Path(original['parent_checkpoint'])
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
state=dict(completed=False,active=None,runs=[],errors=[],seeds=[104869,104831],
    parent_checkpoint=str(checkpoint),parent_sha256=original['parent_sha256'],
    embed_sha256=digest(embed),worlds={str(s):original['worlds'][str(s)] for s in (104869,104831)},
    actor_updates=0,value_updates=0,scope='execution preparation; excluded from learning and policy performance comparison')
def save():write_json(base/'experiment.json',state)
def command(seed):
    rearm_allowed();world=state['worlds'][str(seed)]
    assert digest(checkpoint)==state['parent_sha256'] and digest(embed)==state['embed_sha256']
    assert digest(world['path'])==world['sha256']
    return [sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(seed),'--split','eval',
        '--limit-s','180','--world',world['path'],'--checkpoint',str(checkpoint),'--embed',str(embed)]
save()
try:
    for seed in state['seeds']:
        state['active']='prepared-'+str(seed);save();log=base/(state['active']+'.log')
        wait_owned(command(seed),log)
        folder=Path(log.read_text().splitlines()[0])/'run';report=json.loads((folder/'summary.json').read_text())
        state['runs'].append(dict(seed=seed,path=str(folder),report=report));save()
        validate_rollout(report,'eval');assert not report['rl_training_eligible']
    state['active']='cancel-during-preparation';save()
    path=base/'cancellation.log';folder=None;requested=None
    with path.open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(command(104869),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+780
        try:
            while proc.poll() is None:
                if folder is None:
                    with path.open() as reader:first=reader.readline().strip()
                    if first:folder=Path(first)/'run'
                if folder and (folder/'backend-preparation-started.json').exists():
                    requested=dict(preparation_finished=(folder/'backend-preparation.json').exists(),
                        events_opened=(folder/'events.jsonl').exists(),requested_at=time.time())
                    STOP.write_text('Backend preparation cancellation probe; keep STOP latched.')
                    write_json(base/'cancellation-request.json',requested)
                    break
                if time.monotonic()>=deadline:raise TimeoutError('preparation marker not reached')
                time.sleep(.05)
            proc.wait(timeout=90)
        finally:
            if proc.poll() is None:STOP.write_text('Backend probe timeout; keep STOP latched.');proc.wait(timeout=100)
    assert requested is not None
    report=json.loads((folder/'summary.json').read_text())
    state['cancellation']=dict(path=str(folder),request=requested,report=report,launcher_exit=proc.returncode);save()
    assert report['status']=='cancelled' and report['steps']==0 and not report['process_cleanup']['remaining_live_pids']
    assert not requested['preparation_finished'] and not requested['events_opened']
    state['completed']=True;state['active']=None
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
