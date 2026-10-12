"""One fixed additional training rollout per new world; no actor updates."""
import json,sys
from pathlib import Path
from run_rl_batch import ROOT,rearm_allowed,wait_owned
from minecraftrl.rl_expanded import validate_rollout
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

NEW_SEEDS=(104827,104831,104849,104851,104869,104879,104891,104911)
base=ROOT/'results/rl-value-data-expansion-20261006';base.mkdir(exist_ok=False)
old=json.loads((ROOT/'results/rl-expanded-20261006T095005817265Z/experiment.json').read_text())
checkpoint=Path(old['parent_checkpoint']);embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
assert set(NEW_SEEDS).isdisjoint(old['train_seeds']+old['validation_seeds']+old['reserved_final_seeds'])
state=dict(completed=False,active=None,errors=[],new_seeds=NEW_SEEDS,train_seeds=old['train_seeds']+list(NEW_SEEDS),
 validation_seeds=old['validation_seeds'],reserved_final_seeds=old['reserved_final_seeds'],
 parent_checkpoint=str(checkpoint),parent_sha256=old['parent_sha256'],embed_sha256=digest(embed),worlds={},runs=[],
 actor_updates=0,limit_s=180,training=dict(scope='collection manifest only; no optimizer run',completed=False,trajectories=list(old['training']['trajectories'])))
def save():write_json(base/'experiment.json',state)
save()
try:
 for seed in NEW_SEEDS:
  for prepare in (True,False):
   label=('prepare-' if prepare else 'train-')+str(seed)
   rearm_allowed();assert digest(checkpoint)==state['parent_sha256'] and digest(embed)==state['embed_sha256']
   state['active']=label;save();print(label,flush=True)
   argv=[sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(seed),'--limit-s','180','--split','eval' if prepare else 'train']
   if prepare:argv.append('--prepare')
   else:
    world=state['worlds'][str(seed)];assert digest(world['path'])==world['sha256']
    argv+=['--world',world['path'],'--checkpoint',str(checkpoint),'--embed',str(embed)]
   log=base/(label+'.log');wait_owned(argv,log)
   folder=Path(log.read_text().splitlines()[0])/'run';report=json.loads((folder/'summary.json').read_text())
   state['runs'].append(dict(label=label,seed=seed,path=str(folder),report=report));save()
   assert report['execution_valid'] and not report['process_cleanup']['remaining_live_pids']
   if prepare:
    archive=folder/'initial-world.zip';state['worlds'][str(seed)]=dict(path=str(archive),sha256=digest(archive))
   else:
    validate_rollout(report,'train');path=folder/'trajectory.pt'
    state['training']['trajectories'].append(dict(path=str(path),sha256=digest(path),seed=seed,samples=report['steps'],status=report['status'],reward_sum=report['reward_sum']))
   save()
 state['completed']=True;state['active']=None
except BaseException as exc:state['errors'].append(repr(exc))
finally:save()
raise SystemExit(0 if state['completed'] else 2)
