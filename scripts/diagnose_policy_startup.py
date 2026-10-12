"""Actual model on saved frames; no game and no environment actions delivered."""
import argparse
import faulthandler
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'external/steve1'))
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cold',action='store_true')
args=parser.parse_args();out=args.output.resolve()
assert out.is_relative_to(ROOT/'results') and (ROOT/'results/STOP').exists()
assert '/usr/lib/wsl/lib' in os.environ.get('LD_LIBRARY_PATH','').split(':')
out.mkdir(exist_ok=False)
from minecraftrl.runtime import write_json,jsonable
from minecraftrl.world_store import digest
from audit_rl_pilot import local
import torch
import numpy as np
from PIL import Image
from gym import spaces
from steve1.VPT.agent import ENV_KWARGS,TARGET_ACTION_SPACE
from minecraftrl.rl_policy import RLPolicy,state_fingerprint

state=json.loads((ROOT/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
source=local(next(r['path'] for r in state['runs'] if r['label']=='train-104869'),ROOT)
checkpoint=local(state['parent_checkpoint'],ROOT)
assert digest(checkpoint)==state['parent_sha256']
frame=np.array(Image.open(source/'frames/000000.png'))
log=(out/'stages.jsonl').open('w')

def snapshot():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return dict(user_s=r.ru_utime,system_s=r.ru_stime,major_faults=r.ru_majflt,
        allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20)

def emit(kind,**kwargs):
    row=dict(event=kind,time=time.monotonic(),**kwargs)
    log.write(json.dumps(row)+'\n');log.flush()
    return row

env=SimpleNamespace(task=SimpleNamespace(**ENV_KWARGS),action_space=spaces.Dict(TARGET_ACTION_SPACE))
emit('loading')
policy=RLPolicy(env,ROOT,out,checkpoint,source/'goal-embed.npy',104869,prewarm=not args.cold)
policy.reset('collect wood');torch.cuda.synchronize();emit('ready',usage=snapshot())
initial_state=state_fingerprint(policy.agent.hidden_state)
initial_rng=state_fingerprint([torch.get_rng_state(),torch.cuda.get_rng_state_all()])
started={};hooks=[]
def before(name):
    def hook(module,inputs):
        torch.cuda.synchronize()
        started[name]=(time.monotonic(),snapshot())
        emit('conv_started',name=name,shape=list(inputs[0].shape),dtype=str(inputs[0].dtype),usage=started[name][1])
    return hook
def after(name):
    def hook(module,inputs,output):
        torch.cuda.synchronize();t,old=started[name];now=snapshot()
        emit('conv_finished',name=name,seconds=time.monotonic()-t,usage_delta={k:now[k]-v for k,v in old.items()})
    return hook
for name,module in policy.agent.policy.net.named_modules():
    if isinstance(module,torch.nn.Conv2d):
        hooks += [module.register_forward_pre_hook(before(name)),module.register_forward_hook(after(name))]
timings=[];reference=None
with (out/'first-forward-stack.txt').open('w') as stack:
    faulthandler.dump_traceback_later(20,repeat=True,file=stack)
    try:
        for i in range(3):
            start=time.monotonic();features,next_state=policy.features(frame);torch.cuda.synchronize()
            timings.append(time.monotonic()-start)
            if reference is None:reference=features.detach().cpu()
            else:assert torch.equal(reference,features.cpu())
            assert state_fingerprint(policy.agent.hidden_state)==initial_state
            assert state_fingerprint([torch.get_rng_state(),torch.cuda.get_rng_state_all()])==initial_rng
            emit('forward_finished',repeat=i,seconds=timings[-1])
            if i==0:
                for hook in hooks:hook.remove()
            del next_state,features
    finally:faulthandler.cancel_dump_traceback_later()
data=torch.load(source/'trajectory.pt',map_location='cpu')
assert state_fingerprint(data['initial_recurrent_state'])==initial_state
feature_error=(reference-data['rows'][0]['features']).abs().max().item()
replay=[]
for i,row in enumerate(data['rows'][:10]):
    action=policy.act(np.array(Image.open(source/'frames'/('%06d.png'%i))))
    assert jsonable(action)==row['env_action']
    error=abs(policy.pending['old_logp']-row['old_logp'])
    assert error<2e-4
    replay.append(dict(step=i+1,logp_error=error,feature_max_error=(policy.pending['features']-row['features']).abs().max().item()))
    policy.commit(action,row['reward'],row['terminal'])
report=dict(completed=True,scope='actual model replay on stored pixels; zero environment actions',
    source=str(source),source_trajectory_sha256=digest(source/'trajectory.pt'),parent_sha256=digest(checkpoint),
    backend_preparation=policy.backend_preparation,
    forward_seconds=timings,first_feature_max_error=feature_error,recurrent_state_preserved=True,
    torch_cpu_cuda_rng_preserved=True,replayed_actions=replay,updates=0,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
write_json(out/'diagnosis.json',report);policy.dispose();log.close()
assert (ROOT/'results/STOP').exists()
print(json.dumps(report))
