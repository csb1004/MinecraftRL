"""Reconstruct censored endpoint features from immutable saved pixels/memory; no game."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/steve1'))
from gym import spaces
from steve1.VPT.agent import ENV_KWARGS,TARGET_ACTION_SPACE
from minecraftrl.rl_policy import RLPolicy,state_fingerprint,cpu_tree
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from audit_rl_pilot import local

base=ROOT/'results/rl-horizon-20261007';base.mkdir(exist_ok=True)
out=base/'endpoint-recovery';out.mkdir(exist_ok=False)
assert (ROOT/'results/STOP').exists()
state=json.loads((ROOT/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
parent=local(state['parent_checkpoint'],ROOT);assert digest(parent)==state['parent_sha256']
env=SimpleNamespace(task=SimpleNamespace(**ENV_KWARGS),action_space=spaces.Dict(TARGET_ACTION_SPACE))
policy=None;entries={}
def device_tree(value):
    if torch.is_tensor(value):return value.to('cuda')
    if isinstance(value,tuple):return tuple(device_tree(x) for x in value)
    if isinstance(value,list):return [device_tree(x) for x in value]
    if isinstance(value,dict):return {k:device_tree(v) for k,v in value.items()}
    return value
try:
    for item in state['training']['trajectories']:
        path=local(item['path'],ROOT);assert digest(path)==item['sha256']
        data=torch.load(path,map_location='cpu')
        if data['terminal']:continue
        if policy is None:
            policy=RLPolicy(env,ROOT,out,parent,path.parent/'goal-embed.npy',data['seed'])
        policy.embed_path=path.parent/'goal-embed.npy';policy.reset('collect wood')
        policy.agent.hidden_state=device_tree(data['final_recurrent_state']);policy.first=False
        before=state_fingerprint(policy.agent.hidden_state)
        screen=np.array(Image.open(path.parent/'frames/final.png'))
        features,_=policy.features(screen)
        with torch.no_grad():actual=policy.value(features[0:1]).item()
        assert abs(actual-data['bootstrap'])<2e-5,(actual,data['bootstrap'])
        assert before==state_fingerprint(policy.agent.hidden_state)
        entries[item['sha256']]=dict(features=cpu_tree(features[0:1]).squeeze(0),
            source_sha256=item['sha256'],frame_sha256=digest(path.parent/'frames/final.png'),
            bootstrap_error=abs(actual-data['bootstrap']),original_bootstrap=data['bootstrap'],seed=data['seed'])
        print('recovered',data['seed'],actual,flush=True)
    assert entries
    torch.save(entries,out/'features.pt')
    write_json(out/'recovery.json',dict(completed=True,environment_actions=0,
        entries=[{k:v for k,v in x.items() if k!='features'} for x in entries.values()],
        feature_file_sha256=digest(out/'features.pt'),parent_sha256=state['parent_sha256']))
finally:
    if policy:policy.dispose()
