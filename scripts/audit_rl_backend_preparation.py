"""Audit the fixed startup probes without fitting or changing any policy."""
import json
import sys
from pathlib import Path
import torch
from audit_rl_pilot import audit_run,local
from run_rl_batch import compare_initial
from minecraftrl.rl_expanded import validate_rollout
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'external/steve1'))
base=root/'results/rl-backend-preparation-recovery-20261007'
state=json.loads((base/'experiment.json').read_text())
assert state['completed'] and len(state['runs'])==2
old=json.loads((root/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
checkpoint=local(state['parent_checkpoint'],root)
assert digest(checkpoint)==state['parent_sha256']
parent=torch.load(checkpoint,map_location='cpu')
audits=[]
for item in state['runs']:
    folder=local(item['path'],root)
    report=item['report'];validate_rollout(report,'eval')
    assert not report['rl_training_eligible']
    result=audit_run(folder,parent['actor'],parent['value'],'eval')
    first=None
    with (folder/'events.jsonl').open() as stream:
        for line in stream:
            event=json.loads(line)
            if event['event']=='inference_finished':
                first=event['call_finished_s']-event['call_started_s'];break
    previous=next(r for r in old['runs'] if r['label']=='train-'+str(item['seed']))
    comparison=compare_initial(local(previous['path'],root),folder)
    assert comparison['state_identical']
    world=state['worlds'][str(item['seed'])]
    assert digest(local(world['path'],root))==world['sha256']
    audits.append(dict(seed=item['seed'],audit=result,initial_comparison=comparison,
        preparation=json.loads((folder/'backend-preparation.json').read_text()),
        first_inference_s=first,under_two_seconds=first is not None and first<2))
cancel=state['cancellation']
assert cancel['report']['status']=='cancelled' and cancel['report']['steps']==0
assert not cancel['report']['process_cleanup']['remaining_live_pids']
assert not cancel['request']['events_opened'] and not cancel['request']['preparation_finished']
assert digest(root/'docs/research-spec.md')=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
result=dict(audited=True,probes=audits,cancellation=cancel,
    timing_criterion_met=all(a['under_two_seconds'] for a in audits),actor_updates=0,value_updates=0,
    scope='execution preparation only; no policy improvement inference')
write_json(base/'audit.json',result)
print(json.dumps(result,indent=2))
