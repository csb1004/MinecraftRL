"""Verify the exact trained head/tail tensors were present in actual eval runs."""
import argparse,json
from pathlib import Path
import torch
from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

p=argparse.ArgumentParser();p.add_argument('batch',type=Path);args=p.parse_args()
root=Path(__file__).resolve().parents[1];base=args.batch.resolve()
assert base.is_relative_to(root/'results')
out=base/'tail-runtime-audit.json'
if out.exists():raise FileExistsError(out)
state=json.loads((base/'experiment.json').read_text())
assert len(state['runs'])==4 and all(r['audited'] for r in state['runs'])
assert not state['errors'] or state['errors']==["AssertionError('initial state/pixel mismatch')"]
rows=[]
for item in state['runs']:
    candidate=item['label'].endswith('-candidate')
    checkpoint=Path(state['candidate' if candidate else 'parent'])
    # The batch is made on Windows; resolve project-relative paths in WSL.
    checkpoint=root/str(checkpoint).replace('\\','/').split('/MinecraftRL/',1)[-1]
    run=root/item['path'].replace('\\','/').split('/MinecraftRL/',1)[-1]
    assert digest(checkpoint)==item['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu');trajectory=torch.load(run/'trajectory.pt',map_location='cpu')
    assert trajectory['split']=='eval'
    actual=trajectory['initial_heads'];condition=json.loads((run/'rl-condition.json').read_text())
    for key in ('actor','value'):assert state_fingerprint(saved[key])==state_fingerprint(actual[key])
    if candidate:
        assert saved['scope']=='head_and_tail'
        fingerprint=state_fingerprint(saved['backbone_tail'])
        assert fingerprint==state_fingerprint(actual['backbone_tail'])==condition['backbone_tail_sha256']
        assert saved['base_model_sha256']==digest(root/'weights/steve1/steve1.weights')
    else:
        assert saved['scope']=='head_only' and 'backbone_tail' not in actual
        fingerprint=None
    rows.append(dict(label=item['label'],verified=True,tail_sha256=fingerprint,checkpoint_sha256=digest(checkpoint)))
    del trajectory,saved
write_json(out,dict(verified=True,rows=rows,scope='Actual evaluation checkpoint tensors; not a task success claim'))
print(json.dumps(rows))
