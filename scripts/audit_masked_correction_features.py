"""Compare extracted student-prefix features with the actual collected policy inputs."""
import json
from pathlib import Path
import torch
from minecraftrl.temporal_bc import episode_rows
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];base=root/'results/masked-correction-features-20261009'
out=base/'source-replay-audit.json'
if out.exists():raise FileExistsError('one-use source audit exists')
data=torch.load(base/'features.pt',map_location='cpu',weights_only=False);groups=episode_rows(data)
receipt=json.loads((base/'features.json').read_text())
assert receipt['completed'] and digest(base/'features.pt')==receipt['features_sha256']
reports=[];labels=context=0
for s in data['plan']['sessions']:
    run=root/s['run'];prefix=run/'student-prefix.pt'
    assert digest(prefix)==s['files_sha256']['student-prefix.pt']
    original=torch.load(prefix,map_location='cpu',weights_only=False)
    rows=groups[s['id']];student=original['rows'];assert len(student)==s['handoffs'][0]['step']-1
    error=max(float((a['features']-b['features']).abs().max()) for a,b in zip(student,rows))
    assert error<1e-5,(s['id'],error)
    for i,r in enumerate(rows):
        assert r['loss_eligible']==s['frames'][i]['loss_eligible']
        labels+=r['loss_eligible'];context+=not r['loss_eligible']
        assert (r['action'] is not None)==r['loss_eligible']
    reports.append(dict(session=s['id'],original_student_inputs=len(student),max_feature_error=error,
                        expert_targets=sum(r['loss_eligible'] for r in rows)))
    del original
assert labels==1624 and context==3337
result=dict(verified=True,features_sha256=digest(base/'features.pt'),original_student_inputs=sum(r['original_student_inputs'] for r in reports),
            source_replays=reports,expert_targets=labels,context_without_targets=context,game_reruns=0)
write_json(out,result);print(json.dumps(result))
