"""Admit the frozen six correction episodes for one masked temporal BC pass."""
import json
from pathlib import Path
from minecraftrl.bc_core import validate_plan
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1]
out=root/'configs/masked-correction-bc-20261009.json'
if out.exists():raise FileExistsError('one-use admission exists')
source=root/'configs/student-correction-data-20261009.json'
audit=json.loads((root/'results/student-correction-20261009/final-audit.json').read_text())
assert audit['verified'] and digest(source)==audit['manifest_sha256']
plan=json.loads(source.read_text())
assert digest(root/plan['source_experiment'])==plan['source_experiment_sha256']
assert digest(root/plan['checkpoint'])==plan['checkpoint_sha256']
plan.update(parent=plan['checkpoint'],frame_offset=0,rm_context='recorded',masked_loss_required=True,
    source_manifest=source.relative_to(root).as_posix(),source_manifest_sha256=digest(source),
    scope='Complete correction episodes; teacher-only BC/KL, all context replayed; one temporal pass')
for s in plan['sessions']:
    run=root/s['run']
    for name,sha in s['files_sha256'].items():assert digest(run/name)==sha
    s.update(rows_sha256=s['files_sha256']['correction-rows.json'],events_sha256=s['files_sha256']['events.jsonl'],
             frame_sha256=[f['sha256'] for f in s['frames']])
validate_plan(plan);write_json(out,plan)
print(json.dumps(dict(path=str(out),sha256=digest(out),frames=sum(s['end'] for s in plan['sessions']),
    expert_labels=sum(s['expert_labels'] for s in plan['sessions']))))
