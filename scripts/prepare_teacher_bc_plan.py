"""Single-use immutable admission manifest for the three local teacher sessions."""
import json
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from minecraftrl.bc_core import validate_plan

root=Path(__file__).resolve().parents[1];output=root/'configs/teacher-bc-20261008.json'
if output.exists():raise FileExistsError(output)
embed='results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
plan=dict(data_source='local_teacher',rm_context='recorded',goal='collect wood',frame_offset=0,
          embedding=embed,embedding_sha256=digest(root/embed),
          parent='results/recovery-gap-20261007/train-gap-epochs2/checkpoint.pt',sessions=[])
runs=[('train',104743,'rl-train-104743-20261008T024538270180Z'),
      ('train',104759,'rl-train-104759-20261008T024849167920Z'),
      ('development',104761,'rl-eval-104761-20261008T025814831606Z')]
for split,seed,name in runs:
    relative='results/'+name+'/run';run=root/relative
    summary=json.loads((run/'summary.json').read_text());rows=json.loads((run/'teacher-rows.json').read_text())
    assert summary['demonstration_candidate'] and summary['worker_exit_code']==0
    assert summary['controller']=='privileged_scripted_correction_teacher' and not summary['rl_training_eligible']
    assert summary['initial_world']['experimental_split']==split
    assert not summary['process_cleanup']['remaining_live_pids']
    release=json.loads((run/'input-release-state.json').read_text());assert release['verified'] and release['held_key_count']==0
    assert rows and rows[-1]['terminal'] and not any(row['terminal'] for row in rows[:-1])
    before=rows[0]['server_before']['server_inventory'];after=rows[-1]['server_after']['server_inventory']
    assert after.get('oak_log',0)-before.get('oak_log',0)>=1
    plan['sessions'].append(dict(id=name,split=split,seed=seed,run=relative,context_start=0,start=0,end=len(rows),
                                rows_sha256=digest(run/'teacher-rows.json'),events_sha256=digest(run/'events.jsonl'),
                                frame_sha256=[digest(run/row['frame']) for row in rows]))
validate_plan(plan);write_json(output,plan)
print(json.dumps([dict(id=s['id'],split=s['split'],rows=s['end']) for s in plan['sessions']]))
