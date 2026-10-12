"""Final read-only provenance and denominators for the completed local work."""
import json
import re
from pathlib import Path

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root=Path(__file__).resolve().parents[1];base=root/'results/goal-recovery-comparison-20261009'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
out=base/'final-verification.json'
if out.exists():raise FileExistsError(out)
state=read(base/'experiment.json')
assert state['completed'] and not state['errors'] and state['active'] is None
assert len(state['runs'])==state['planned_conditions']==6 and not state['unrun']
assert state['optimizer_updates']==state['llm_calls']==state['colab_connections']==0 and not state['default_promoted']
for path,sha in state['frozen'].items():assert digest(path)==sha,path
for world in state['worlds'].values():assert digest(world['path'])==world['sha256']
assert digest(root/'docs/research-spec.md')=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
rows=[]
for trial in state['runs']:
    run=Path(trial['path']);report=read(run/'summary.json');audit=read(run/'goal-recovery-audit.json')
    assert report==trial['report'] and audit==trial['audit'] and trial['audited'] and audit['verified']
    for name,sha in audit['source_sha256'].items():assert digest(run/name)==sha,(trial['label'],name)
    assert audit['post_stop_dispatches']==audit['post_stop_inferences']==audit['released_held_keys']==0
    assert not audit['remaining_owned_pids'] and audit['action_head_logp_max_error']<=2e-4
    assert report['seed'] not in (3000017,3000029,3000047,3000061,3000073)
    evidence=read(base/'planner-evidence'/(trial['label']+'.json'))
    assert evidence['actual_llm_calls']==evidence['new_game_actions']==0
    assert evidence['actual_log_gain']==audit['log_gain']
    rows.append(dict(label=trial['label'],steps=audit['steps'],wood=audit['log_gain'],
        intervention_steps=audit['intervention_steps'],first_wall_crossing=audit['first_wall_crossing'],
        failure_or_handoff=evidence['attempt_assessment']['classification'],
        action_head_logp_max_error=audit['action_head_logp_max_error']))
    print('verified '+trial['label'],flush=True)
technical=state['technical_attempts'];assert len(technical)==1
for attempt in technical:
    assert not attempt['counts_in_performance'] and attempt['report']['steps']==0
    assert attempt['report']['inference_ms']['count']==0 and not attempt['report']['process_cleanup']['remaining_live_pids']
cancel=read(root/'results/goal-recovery-stop-20261009/work.json')
assert cancel['completed'] and cancel['error'] is None and not cancel['performance_sample']
assert cancel['post_acknowledgment_policy_dispatches']==cancel['post_acknowledgment_inferences']==cancel['held_keys_after_release']==0
assert not cancel['remaining_owned_pids']
for name,sha in cancel['evidence_sha256'].items():assert digest(Path(cancel['run'])/name)==sha
serialization=read(root/'results/goal-recovery-stop-20261009/serialization-audit.json')
assert serialization['verified'] and not serialization['training_eligible']
assert serialization['complete_trajectory_rows']==cancel['committed_steps']
for name,sha in serialization['sha256'].items():assert digest(Path(cancel['run'])/name)==sha
prepared=read(root/'results/recovery-draft-preparation-20261009/manifest.json')
assert prepared['actual_api_calls']==0 and prepared['maximum_calls']==2 and not prepared['paid_approval_received']
assert not (root/'results/recovery-draft-preparation-20261009/paid-probe').exists()
tests={}
for platform in ('windows','wsl'):
    path=root/f'results/goal-recovery-implementation-20261009/{platform}-tests-final.txt'
    text=path.read_text(encoding='utf-8',errors='replace')
    ran=re.search(r'^Ran (\d+) tests',text,re.M);ok=re.search(r'^OK(?: \(skipped=(\d+)\))?$',text,re.M)
    assert ran and ok and not re.search(r'^FAILED',text,re.M),platform
    tests[platform]=dict(ran=int(ran[1]),skipped=int(ok[1] or 0),sha256=digest(path))
assert (root/'results/STOP').exists()
result=dict(verified=True,performance_games=6,rows=rows,technical_startup_failures=1,
    separate_stop_trials=1,performance_transitions=sum(r['steps'] for r in rows),
    cancellation_worker_exit_code=cancel['worker_exit_code'],cancellation_serialization_verified=True,
    max_action_head_logp_error=max(r['action_head_logp_max_error'] for r in rows),
    optimizer_updates=0,actual_llm_calls=0,colab_connections=0,default_promoted=False,
    full_suites=tests,research_spec_unchanged=True,reserved_final_seeds_untouched=True,
    stop_latched=True,stop_sha256=digest(root/'results/STOP'),
    experiment_sha256=digest(base/'experiment.json'),cancellation_sha256=digest(root/'results/goal-recovery-stop-20261009/work.json'),
    prepared_request_sha256=digest(root/'results/recovery-draft-preparation-20261009/manifest.json'),
    scope='Local development evidence and provenance; not final success rate or learned improvement claim')
write_json(out,result);print(json.dumps(result))
