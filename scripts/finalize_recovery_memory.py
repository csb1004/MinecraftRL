"""Freeze completed memory integration checks without rerunning a trial."""
import json
import re
import shutil
from pathlib import Path

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root=Path(__file__).resolve().parents[1];out=root/'results/recovery-memory-20261010'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
destination=out/'final-verification.json'
if destination.exists():raise FileExistsError(destination)
work_path=root/'results/recovery-memory-game-20261010/work.json';work=read(work_path)
assert work['completed'] and work['error'] is None and work['sources_unchanged']
assert work['actual_games']==1 and work['api_calls']==work['optimizer_updates']==0
hashes={}
def checked(path,expected=None):
    path=Path(path);path=path if path.is_absolute() else root/path
    key=path.relative_to(root).as_posix()
    if key not in hashes:hashes[key]=digest(path)
    if expected is not None:assert hashes[key]==expected,key
    return hashes[key]
post_game_changes=[]
for name,expected in work['frozen_sha256'].items():
    if digest(root/name)!=expected:
        assert name.replace('\\','/')=='src/minecraftrl/replanning.py'
        checked(out/'source-at-game'/name,expected);post_game_changes.append(name)
    else:checked(name,expected)
checked(work_path)
run=Path(work['run']);audit=read(run/'recovery-memory-audit.json')
assert audit['verified'] and audit['memory_packet_verified'] and audit['execution_receipts']==1
assert audit['post_stop_dispatches']==audit['post_stop_inferences']==audit['held_keys_after_release']==0
assert not audit['remaining_owned_pids']
for field in ('source_sha256','memory_source_sha256'):
    for name,expected in audit[field].items():checked(run/name,expected)
checked(run/'recovery-memory-audit.json')
replay=read(out/'replay/report.json');assert replay['verified'] and len(replay['rows'])==2
for row in replay['rows']:
    assert row['next_available_choices']==['remain_paused'] and row['model_calls']==row['new_actions']==0
    for name,expected in row['source_sha256'].items():checked(root/row['source_run']/name,expected)
for name,expected in replay['source_sha256'].items():checked(out/'replay/source'/name,expected)
tests={}
for name,count,skip in (('windows-complete-tests.log',456,19),('wsl-complete-tests.log',461,9)):
    path=out/name;log=path.read_text(encoding='utf-8-sig')
    assert re.search(rf'Ran {count} tests in ',log)
    assert ('OK (skipped='+str(skip)+')' if skip else '\nOK') in log and 'FAILED (' not in log
    tests[name]=dict(total=count,skipped=skip);checked(path)
processes=read(out/'process-final.json')
assert not processes['windows_project_processes']
assert not any(re.search(r'\b(?:python\S*|java\S*|Xvfb|micromamba)$',s.strip()) for s in processes['wsl_ps'])
assert processes['stop_present'] and (root/'results/STOP').exists()
checked(out/'process-final.json');checked('results/STOP')
checked('docs/research-spec.md','4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad')
files={p.relative_to(out/'source-before') for p in (out/'source-before').rglob('*') if p.is_file()}
files.update(Path(n) for n in ('tests/test_planning_memory.py','scripts/replay_recovery_memory.py',
    'scripts/audit_recovery_memory.py','scripts/run_recovery_memory_validation.py',
    'scripts/finalize_recovery_memory.py','docs/recovery-memory-protocol-2026-10-10.md',
    'docs/recovery-memory-validation-2026-10-10.md','docs/llm-high-level-planner-2026-10-07.md'))
snapshot=out/'source-final';snapshot.mkdir(exist_ok=False);changes=[]
for name in sorted(files):
    target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    after=checked(name);assert digest(target)==after
    before=out/'source-before'/name
    changes.append(dict(path=name.as_posix(),before_sha256=digest(before) if before.exists() else None,after_sha256=after))
write_json(out/'source-diff.json',changes)
for path in (out/'replay').rglob('*.json'):checked(path)
result=dict(completed=True,actual_games=1,api_calls=0,optimizer_updates=0,
    memory_packet_verified=True,execution_receipts=audit['execution_receipts'],outcomes=audit['outcomes'],
    offline_next_requests=2,duplicate_methods_blocked=2,actual_multi_call_llm_recovery=False,
    status=work['status'],log_gain=work['log_gain'],audited_transitions=audit['steps'],tests=tests,
    stop_latched=True,remaining_owned_pids=[],post_stop_dispatches=0,held_keys=0,
    default_policy_promotions=0,final_evaluation_seeds_used=[],remaining_authorized_paid_calls=0,
    scope='Memory/duplicate integration verified; fixed local response, no learned detour or LLM performance claim.',sha256=hashes)
result['post_game_review_changes']=post_game_changes
write_json(destination,result)
print(json.dumps({k:v for k,v in result.items() if k!='sha256'}))
