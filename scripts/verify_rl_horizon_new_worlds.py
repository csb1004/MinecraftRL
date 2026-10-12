"""Final evidence checks for the fixed October 7 development comparison."""
import datetime,json,re
from pathlib import Path
from minecraftrl.world_store import digest
from minecraftrl.runtime import write_json

root=Path(__file__).resolve().parents[1]
base=root/'results/rl-horizon-new-worlds-20261007'
state=json.loads((base/'experiment.json').read_text())
audit=json.loads((base/'audit.json').read_text())
diagnosis=json.loads((base/'action-diagnosis/diagnosis.json').read_text())
assert diagnosis['completed'] and diagnosis['replayed_actions']==audit['actual_actions']
assert diagnosis['max_probability_error']<2e-4 and diagnosis['environment_actions']==diagnosis['optimizer_updates']==0
failures=json.loads((base/'failure-diagnosis.json').read_text())
assert failures['completed'] and len(failures['runs'])==sum(not r['report']['server_target_reached'] for r in state['runs'])
assert failures['environment_actions']==failures['optimizer_updates']==0
assert state['completed'] and not state['errors'] and not state['unrun'] and audit['verified']
assert audit['actual_evaluations']==30 and audit['optimizer_updates']==0
assert len(state['preparations'])==5 and len(state['comparisons'])==25
assert state['final_seeds_unrun']==[3000017,3000029,3000047,3000061,3000073]
for item in state['checkpoints'].values():assert digest(item['path'])==item['sha256']
assert digest(root/'docs/research-spec.md')=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
baseline=json.loads((base/'runtime-source-baseline.json').read_text())
for path,sha in baseline.items():assert digest(root/path)==sha,path
prior=json.loads((root/'results/rl-horizon-20261007/final-verification.json').read_text())
changed={'src/minecraftrl/world_save_patch.py','src/minecraftrl/rl_worlds.py','tests/test_rl_batch.py'}
for path,sha in {**prior['preserved_sources'],**prior['new_source_sha256']}.items():
    if path.replace('\\','/') not in changed:assert digest(root/path)==sha,path
original=json.loads((root/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
for item in original['training']['trajectories']:
    path=str(item['path']).replace('/mnt/d/DATA/Desktop/MinecraftRL/',str(root)+'/')
    assert digest(path)==item['sha256']
assert digest(root/'results/rl-expanded-20261006T095005817265Z/training/checkpoint.pt')=='299c37fd9fbbbea3bff5c8862e75486546a3ecd66dc35193f87f98fe44e60049'
tests={}
for platform in ('windows','wsl'):
    path=base/f'tests-{platform}.txt';log=path.read_text(encoding='utf-8')
    count=re.search(r'Ran (\d+) tests in',log);skip=re.search(r'^OK \(skipped=(\d+)\)',log,re.M)
    assert count and re.search(r'^OK(?: \(skipped=\d+\))?\s*$',log,re.M)
    tests[platform]=dict(run=int(count[1]),skipped=int(skip[1]) if skip else 0,sha256=digest(path))
wsl=json.loads((base/'process-check-wsl.json').read_text())
windows=json.loads((base/'process-check-windows.json').read_text(encoding='utf-8-sig'))
assert not wsl['remaining_project_runtime_processes'] and not windows['project_processes']
assert (root/'results/STOP').exists()
write_json(base/'final-verification.json',dict(verified=True,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    actual_evaluations=30,optimizer_updates=0,tests=tests,preserved_training_trajectories=len(original['training']['trajectories']),
    runtime_source_sha256=baseline,research_spec_sha256=digest(root/'docs/research-spec.md'),
    changed_and_new_source_sha256={p:digest(root/p) for p in sorted(changed|{
        'src/minecraftrl/rl_validation.py','tests/test_world_identity.py','tests/test_rl_validation.py',
        'scripts/run_rl_horizon_new_worlds.py','scripts/audit_rl_horizon_new_worlds.py',
        'scripts/diagnose_rl_new_world_actions.py','scripts/diagnose_rl_new_world_failures.py',
        'scripts/verify_rl_horizon_new_worlds.py'})},
    audit_sha256=digest(base/'audit.json'),checkpoints=state['checkpoints'],
    action_diagnosis_sha256=digest(base/'action-diagnosis/diagnosis.json'),replayed_actions=diagnosis['replayed_actions'],
    failure_diagnosis_sha256=digest(base/'failure-diagnosis.json'),
    stop_latched=True,wsl_process_check=wsl,windows_process_check=windows,
    default_policy_promoted=False,final_seeds_unrun=state['final_seeds_unrun']))
print('Verified completed comparison, preserved originals, tests, STOP and process cleanup.')
