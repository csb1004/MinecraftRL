"""Bind actual trials, frozen assets and checks; raw files stay in local results."""
import json,re
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];base=root/'results/blind-recovery-20261012'
out=base/'final-verification.json'
if out.exists():raise ValueError('one-use verification already exists')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
summary=root/'docs/reports/2026-10-12/recovery-summary.json';comparison=read(summary)
sources=dict(comparison['source_sha256']);checks=[]
for case in ('far','corner'):
    for planner in ('fixture','visit_memory'):
        folder=base/case/planner;run=folder/'run';work=read(folder/'work.json');audit=read(run/'waypoint-audit.json')
        assert work['completed'] and not work['error'] and audit['verified']
        assert audit['stop_dispatches']==audit['keys_remaining']==0 and audit['owned_pids_remaining']==[]
        for name,sha in work['frozen_assets'].items():assert digest(root/name)==sha,name
        for name in ('rl_core.py','rl_policy.py','runtime.py','rl_collect.py','cli.py','exploration_route.py'):
            assert digest(root/'src/minecraftrl'/name)==digest(folder/'source-at-game/src/minecraftrl'/name),name
        # Only the module description changed after the four games.
        current=(root/'src/minecraftrl/waypoint_control.py').read_bytes().splitlines()[1:]
        recorded=(folder/'source-at-game/src/minecraftrl/waypoint_control.py').read_bytes().splitlines()[1:]
        assert current==recorded
        sources[(folder/'work.json').relative_to(root).as_posix()]=digest(folder/'work.json')
        checks.append(dict(case=case,planner=planner,steps=audit['steps'],action_logp_max_error=audit['action_logp_max_error'],
            root_success=work['root_success'],recovery_evaluated=work['recovery_called']))
expected={
    'docs/research-spec.md':'4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad',
    'results/onset-coverage-features-20261009/features.pt':'152953150fb427bd10a8ad924e76ad9a368e1bf794771b30f8d878243cd87270',
    'results/waypoint-features-20261010/features.pt':'cceccbfdf31fad9180f29cee576a37b68ad66926e84747973ca97767dace3a13',
    'results/diverse-teacher-worlds-20261008/development-far.zip':'7408a4a201df7706c39b9666e35f555633081825d64c566639bf30d03ab02cce',
    'results/diverse-teacher-worlds-20261008/development-corner.zip':'2851b755a45623f97ef800ac844b66e0c22964481352de66f1fae4220a0d7eef',
}
for name,sha in expected.items():assert digest(root/name)==sha,name;sources[name]=sha
tests={}
for name,label,count,skip in [('windows-full-tests-workspace-temp.log','windows_full',548,21),
        ('windows-final-metric-tests.log','windows_changed_metric',2,0),('wsl-final-tests.log','wsl_affected',23,0)]:
    p=base/name;log=p.read_text(encoding='utf-8-sig')
    assert re.search(r'Ran '+str(count)+r' tests',log) and '\nOK' in log,name
    if skip:assert f'OK (skipped={skip})' in log
    else:assert '\nOK\n' in log
    tests[label]=dict(total=count,skipped=skip,passed=count-skip,log=p.relative_to(root).as_posix())
    sources[p.relative_to(root).as_posix()]=digest(p)
for name in ('windows-full-tests.log','far/fixture/audit-console.log','world-preparation.log'):
    p=base/name
    if p.exists():sources[p.relative_to(root).as_posix()]=digest(p)
for pattern in ('src/minecraftrl/exploration_route.py','src/minecraftrl/waypoint*.py','tests/test_*recovery_metrics.py',
        'scripts/*blind_recovery*.py','scripts/audit_waypoint_run.py','docs/blind-recovery*.md','docs/reports/2026-10-12/*'):
    for p in root.glob(pattern):sources[p.relative_to(root).as_posix()]=digest(p)
stop=(root/'results/STOP').read_text();assert stop.startswith('RL collection ended;')
result=dict(completed=True,checks=checks,total_actual_transitions=sum(r['steps'] for r in checks),tests=tests,
    new_api_calls=0,optimizer_updates=0,final_evaluation_seeds_used=0,default_policy_promoted=False,
    llm_executable_catalogue_extended=False,stop_text=stop,preserved_assets_verified=True,source_sha256=sources,
    scope=comparison['scope'])
write_json(out,result)
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))
