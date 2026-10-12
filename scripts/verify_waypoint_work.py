"""Bind completed experiments and checks; no games, training, API, or capability promotion."""
import json,re
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];out=root/'results/waypoint-comparison-20261010/final-verification.json'
if out.exists():raise ValueError('verification already exists')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
comparison=read(out.parent/'comparison.json');sources=dict(comparison['source_sha256']);checks=[]
for folder in ('waypoint-validation-20261010','waypoint-focused-validation-20261010','waypoint-initial-validation-20261010'):
    for case in ('far','corner'):
        base=root/'results'/folder/case;run=base/'run';work=read(base/'work.json');audit=read(run/'waypoint-audit.json')
        assert work['completed'] and not work['error'] and audit['verified']
        assert audit['stop_dispatches']==audit['keys_remaining']==0 and audit['owned_pids_remaining']==[]
        for name,sha in work['frozen_assets'].items():assert digest(root/name)==sha,name
        for name in ('rl_core.py','rl_policy.py','runtime.py','rl_collect.py'):
            assert digest(root/'src/minecraftrl'/name)==digest(base/'source-at-game/src/minecraftrl'/name),name
        sources[str((base/'work.json').relative_to(root))]=digest(base/'work.json')
        checks.append(dict(path=str(base.relative_to(root)),steps=audit['steps'],action_logp_max_error=audit['action_logp_max_error'],
            root_success=audit['root_success'],route_completed=audit['route_completed']))

expected={
    'docs/research-spec.md':'4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad',
    'results/onset-coverage-features-20261009/features.pt':'152953150fb427bd10a8ad924e76ad9a368e1bf794771b30f8d878243cd87270',
    'results/waypoint-features-20261010/features.pt':'cceccbfdf31fad9180f29cee576a37b68ad66926e84747973ca97767dace3a13',
}
for name,sha in expected.items():assert digest(root/name)==sha; sources[name]=sha
tests={}
for name,label,count,skip in [('windows-final-tests.log','windows_full',538,21),('wsl-initial-tests.log','wsl_affected',13,0)]:
    p=root/'results/waypoint-training-goal-focused-20261010'/name;log=p.read_text(encoding='utf-8-sig')
    assert re.search(r'Ran '+str(count)+r' tests',log) and '\nOK' in log
    if skip:assert f'OK (skipped={skip})' in log
    else:assert '\nOK\n' in log
    tests[label]=dict(total=count,skipped=skip,passed=count-skip,log=str(p.relative_to(root)))
    sources[str(p.relative_to(root))]=digest(p)
for name in ('report.json',):
    p=root/'results/waypoint-runtime-v2-20261010'/name;assert read(p)['completed'];sources[str(p.relative_to(root))]=digest(p)
for pattern in ('src/minecraftrl/waypoint*.py','tests/test_waypoint*.py','scripts/*waypoint*.py','docs/waypoint*.md'):
    for p in root.glob(pattern):sources[str(p.relative_to(root))]=digest(p)
for name in ('src/minecraftrl/cli.py','docs/progress.md','docs/next-task-units.md'):
    sources[name]=digest(root/name)
for name in ('comparison.json','actual-routes-v2.png','plot-sources-v2.json'):sources[str((out.parent/name).relative_to(root))]=digest(out.parent/name)
stop=root/'results/STOP';assert stop.exists();stop_text=stop.read_text()
assert stop_text.startswith('RL collection ended;')
result=dict(completed=True,checks=checks,total_actual_transitions=sum(r['steps'] for r in checks),tests=tests,
    new_api_calls=0,new_ppo_updates=0,bc_updates=528+1056,final_evaluation_seeds_used=0,default_policy_promoted=False,
    llm_executable_catalogue_extended=False,stop_text=stop_text,preserved_assets_verified=True,
    runtime_source_sha256=sources,
    scope='Known development fixtures and privileged pose. Initial-trigger diagnostics are separate from stall recovery; uncalled skills are unrun.')
write_json(out,result);print(json.dumps({k:v for k,v in result.items() if k!='runtime_source_sha256'}))
