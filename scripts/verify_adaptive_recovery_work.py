"""Bind real game evidence, bounded paid receipts and post-batch regression checks."""
import json,re
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    root=Path(__file__).resolve().parents[1];base=root/'results/adaptive-recovery-20261014';out=base/'final-verification.json'
    if out.exists():raise FileExistsError('verification exists')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    comparison=read(root/'docs/reports/2026-10-14/recovery-summary.json');index=read(base/'completed-trials.json')
    assert len(index)==6;sources=dict(comparison['source_sha256']);checks=[]
    for key,relative in index.items():
        run=base/relative;folder=run.parent;work=read(folder/'work.json');audit=read(run/'waypoint-audit.json')
        assert work['completed'] and not work['error'] and audit['verified']
        assert audit['stop_dispatches']==audit['keys_remaining']==0 and audit['owned_pids_remaining']==[]
        for name,sha in audit['source_sha256'].items():
            assert digest(run/name)==sha,name;sources[(run/name).relative_to(root).as_posix()]=sha
        for name,sha in work['frozen_assets'].items():assert digest(root/name)==sha,name
        for name in ('rl_core.py','rl_policy.py','runtime.py','rl_collect.py','cli.py','exploration_route.py'):
            assert digest(root/'src/minecraftrl'/name)==digest(folder/'source-at-game/src/minecraftrl'/name),name
        checks.append(dict(trial=key,steps=audit['steps'],action_logp_max_error=audit['action_logp_max_error'],
            root_success=work['root_success'],recovery_evaluated=work['recovery_called'],actual_llm_calls=work['api_calls']))
        sources[(folder/'work.json').relative_to(root).as_posix()]=digest(folder/'work.json')
    expected={
        'docs/research-spec.md':'4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad',
        'results/onset-coverage-features-20261009/features.pt':'152953150fb427bd10a8ad924e76ad9a368e1bf794771b30f8d878243cd87270',
        'results/waypoint-features-20261010/features.pt':'cceccbfdf31fad9180f29cee576a37b68ad66926e84747973ca97767dace3a13',
        'results/diverse-teacher-worlds-20261008/development-far.zip':'7408a4a201df7706c39b9666e35f555633081825d64c566639bf30d03ab02cce',
        'results/diverse-teacher-worlds-20261008/development-corner.zip':'2851b755a45623f97ef800ac844b66e0c22964481352de66f1fae4220a0d7eef'}
    for name,sha in expected.items():assert digest(root/name)==sha,name;sources[name]=sha
    tests={}
    for name,label,count,skip in [('windows-final-tests.log','windows_full',566,21),('wsl-final-tests.log','wsl_affected',28,0)]:
        path=base/name;log=path.read_text(encoding='utf-8-sig')
        assert re.search(r'Ran '+str(count)+r' tests',log) and ('OK (skipped='+str(skip)+')' if skip else '\nOK\n') in log,name
        tests[label]=dict(total=count,skipped=skip,passed=count-skip,log=path.relative_to(root).as_posix())
        sources[path.relative_to(root).as_posix()]=digest(path)
    old=read(base/'far/visit_memory/run/summary.json');assert old['status']=='blocked' and old['steps']==0
    replay_path=base/'route-history-replay.json';replay=read(replay_path)
    assert replay['verified'] and replay['original_unchanged'] and replay['new_api_calls']==replay['new_game_actions']==0
    assert len(replay['genuine_navigation_failures'])==1 and len(replay['interrupted_candidates'])==2
    for name,sha in replay['source_sha256'].items():assert digest(base/'far/adaptive_llm/run'/name)==sha
    sources[replay_path.relative_to(root).as_posix()]=digest(replay_path)
    budget=read(base/'paid-budget.json') if (base/'paid-budget.json').exists() else dict(calls=[],reserved_micro_usd=0)
    assert len(budget['calls'])==comparison['api_calls']<=4 and budget['reserved_micro_usd']<=1000000
    paid_receipts=[]
    for relative in index.values():
        for path in (base/relative/'candidate-exchange').glob('request-*/api-receipt.json'):
            paid_receipts.append(read(path));sources[path.relative_to(root).as_posix()]=digest(path)
    assert sum(r['network_calls'] for r in paid_receipts)==comparison['api_calls']
    assert abs(sum(r.get('estimated_usd') or 0 for r in paid_receipts)-comparison['estimated_usd'])<1e-10
    for name in ('paid-budget.json','authorization.json','batch-audit.log','interruption-red.log','interruption-green.log'):
        sources[(base/name).relative_to(root).as_posix()]=digest(base/name)
    for pattern in ('src/minecraftrl/*adaptive*.py','src/minecraftrl/navigation_choice.py','src/minecraftrl/preflight.py',
            'src/minecraftrl/waypoint_control.py','tests/test_*adaptive*.py','tests/test_navigation_choice.py','tests/test_route_interruption.py',
            'tests/test_preflight_timeout.py','scripts/*adaptive_recovery*.py','scripts/run_navigation_choice_broker.py','scripts/audit_waypoint_run.py',
            'scripts/replay_route_history_fix.py','results/adaptive-recovery-20261014/source-at-audit/*.py',
            'docs/adaptive-recovery*.md','docs/reports/2026-10-14/*'):
        for p in root.glob(pattern):sources[p.relative_to(root).as_posix()]=digest(p)
    stop=(root/'results/STOP').read_text();assert stop.startswith('RL collection ended;')
    result=dict(completed=True,checks=checks,total_actual_transitions=sum(r['steps'] for r in checks),tests=tests,
        new_api_calls=comparison['api_calls'],reserved_usd=comparison['reserved_usd'],estimated_usd=comparison['estimated_usd'],
        optimizer_updates=0,final_evaluation_seeds_used=0,default_policy_promoted=False,stop_text=stop,
        preserved_assets_verified=True,post_batch_history_fix='unit/replay verified; six games used archived v1 source',
        history_replay=dict(archived_steps=replay['archived_steps'],navigation_failures=1,interrupted_candidates=2,
            new_api_calls=0,new_game_actions=0),
        source_sha256=sources,scope=comparison['scope'])
    write_json(out,result);write_json(root/'docs/reports/2026-10-14/verification.json',{k:v for k,v in result.items() if k!='source_sha256'})
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
