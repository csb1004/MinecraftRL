"""Check already collected integration evidence; no gameplay or training."""
import json,re,subprocess
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/rm-refined-navigation-20261015'


def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    workflow=read(BASE/'workflow.json');audit_recovery=None
    if not workflow['completed']:
        assert workflow['active']=='audit-corner' and workflow['error']=="RuntimeError('audit-corner exited 1')"
        assert [s['label'] for s in workflow['steps']]==['game-far','audit-far','game-corner']
        resumed=(BASE/'audit-corner-resume-v2.json').exists()
        audit_recovery=read(BASE/('audit-corner-resume-v2.json' if resumed else 'audit-corner-recovery.json'))
        assert audit_recovery['completed'] and audit_recovery['exit_code']==0
        original=read(BASE/'audit-corner-recovery.json')
        assert original['workflow_sha256']==digest(BASE/'workflow.json')
        assert original['original_failed_log_sha256']==digest(BASE/'audit-corner.log')
        assert original['log_sha256']==digest(BASE/'audit-corner-retry.log')
        assert audit_recovery['log_sha256']==digest(BASE/('audit-corner-resume-v2.log' if resumed else 'audit-corner-retry.log'))
    else:assert not workflow['error']
    games=[]
    for label in ('far','corner'):
        folder=BASE/('game-'+label);run=folder/'run';work=read(folder/'work.json')
        report=read(run/'summary.json');audit=read(run/'rm-navigation-audit.json')
        assert work['completed'] and not work['error'] and audit['verified'] and audit['root_preserved']
        for key in ('actual_llm_calls','optimizer_updates','post_stop_dispatches','post_stop_inferences','held_keys_after_release'):
            assert audit[key]==0
        assert audit['remaining_owned_pids']==[]
        for name,sha in work['frozen_assets'].items():assert digest(ROOT/name)==sha
        for name,sha in audit['source_sha256'].items():assert digest(run/name)==sha
        planning=report['condition']['planning'];assert planning['navigation_input']=='motion_goal_10'
        assert planning['navigation_actor_sha256']=='e033f1ef7902102569d01459d85cdd80252c141f4e1346bdb40832502e8acaea'
        rows=read(run/'goal-recovery-rows.json');nav=[r for r in rows if 'waypoint_goal' in r]
        assert all(len(r['waypoint_goal'])==10 for r in nav)
        assert audit['action_head_logp_max_error']<2e-4
        games.append(dict(world=label,steps=audit['steps'],navigation_steps=len(nav),
            navigation=[dict(outcome=r['outcome'],diagnosis=r['assessment']['diagnosis'],steps=r['committed_steps'],
                already_satisfied=r['already_satisfied'],evidence=r['assessment']['evidence']) for r in audit['navigation_receipts']],
            subgoals=audit['subgoal_receipts'],log_gain=audit['log_gain'],status=audit['status'],
            actual_llm_calls=0,logp_error=audit['action_head_logp_max_error'],
            audit_sha256=digest(run/'rm-navigation-audit.json')))
    tests={}
    for name in ('windows-tests.log','wsl-tests.log'):
        log=(BASE/name).read_text(encoding='utf-8-sig')
        assert re.search(r'^OK(?: \(skipped=\d+\))?\s*$',log,re.M) and not re.search(r'^FAILED',log,re.M)
        tests[name]=dict(total=int(re.search(r'Ran (\d+) tests',log)[1]),sha256=digest(BASE/name))
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    cleanup=read(BASE/'process-check.json');assert cleanup['remaining_project_game_processes']==[]
    assert (ROOT/'results/STOP').read_text().startswith('RL collection ended')
    result=dict(verified=True,games=games,tests=tests,automatic_stop_retained=True,audit_recovery=audit_recovery,
        original_weights_worlds_spec_preserved=True,post_stop_actions=0,keys_remaining=0,owned_pids_remaining=[],
        scope='Refined actor in RM; privileged observations and fixed planner; no causal or LLM effectiveness claim')
    write_json(BASE/'final-verification.json',result)
    write_json(ROOT/'docs/reports/2026-10-15/rm-refined-navigation-verification.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
