"""One-use evidence closure for the local neural-subgoal unit. No game or API."""
import json
import re
from pathlib import Path

from minecraftrl.runtime import write_json
from minecraftrl.subgoal_skills import load_catalogue
from minecraftrl.world_store import digest


def main():
    root=Path(__file__).resolve().parents[1];base=root/'results/subgoal-skills-20261010'
    destination=base/'final-verification.json'
    if destination.exists():raise FileExistsError(destination)
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    success=root/'results/goal-recovery-eval-104761-20261009T174948864354Z/run'
    failure=root/'results/goal-recovery-eval-104761-20261009T174202267743Z/run'
    good=read(success/'subgoal-audit.json');bad=read(failure/'subgoal-audit.json')
    probe=read(base/'condition-probe/report.json');game=read(base/'game-gui-fixed/work.json')
    assert game['completed'] and not game['error']
    assert good['verified'] and good['execution_valid'] and good['steps']==1315 and good['log_gain']==3
    assert good['status']=='target_reached' and good['worker_exit_code']==0
    assert bad['verified'] and not bad['execution_valid'] and bad['steps']==191 and bad['status']=='error'
    assert bad['runtime_error']=='ValueError: subgoal requires valid live server observation and closed GUI'
    for audit in (good,bad):
        assert audit['action_head_logp_max_error']==0
        assert audit['post_stop_dispatches']==audit['post_stop_inferences']==audit['held_keys_after_release']==0
        assert not audit['remaining_owned_pids'] and audit['actual_llm_calls']==audit['optimizer_updates']==0
    assert probe['completed'] and not probe['error'] and probe['conditions_change_distribution']
    assert len(probe['branches'])==4 and probe['game_actions']==probe['api_calls']==probe['optimizer_updates']==0
    tests={}
    for platform,name,count,skips in (('windows','windows-final-v2-tests.log',516,21),('wsl','wsl-final-v2-tests.log',521,9)):
        content=(base/name).read_text(encoding='utf-8-sig')
        assert re.search(r'^Ran '+str(count)+r' tests in ',content,re.M),name
        assert 'OK (skipped='+str(skips)+')' in content and 'FAILED (' not in content,name
        tests[platform]=dict(tests=count,skipped=skips,failures=0,log_sha256=digest(base/name))
    assert digest(root/'docs/research-spec.md')=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
    final_changes=[]
    for name,sha in game['frozen_sha256'].items():
        if name.startswith('src/') or name.startswith('scripts/'):
            assert digest(base/'game-gui-fixed/source-at-game'/name)==sha,name
            if digest(root/name)!=sha:final_changes.append(name)
        else:assert digest(root/name)==sha,name
    assert sorted(final_changes)==['src/minecraftrl/subgoal_control.py','src/minecraftrl/subgoal_planner.py']
    catalogue=load_catalogue(base/'embeddings/catalogue.json',root);assert len(catalogue)==3
    assert read(base/'embeddings/preparation.json')['completed']
    assert (root/'results/STOP').exists()
    assert not list(base.rglob('api-receipt.json')) and not list(base.rglob('api-response.json'))
    sources=list((root/'src/minecraftrl').glob('*subgoal*.py'))
    sources += [root/'src/minecraftrl'/n for n in ('rm_planner.py','rm_planning_control.py','planning_recovery.py','replanning.py','cli.py')]
    sources += [root/'scripts'/n for n in ('run_goal_recovery_pilot.py','audit_fresh_target_handoff.py',
        'audit_subgoal_schedule.py','prepare_subgoal_embeddings.py','probe_subgoal_conditions.py',
        'prepare_subgoal_live.py','run_authorized_subgoal_live.py','finalize_subgoal_validation.py')]
    sources += [root/'tests/test_subgoal_skills.py',root/'tests/test_subgoal_control.py']
    evidence=[success/'subgoal-audit.json',failure/'subgoal-audit.json',base/'condition-probe/report.json',
        base/'game/work.json',base/'game-gui-fixed/work.json',base/'embeddings/catalogue.json',
        base/'live-prepared-v3/manifest.json']
    result=dict(completed=True,tests=tests,api_calls=0,optimizer_updates=0,real_games=2,
        scope='Local fixed-plan and condition validation, before any separately authorized live test',
        technical_failures_before_planning=1,fixed_plan_development_successes=1,audited_transitions=1506,
        actual_llm_gameplay_validated=False,approach_skill_gameplay_actions=0,game_sources_preserved=True,assets_unchanged=True,
        post_game_source_changes=final_changes,post_game_fix='Zero-action expiry is unrun and pauses without inventing failure or retry',
        stop_latched=True,final_evaluation_seeds_used=[],
        final_source_sha256={p.relative_to(root).as_posix():digest(p) for p in sources},
        evidence_sha256={p.relative_to(root).as_posix():digest(p) for p in evidence},
        next_scope='Prepared live plan requires new approval: one API call, one development game, no retry')
    write_json(destination,result);print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
