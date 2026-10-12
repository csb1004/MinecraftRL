"""Check preserved evidence and write the bounded RM-planner unit's final record."""
import json
from pathlib import Path
import re
from datetime import datetime, timezone

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/rm-planner-20261010'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    destination=BASE/'final-verification.json'
    if destination.exists():raise SystemExit('Final unit already recorded; inspect instead of rerunning.')
    tests={}
    for label,name in [('windows','windows-final-all-tests.log'),('wsl','wsl-final-all-tests.log')]:
        log=BASE/name;text=log.read_text(encoding='utf-8')
        count=re.search(r'Ran (\d+) tests in ([\d.]+)s',text)
        status=re.search(r'^OK \(skipped=(\d+)\)',text,re.MULTILINE)
        assert count and status and 'FAILED (' not in text,(label,'incomplete test log')
        tests[label]=dict(tests=int(count[1]),skipped=int(status[1]),elapsed_s=float(count[2]),
                          path=str(log.relative_to(ROOT)),sha256=digest(log))
        assert tests[label]['tests']>=499
    plan_run=ROOT/'results/goal-recovery-eval-104761-20261009T162019751448Z/run'
    incomplete=ROOT/'results/goal-recovery-eval-104761-20261009T162733827875Z/run'
    unexercised=ROOT/'results/goal-recovery-eval-104761-20261009T163805995692Z/run'
    race_run=ROOT/'results/goal-recovery-eval-104761-20261009T165615385137Z/run'
    cancel_batch=ROOT/'results/rm-planner-cancel-race-fixed-20261010'
    work=read(cancel_batch/'work.json');assert work['completed'] and work['sources_unchanged'] and not work['error']
    cancel=Path(work['run'])
    audits=[]
    for run,expected in [(plan_run,'target_reached'),(unexercised,'planning_paused'),(cancel,'cancelled')]:
        audit=read(run/'rm-plan-audit.json');assert audit['verified'] and audit['status']==expected
        for field in ('actual_llm_calls','optimizer_updates','post_stop_dispatches','post_stop_inferences',
                      'held_keys_after_release','action_head_logp_max_error'):
            assert audit[field]==0,(run,field)
        assert not audit['remaining_owned_pids'] and audit['root_preserved']
        for name,sha in audit['source_sha256'].items():assert digest(run/name)==sha
        audits.append(dict(run=str(run.relative_to(ROOT)),status=expected,steps=audit['steps'],
            log_gain=audit['log_gain'],planner_calls=audit['planner_calls'],sha256=digest(run/'rm-plan-audit.json')))
    assert audits[0]['steps']==1004 and audits[0]['log_gain']==1 and audits[0]['planner_calls']==1
    assert audits[1]['steps']==1083 and audits[1]['planner_calls']==0
    broken=read(incomplete/'summary.json');assert broken['worker_exit_code']==-9 and broken['status']=='cancelled'
    cancellation=read(cancel/'summary.json')
    assert cancellation['rl_cancelled_archive'] and not cancellation['rl_archive_timed_out']
    assert cancellation['worker_exit_code']==2 and not cancellation['rl_training_eligible']
    assert read(cancel_batch/'controlled-stop.json')['requested']
    source_copies={}
    for folder in ('rm-planner-game-20261010','rm-planner-stop-fix-20261010',
                   'rm-planner-cancel-fixed-20261010','rm-planner-cancel-race-fixed-20261010'):
        directory=ROOT/'results'/folder;record=read(directory/'work.json');checked=0
        for name,sha in record['frozen_sha256'].items():
            if name.replace('\\','/').startswith(('src/','scripts/')):
                assert digest(directory/'source-at-game'/name)==sha,(folder,name)
                checked+=1
        source_copies[folder]=dict(verified_files=checked,work_sha256=digest(directory/'work.json'),
                                 completed=record['completed'],error=record['error'])
    for name,sha in work['frozen_sha256'].items():assert digest(ROOT/name)==sha,('final source changed',name)
    paid=BASE/'api-prepared-v2/paid-probe';api=read(paid/'offline-revalidation.json')
    receipt=read(paid/'receipt.json')
    assert receipt['network_calls']==1 and receipt['status']=='failed' and not receipt['completed']
    assert api['offline_revalidation_valid'] and not api['initial_contract_valid']
    assert api['retries']==api['paid_authorization_remaining']==api['new_game_actions']==0
    assert len(api['unsupported_skills'])==3
    for name,sha in api['source_sha256'].items():assert digest(ROOT/name)==sha,name
    expected_assets={
        'docs/research-spec.md':'4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad',
        'results/onset-coverage-training-20261009/checkpoint.pt':'473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd',
        'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy':'810512d05a7f228d77aed45f29a21277156f934408e2387f2ae3fd6f4217be40',
        'results/goal-conditioning-diagnosis-20261009/explore_9711.npy':'f8839fa46c254dbb7d8834b97d59afd30b995c53b671d22ae64b1ba9ea397d55',
        'results/diverse-teacher-worlds-20261008/development-corner.zip':'2851b755a45623f97ef800ac844b66e0c22964481352de66f1fae4220a0d7eef'}
    for name,sha in expected_assets.items():assert digest(ROOT/name)==sha,name
    assert (ROOT/'results/STOP').exists()
    report=dict(completed=True,recorded_at=datetime.now(timezone.utc).isoformat(),tests=tests,
        actual_game_runs=5,audited_runs=audits,audited_transitions=sum(a['steps'] for a in audits),
        incomplete_artifact_run=str(incomplete.relative_to(ROOT)),incomplete_steps=broken['steps'],
        unexercised_stop_trial=str(unexercised.relative_to(ROOT)),
        cancelled_guard_error_run=str(race_run.relative_to(ROOT)),
        cancelled_guard_error_forensics_sha256=digest(BASE/'cancel-race-forensics.json'),
        cancellation_trigger=read(cancel_batch/'controlled-stop.json')['trigger']['event'],
        cancellation_saved=True,cancelled_training_eligible=False,source_snapshots=source_copies,
        actual_llm_calls=1,api_retries=0,paid_authorization_remaining=0,llm_plan_game_actions=0,
        initial_api_contract_valid=False,offline_revalidation_valid=True,api_usage=api['usage'],
        api_evidence_sha256=digest(paid/'offline-revalidation.json'),
        optimizer_updates=0,heldout_seeds_used=[],default_policy_promotions=0,colab_connections=0,
        stop_latched=True,assets_sha256=expected_assets,
        final_source_sha256=work['frozen_sha256'],
        final_broker_entrypoint_sha256=digest(ROOT/'scripts/run_recovery_broker.py'),
        final_broker_tests_sha256=digest(ROOT/'tests/test_rm_broker_entrypoint.py'),
        scope='Bounded root/subgoal integration, one archived LLM proposal, and cancellation storage repair; no general recovery or causal learning improvement claim')
    write_json(destination,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('final_source_sha256','assets_sha256','source_snapshots')}))


if __name__=='__main__':main()
