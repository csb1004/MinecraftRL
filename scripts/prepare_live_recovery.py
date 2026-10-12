"""Freeze a reviewable one-call experiment after all local integration audits pass."""
import json
from pathlib import Path

from minecraftrl.recovery_live import LIVE_PROMPT
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    root=Path(__file__).resolve().parents[1]
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    impl=root/'results/planning-recovery-implementation-20261009'
    batches=[root/'results'/name/'work.json' for name in (
        'planning-recovery-validation-20261009','planning-recovery-validation-20261009-handoff',
        'planning-recovery-validation-20261009-key-count-fix','planning-recovery-mailbox-game-20261009')]
    states=[read(p) for p in batches]
    assert states[2]['completed'] and states[3]['completed']
    assert states[3]['broker_result']['status']=='delivered' and states[3]['broker_exit']==0
    runs=[Path(row['run']) for state in states[:3] for row in state['runs'] if row.get('run')]
    runs.append(Path(states[3]['run']))
    assert len(runs)==5
    audits=[read(run/'planning-recovery-audit.json') for run in runs]
    assert all(a['verified'] and a['actual_llm_calls']==a['optimizer_updates']==0 for a in audits)
    assert all(a['post_stop_dispatches']==a['post_stop_inferences']==a['held_keys_after_release']==0
               and not a['remaining_owned_pids'] for a in audits)
    assert audits[-1]['handoff_exercised'] and audits[-1]['broker_status']=='delivered'
    for relative,expected in states[3]['sources'].items():
        assert digest(root/relative)==expected,relative
    test_logs=['windows-mailbox-final-tests.txt','wsl-final-tests.txt','preparation-tests.txt','wsl-preparation-tests.txt']
    for name in test_logs:
        text=(impl/name).read_text(encoding='utf-8-sig')
        assert '\nOK' in text and '\nFAILED' not in text,name
    stop=root/'results/STOP'
    assert stop.read_text(encoding='utf-8').startswith(('RL collection ended','RL training ended'))
    spec=root/'docs/research-spec.md'
    assert digest(spec)=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
    checkpoint=root/'results/onset-coverage-training-20261009/checkpoint.pt'
    assert digest(checkpoint)=='473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd'
    folder=root/'results/planning-recovery-live-prepared-20261009';folder.mkdir(exist_ok=False)
    (folder/'prompt.txt').write_text(LIVE_PROMPT,encoding='utf-8')
    files=list((root/'src').rglob('*.py'))+[root/name for name in (
        'scripts/run_planner_exchange_pilot.py','scripts/run_goal_recovery_pilot.py',
        'scripts/run_recovery_broker.py','scripts/audit_planning_recovery.py','scripts/run_rl_batch.py',
        'scripts/prepare_live_recovery.py','configs/experiment.json','configs/models.json','configs/sources.lock.json',
        'docs/research-spec.md','docs/planning-recovery-live-protocol-2026-10-09.md',
        'weights/steve1/2x.model','weights/steve1/steve1.weights',
        'results/diverse-teacher-worlds-20261008/development-corner.zip',
        'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy',
        'results/goal-conditioning-diagnosis-20261009/explore_9711.npy')]+[checkpoint,folder/'prompt.txt']
    manifest=dict(schema=1,approved=False,status='prepared; separate user approval required',
        maximum_api_calls=1,api_retries=0,model='gpt-5.3-codex',max_output_tokens=1024,
        planner_wait_s=20,control_limit_s=180,recovery_limit_ticks=200,recovery_limit_s=30,
        sampling_seed=9721,world_seed=104761,optimizer_updates=0,training_eligible=False,
        reserved_final_seeds=[3000017,3000029,3000047,3000061,3000073],
        files={p.relative_to(root).as_posix():digest(p) for p in files},
        local_prerequisite_audits={p.relative_to(root).as_posix():digest(p) for p in
                                  [run/'planning-recovery-audit.json' for run in runs]})
    write_json(folder/'manifest.json',manifest)
    from run_planner_exchange_pilot import verify_prepared
    verified=verify_prepared(root)
    result=dict(completed=True,local_integration_verified=True,actual_api_calls=0,optimizer_updates=0,
        runs=[dict(run=str(run),**audit) for run,audit in zip(runs,audits)],
        preserved_research_spec_sha256=digest(spec),checkpoint_sha256=digest(checkpoint),
        test_logs={name:digest(impl/name) for name in test_logs},prepared_live=verified,
        next_step='User decision: one live paid API call, no retries; no automatic paid execution',
        performance_sample=False,scope='Local execution integration, no learning or causal LLM performance claim')
    destination=impl/'final-verification.json'
    if destination.exists():raise FileExistsError(destination)
    write_json(destination,result)
    print(json.dumps(dict(completed=True,runs=len(runs),transitions=sum(a['steps'] for a in audits),
                          actual_api_calls=0,prepared=str(folder/'manifest.json'))))


if __name__=='__main__':main()
