"""Freeze completed local units and verify preserved inputs; never run a game."""
import json
import re
import shutil
from pathlib import Path

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
out = root / 'results/recovery-outcome-fix-20261009'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
destination = out / 'final-verification.json'
if destination.exists():
    raise FileExistsError(destination)
hashes = {}


def checked(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    key = path.relative_to(root).as_posix()
    if key not in hashes:
        hashes[key] = digest(path)
    if expected is not None:
        assert hashes[key] == expected, key
    return hashes[key]


games = []
for directory, audit_name in (
    ('recovery-outcome-game-20261009', 'recovery-outcome-audit.json'),
    ('fresh-target-handoff-game-20261009', 'fresh-target-handoff-audit.json'),
):
    work_path = root / 'results' / directory / 'work.json'
    work = read(work_path)
    assert work['completed'] and work['error'] is None and work['sources_unchanged']
    assert work['actual_games'] == 1 and work['api_calls'] == 0
    run = Path(work['run'])
    audit = read(run / audit_name)
    assert audit['verified'] and audit['post_stop_dispatches'] == audit['post_stop_inferences'] == 0
    assert audit['held_keys_after_release'] == 0 and not audit['remaining_owned_pids']
    for name, expected in audit['source_sha256'].items():
        checked(run / name, expected)
    for name, expected in work['frozen_sha256'].items():
        path = Path(name)
        # Keep v1's executed code as evidence instead of pretending v2 ran then.
        if directory == 'recovery-outcome-game-20261009':
            archived = out / 'source-after-first-game' / path
            if archived.exists():
                path = archived
        checked(path, expected)
    checked(work_path)
    checked(run / audit_name)
    games.append(dict(run=str(run.relative_to(root)), steps=audit['steps'],
        status=audit['status'], log_gain=audit['log_gain'],
        post_stop_dispatches=0, held_keys=0, remaining_owned_pids=[]))

training_path = root / 'results/coverage-anchor-ablation-20261009/training.json'
training = read(training_path)
assert training['completed'] and training['error'] is None and training['updates'] == 662
assert training['sources_unchanged'] and training['reload_max_error'] == 0
assert not training['diagnostic_eligible'] and not training['runtime_comparison_eligible']
for name, expected in training['frozen_sha256'].items():
    checked(name, expected)
checked(training_path)
checked(training_path.parent / 'checkpoint.pt', training['checkpoint_sha256'])
supervision = read(training_path.parent / 'supervision-audit.json')
assert supervision['verified'] and supervision['development_optimizer_targets'] == 0
admission_path = root / 'results/anchor-bc-comparison-20261009/audit.json'
admission = read(admission_path)
assert admission['verified'] and admission['actual_games'] == 0 and len(admission['unrun_conditions']) == 4
checked(admission_path)

tests = {}
for platform, total, skips in (('windows', 433, 19), ('wsl', 438, 9)):
    path = out / (platform + '-tests-fresh-target.log')
    log = path.read_text(encoding='utf-8-sig')
    assert re.search(rf'Ran {total} tests in ', log)
    assert f'OK (skipped={skips})' in log and 'FAILED (' not in log
    tests[platform] = dict(total=total, skipped=skips, passed=total-skips)
    checked(path)

processes = read(out / 'process-final.json')
assert not processes['windows_project_processes']
assert all(not p['original_launcher_alive'] for p in processes['launchers'])
assert not any(re.search(r'\b(?:python\S*|java\S*|Xvfb|micromamba)$', line.strip())
               for line in processes['wsl_ps'])
assert processes['stop_present'] and (root / 'results/STOP').exists()
checked(out / 'process-final.json')
checked('results/STOP')

previous_api = read(root / 'results/planning-recovery-live-20261009/final-verification.json')
assert previous_api['remaining_authorized_calls'] == 0
for name, expected in previous_api['sha256'].items():
    if 'api-receipt' in name:
        checked(name, expected)
checked('docs/research-spec.md', '4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad')

source_files = {p.relative_to(out / 'source-before') for p in (out / 'source-before').rglob('*') if p.is_file()}
source_files.update(Path(p) for p in (
    'scripts/run_recovery_outcome_validation.py', 'scripts/audit_recovery_outcome.py',
    'scripts/train_coverage_anchor_ablation.py', 'scripts/audit_coverage_anchor_ablation.py',
    'scripts/summarize_anchor_ablation.py', 'scripts/run_anchor_bc_comparison.py',
    'scripts/audit_anchor_bc_games.py', 'scripts/run_fresh_target_handoff.py',
    'scripts/audit_fresh_target_handoff.py', 'scripts/report_fresh_target_handoff.py',
    'scripts/replay_fresh_target_handoff.py', 'scripts/diagnose_fresh_target_failure.py',
    'scripts/finalize_recovery_outcome_fix.py',
    'docs/recovery-outcome-fix-protocol-2026-10-09.md',
    'docs/recovery-outcome-fix-validation-2026-10-09.md',
    'docs/fresh-target-handoff-protocol-2026-10-09.md',
    'docs/coverage-anchor-runtime-protocol-2026-10-09.md',
    'docs/bc-anchor-objective-analysis-2026-10-09.md',
    'docs/next-task-units.md', 'docs/llm-high-level-planner-2026-10-07.md',
))
snapshot = out / 'source-final'
snapshot.mkdir(exist_ok=False)
diff = []
for name in sorted(source_files):
    path = root / name
    target = snapshot / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    after = checked(path)
    assert digest(target) == after
    before = out / 'source-before' / name
    diff.append(dict(path=name.as_posix(), before_sha256=digest(before) if before.exists() else None,
                     after_sha256=after))
write_json(out / 'source-diff-final.json', diff)
for directory in ('results/coverage-anchor-ablation-20261009',
                  'results/fresh-target-handoff-game-20261009/analysis'):
    for path in (root / directory).glob('*'):
        if path.suffix in ('.json', '.png'):
            checked(path)

result = dict(completed=True, scope='Bounded control fix and KL ablation completed; generalized detour behavior remains unresolved.',
    local_game_runs=2, games=games, audited_transitions=sum(g['steps'] for g in games),
    extra_api_calls=0, actual_colab_connections=0, remaining_authorized_paid_calls=0,
    optimizer_updates=662, new_rl_updates=0, default_policy_promotions=0,
    training_and_game_sources_preserved=True, conditional_candidate_games=0,
    conditional_unrun_games=4, final_evaluation_seeds_used=[], tests=tests,
    stop_latched=True, remaining_owned_game_training_processes=[],
    handoff_verified=True, generalized_recovery_solved=False, sha256=hashes)
write_json(destination, result)
print(json.dumps({k: v for k, v in result.items() if k != 'sha256'}))
