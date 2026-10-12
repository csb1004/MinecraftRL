"""One-use eight-world correction collection after the paired diagnostic games."""
import json
import sys
import traceback
from pathlib import Path

from run_rl_batch import ROOT, rearm_allowed, wait_owned
from run_student_corrections import audit
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    comparison = ROOT/'results/onset-bc-comparison-20261009'
    results = json.loads((comparison/'audit.json').read_text())
    assert results['verified']
    candidates = [r for r in results['rows'] if r['label'].startswith('long_wall-') and r['model'] != 'parent']
    assert candidates and not any(r['server_success'] for r in candidates), 'coverage contingency not met'
    world_file = ROOT/'results/onset-coverage-worlds-20261009/worlds.json'
    worlds = json.loads(world_file.read_text())
    assert worlds['completed'] and len(worlds['worlds']) == 8
    base = ROOT/'results/onset-coverage-collection-20261009'
    base.mkdir(exist_ok=False)
    checkpoint = ROOT/'results/temporal-bc-temporal-20261008/checkpoint.pt'
    embed = ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
    assert digest(checkpoint) == '8ce793b4f02af4218eb65d7e741ca3ac021d0caee1d891cc324722eeb288fdb9'
    assert digest(embed) == '810512d05a7f228d77aed45f29a21277156f934408e2387f2ae3fd6f4217be40'
    frozen = {str(p): digest(p) for p in (
        checkpoint, embed, world_file, comparison/'audit.json', ROOT/'docs/research-spec.md',
        ROOT/'docs/onset-coverage-protocol-2026-10-09.md', ROOT/'weights/steve1/steve1.weights',
        Path(__file__), ROOT/'scripts/audit_onset_coverage.py', ROOT/'scripts/run_student_corrections.py', ROOT/'scripts/run_rl_pilot.py',
        ROOT/'src/minecraftrl/student_correction.py')}
    state = dict(completed=False, active=None, runs=[], errors=[], optimizer_updates=0,
                 planned_labels=list(worlds['worlds']), checkpoint=str(checkpoint), frozen=frozen,
                 worlds=worlds['worlds'], scope='Explicit training demonstrations; teacher wins are assisted')

    def save():
        write_json(base/'experiment.json', state)

    save()
    try:
        for label, world in worlds['worlds'].items():
            for path, sha in frozen.items():
                assert digest(path) == sha
            assert world['split'] == 'train' and world['seed'] in (104743, 104759)
            assert digest(world['path']) == world['sha256']
            rearm_allowed()
            state['active'] = label
            save()
            print('collect '+label, flush=True)
            log = base/(label+'.log')
            report = run = None
            try:
                wait_owned([sys.executable, str(ROOT/'scripts/run_rl_pilot.py'), '--seed', str(world['seed']),
                    '--split', 'train', '--sampling-seed', str(world['sampling_seed']), '--student-correction',
                    '--checkpoint', str(checkpoint), '--embed', str(embed), '--world', world['path'],
                    '--reward-mode', 'wood-potential-v4', '--limit-s', '180'], log)
            finally:
                lines = log.read_text(encoding='utf-8').splitlines() if log.exists() else []
                if lines:
                    run = Path(lines[0])/'run'
                    if (run/'summary.json').exists():
                        report = json.loads((run/'summary.json').read_text())
                        state['runs'].append(dict(label=label, path=str(run), split='train', seed=world['seed'],
                                                 sampling_seed=world['sampling_seed'], report=report, audited=False))
                        save()
            assert report is not None, 'collection ended without a summary'
            result = audit(run, report)
            write_json(base/(label+'-audit.json'), result)
            state['runs'][-1].update(audited=True, audit=str(base/(label+'-audit.json')),
                teacher_episode_candidate=bool(result['owners'].get('teacher') and report['teacher_assisted_target_reached']))
            save()
            print(json.dumps(dict(label=label, owners=result['owners'], acquired=result['actual_log_delta'],
                                  student_success=report['student_target_reached'])), flush=True)
            assert digest(world['path']) == world['sha256']
        for path, sha in frozen.items():
            assert digest(path) == sha
        assert len(state['runs']) == 8
        state['completed'] = True
    except BaseException as exc:
        state['errors'].append(repr(exc))
        (base/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        state.update(active=None, execution_finished=True)
        save()
    return 0 if state['completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
