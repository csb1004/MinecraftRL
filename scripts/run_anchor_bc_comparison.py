"""One-use KL1/.1 candidate comparison with unchanged diagnostic admission."""
import json
import sys
from pathlib import Path

from run_rl_batch import ROOT, LINUX, compare_initial, linux, rearm_allowed, wait_owned
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


base = ROOT/'results/anchor-bc-comparison-20261009'
base.mkdir(exist_ok=False)
paths = dict(parent=ROOT/'results/onset-coverage-training-20261009/checkpoint.pt',
             coverage=ROOT/'results/coverage-anchor-ablation-20261009/checkpoint.pt')
embed = ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
protocol = ROOT/'docs/coverage-anchor-runtime-protocol-2026-10-09.md'
old = json.loads((ROOT/'results/recovery-curriculum-worlds-20261007/worlds.json').read_text())['worlds']
natural = json.loads((ROOT/'results/rl-batch-20261005T040215634778Z/batch.json').read_text())['worlds']['2000029']
worlds = [dict(old['eval-long-2000003'], kind='long_wall'),
          dict(natural, seed=2000029, split='eval', kind='natural')]
state = dict(completed=False, active=None, runs=[], errors=[], warnings=[], comparisons={}, worlds=worlds,
             checkpoints={k: str(p) for k, p in paths.items()}, optimizer_updates=0,
             protocol_sha256=digest(protocol), gate_policy='preregistered_coverage_diagnostic',
             planned_conditions=4, unrun=[], comparison='KL1 versus KL.1: same parent/data/order/662 updates; coverage label denotes KL.1')
frozen = {str(p): digest(p) for p in (embed, protocol, ROOT/'docs/research-spec.md',
          Path(__file__), ROOT/'scripts/audit_anchor_bc_games.py', ROOT/'scripts/run_rl_pilot.py',
          ROOT/'src/minecraftrl/rl_control.py')}


def save():
    write_json(base/'experiment.json', state)


def trial(model, world):
    rearm_allowed()
    for path, sha in frozen.items():
        assert digest(path) == sha
    assert digest(world['path']) == world['sha256']
    label = world['kind']+'-'+model
    checkpoint = paths[model]
    state['active'] = label
    save()
    print(label, flush=True)
    log = base/(label+'.log')
    report = run = None
    try:
        wait_owned([sys.executable, str(ROOT/'scripts/run_rl_pilot.py'), '--seed', str(world['seed']),
                    '--split', 'eval', '--sampling-seed', str(world['seed']), '--reward-mode', 'wood-potential-v4',
                    '--limit-s', '180', '--world', world['path'], '--checkpoint', str(checkpoint), '--embed', str(embed)], log)
    finally:
        lines = log.read_text(encoding='utf-8').splitlines() if log.exists() else []
        if lines:
            run = Path(lines[0])/'run'
            if (run/'summary.json').exists():
                report = json.loads((run/'summary.json').read_text())
                state['runs'].append(dict(label=label, model=model, path=str(run), world=world,
                                          checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                                          report=report, audited=False))
                save()
    assert report and report['execution_valid'] and report['worker_exit_code'] == 0
    assert report['rl_split'] == 'eval' and not report['rl_training_eligible']
    assert report['controller'] == 'steve1_head_rl' and not report['process_cleanup']['remaining_live_pids']
    release = json.loads((run/'input-release-state.json').read_text())
    assert release['verified'] and release['held_key_count'] == 0
    wsl = ['wsl.exe', '-d', 'Ubuntu-24.04', '--cd', LINUX, '--', 'env', 'PYTHONPATH=src',
           'LD_LIBRARY_PATH=/usr/lib/wsl/lib', 'MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',
           '.runtime/bin/micromamba', 'run', '-p', LINUX+'/.runtime/steve1', 'python']
    wait_owned(wsl+['scripts/audit_wood_shaping.py', linux(run)], base/(label+'-reward-audit.log'))
    wait_owned([sys.executable, str(ROOT/'scripts/audit_approach_recovery.py'), str(run)],
               base/(label+'-recovery-audit.log'))
    state['runs'][-1].update(audited=True, release_evidence=release)
    save()
    print(json.dumps(dict(label=label, steps=report['steps'], success=report['server_target_reached'])), flush=True)
    return run


save()
try:
    admitted = []
    state['strict_gate_passed'] = {}
    for model, path in paths.items():
        training = json.loads((path.parent/'training.json').read_text())
        assert training['completed'] and training['error'] is None and training['checkpoint_sha256'] == digest(path)
        frozen[str(path)] = digest(path)
        frozen[str(path.parent/'training.json')] = digest(path.parent/'training.json')
        state['strict_gate_passed'][model] = training['runtime_comparison_eligible']
        if model == 'parent':
            continue
        assert training['updates'] == training['planned_updates'] and training['reload_max_error'] == 0
        assert 260 < training['updates'] <= 2400
        reference_training = json.loads((paths['parent'].parent/'training.json').read_text())
        assert training['parent_sha256'] == reference_training['parent_sha256'] and training['value_unchanged']
        audit_path = path.parent/'supervision-audit.json'
        assert json.loads(audit_path.read_text())['verified']
        frozen[str(audit_path)] = digest(audit_path)
        if training['diagnostic_eligible']:
            admitted.append(model)
        else:
            state['unrun'].extend(dict(model=model, world=w['kind'], reason='offline diagnostic admission not met') for w in worlds)
    state['admitted'] = admitted
    state['frozen'] = frozen
    save()
    if admitted:
        runtime_path = ROOT/'results/anchor-ablation-runtime-20261009/report.json'
        runtime = json.loads(runtime_path.read_text())
        assert runtime['completed'] and runtime['error'] is None
        assert {r['sha256'] for r in runtime['checkpoints']} == {digest(paths[k]) for k in admitted}
        frozen[str(runtime_path)] = digest(runtime_path)
        for i, world in enumerate(worlds):
            models = ['parent']+admitted if i == 0 else list(reversed(admitted))+['parent']
            runs = {model: trial(model, world) for model in models}
            for model in admitted:
                key = world['kind']+'-'+model
                comparison = compare_initial(runs['parent'], runs[model])
                state['comparisons'][key] = comparison
                if not (comparison['state_identical'] and comparison['pixel_identical']):
                    state['warnings'].append(key+': initial mismatch; causal/time comparison invalid')
            save()
    else:
        state['unrun'].extend(dict(model='parent', world=w['kind'], reason='no candidate admitted') for w in worlds)
    for path, sha in frozen.items():
        assert digest(path) == sha, path
    for world in worlds:
        assert digest(world['path']) == world['sha256']
    assert len(state['runs'])+len(state['unrun']) == state['planned_conditions']
    state['comparison_valid'] = {k: v['state_identical'] and v['pixel_identical'] for k, v in state['comparisons'].items()}
    state['completed'] = True
except BaseException as exc:
    state['errors'].append(repr(exc))
finally:
    state.update(active=None, execution_finished=True)
    save()
raise SystemExit(0 if state['completed'] else 2)
