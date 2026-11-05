"""Task 2 skill probes for the iron demo (development world, no API calls).

Each trial is one supervised game (`minecraftrl.pipeline.cli probe`) with a
development-only fixture variant (`give` items) written under the batch folder.
Finished trials are skipped on rerun. `<output>/STOP` stops the current trial and the batch.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

GROUPS = {
    # skill group -> candidates in order of prior evidence; a candidate with >=2/3 ends its group.
    'log': ['wood_visual'],
    'cobblestone': ['dig_visual', 'mine_stone_text'],
    'iron_ore': ['mine_iron_text', 'break_iron_text', 'dig_visual'],
}
TRIALS = (1, 2, 3)
POLICY_DEVICE_MIB = 2150  # prior STEVE-1 runs: torch reserved 2020 MiB, plus margin


def gpu_free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits', '-i', '0'],
                         capture_output=True, text=True, timeout=10)
    used, total = (float(v) for v in out.stdout.strip().split(','))
    return total - used


def policy_probe(group, skill):
    if group == 'log':
        return dict(give=[], limit_s=180, steps=[dict(id='logs', skill=skill, kind='policy', target='log', count=3, block='log')])
    if group == 'cobblestone':
        return dict(give=[dict(item='wooden_pickaxe', count=1)], limit_s=180,
                    steps=[dict(id='cobblestone', skill=skill, kind='policy', target='cobblestone', count=3, block='stone')])
    return dict(give=[dict(item='stone_pickaxe', count=1)], limit_s=180,
                steps=[dict(id='iron', skill=skill, kind='policy', target='iron_ore', count=1, block='iron_ore')])


def aim_probe(skill):
    # Bare hand: stop at the guard's decision point; then a scripted hold measures the outcome (not policy success).
    return dict(give=[], limit_s=180, steps=[
        dict(id='aim', skill=skill, kind='policy', target='iron_ore', count=1, block='iron_ore', params={'until': 'mining_start'}),
        dict(id='telemetry', skill='attack_hold', kind='scripted', target='iron_ore', count=1, block='iron_ore',
             params={'limit_steps': 600})])


RULE_PROBES = {
    'craft_chain': dict(give=[dict(item='oak_log', count=4), dict(item='cobblestone', count=3)], limit_s=240, steps=[
        dict(id='wooden', skill='craft', kind='rule', target='wooden_pickaxe', count=1,
             params={'goals': ['wooden_pickaxe'], 'equip': 'wooden_pickaxe'}),
        dict(id='stone_reuse_table', skill='craft', kind='rule', target='stone_pickaxe', count=1,
             params={'goals': ['stone_pickaxe'], 'equip': 'stone_pickaxe'})]),
    'craft_stone_new_table': dict(give=[dict(item='cobblestone', count=3), dict(item='stick', count=2),
                                        dict(item='crafting_table', count=1)], limit_s=180, steps=[
        dict(id='stone_new_table', skill='craft', kind='rule', target='stone_pickaxe', count=1,
             params={'goals': ['stone_pickaxe'], 'equip': 'stone_pickaxe'})]),
}


def succeeded(summary, probe):
    stages = summary.get('stage_results', [])
    if probe == 'aim':
        return bool(stages) and stages[0]['status'] == 'completed'
    return bool(summary.get('execution_valid')) and bool(summary.get('all_steps_completed'))


def run_trial(args, name, spec, trial, base):
    folder = args.output / name / ('trial-%d' % trial)
    summary_path = folder / 'summary.json'
    if summary_path.exists(): return json.loads(summary_path.read_text()), False
    fixture = args.output / 'fixtures' / (name + '.json')
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps(dict(base, give=spec['give'],
        description=base['description'] + ' DEVELOPMENT PROBE VARIANT: start items given by fixture.'), indent=2) + '\n')
    plan = args.output / 'plans' / (name + '.json')
    plan.parent.mkdir(parents=True, exist_ok=True); plan.write_text(json.dumps(spec['steps'], indent=2) + '\n')
    folder.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'minecraftrl.pipeline.cli', 'probe', '--root', str(args.root),
                    '--output', str(folder), '--config', str(args.config), '--plan-json', str(plan), '--fixture', str(fixture),
                    '--stop-file', str(args.output / 'STOP'), '--seed', str(args.seed), '--sampling-seed', str(trial),
                    '--limit-s', str(spec['limit_s']), '--frame-every', str(args.frame_every)], cwd=str(args.root), check=False)
    return (json.loads(summary_path.read_text()) if summary_path.exists() else dict(status='missing_summary')), True


def exclude(args, name, trial):
    """Move an invalid launch aside so a rerun repeats the trial instead of resuming it as a loss."""
    folder = args.output / name / ('trial-%d' % trial)
    parent = args.output / 'excluded' / name; parent.mkdir(parents=True, exist_ok=True)
    target = parent / ('trial-%d-%d' % (trial, len(list(parent.glob('trial-%d-*' % trial))) + 1))
    folder.rename(target)
    return target


def record(args, name, trial, probe, result):
    """Policy outcome of one trial; None when the launch itself was invalid (excluded, batch must halt)."""
    summary, fresh = result
    if not fresh: return succeeded(summary, probe)
    stages = [{k: s.get(k) for k in ('id', 'status', 'completion', 'steps', 'gain', 'destroy', 'reason')}
              for s in summary.get('stage_results', [])]
    line = dict(name=name, trial=trial, probe=probe, success=succeeded(summary, probe), status=summary.get('status'),
                execution_valid=summary.get('execution_valid'), elapsed_control_s=summary.get('elapsed_control_s'),
                steps=summary.get('steps'), stages=stages, remaining_pids=summary.get('process_cleanup', {}).get('remaining_live_pids'))
    if not summary.get('execution_valid'):
        # Startup crash, cancelled launch or unreleased input: not a policy result.
        line.update(success=False, excluded=True, moved_to=str(exclude(args, name.replace(':', '-'), trial).relative_to(args.output)))
    with (args.output / 'probe-results.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(line) + '\n')
    print(json.dumps(line), flush=True)
    return None if line.get('excluded') else line['success']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--seed', type=int)
    p.add_argument('--frame-every', type=int, default=5)
    p.add_argument('--trials', type=int, default=3, help='run trials 1..N of each probe (resume fills the rest)')
    p.add_argument('--only', nargs='*', help='probe names, e.g. log:wood_visual aim:mine_iron_text craft_chain')
    args = p.parse_args()
    args.root = args.root.resolve(); args.output = args.output.resolve()
    args.config = (args.config or args.root / 'configs/iron-demo.json').resolve()
    config = json.loads(args.config.read_text())
    args.seed = config['development_seed'] if args.seed is None else args.seed
    base = json.loads((args.fixture or args.root / config.get('fixture', 'configs/iron-demo-fixture.json')).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    stop = args.output / 'STOP'
    wanted = lambda name: not args.only or name in args.only
    selected = {}
    for group, candidates in GROUPS.items():
        for skill in candidates:
            probes = [('policy', group + ':' + skill, policy_probe(group, skill))]
            if group == 'iron_ore': probes.append(('aim', 'aim:' + skill, aim_probe(skill)))
            probes = [x for x in probes if wanted(x[1])]
            if not probes: continue
            passed = True
            for probe, name, spec in probes:
                wins = 0
                for trial in TRIALS[:args.trials]:
                    if stop.exists(): print('STOP present; batch halted'); return 1
                    if not (args.output / name.replace(':', '-') / ('trial-%d' % trial) / 'summary.json').exists():
                        free = gpu_free_mib()
                        # Other desktop GPU use would starve STEVE-1; refuse rather than record an invalid trial.
                        if free < POLICY_DEVICE_MIB: print('GPU busy: %.0f MiB free; batch halted' % free); return 3
                    won = record(args, name, trial, probe, run_trial(args, name.replace(':', '-'), spec, trial, base))
                    if won is None: print('invalid launch excluded; batch halted'); return 4
                    wins += won
                passed &= wins >= 2
            if passed: selected[group] = skill; break
    for name, spec in RULE_PROBES.items():
        if not wanted(name): continue
        for trial in (1, 2)[:args.trials]:
            if stop.exists(): print('STOP present; batch halted'); return 1
            if record(args, name, trial, 'rule', run_trial(args, name, spec, trial, base)) is None:
                print('invalid launch excluded; batch halted'); return 4
    (args.output / 'probe-selection.json').write_text(json.dumps(dict(selected=selected, scope=
        'first candidate per group reaching >=2/3 on every probe of that group; development seed only'), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
