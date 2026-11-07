"""Compare reacquire arms on the same probe scene (design 2026-11-07 §7).

Success is the server inventory (the probe step completed on iron_ore +1). Stage seconds come from stage_replay (game
seconds = server ticks / 20; fixture positions are privileged, analysis only). The reacquire share pools every run of
an arm, successful or not. The verdict uses the three checks fixed before the runs.
"""
import json
import statistics
from pathlib import Path

from .pipeline.stage_replay import ANGLE_BINS, STAGES, build_report

RELEASED = ('neutral_command_sent', 'environment_done_no_neutral_step')


def run_record(run_dir):
    s = json.loads((Path(run_dir) / 'summary.json').read_text(encoding='utf-8'))
    return dict(run=Path(run_dir).name, seed=s.get('seed'), sampling=s.get('sampling_seed'), status=s.get('status'),
                execution_valid=bool(s.get('execution_valid')),
                success=bool(s.get('execution_valid') and s.get('all_steps_completed')),
                release=s.get('release'), remaining_live_pids=(s.get('process_cleanup') or {}).get('remaining_live_pids'),
                elapsed_control_s=s.get('elapsed_control_s'), reacquire=s.get('reacquire'))


def arm_stats(runs, steps):
    """Pooled stage seconds and outcome counts; `steps` are stage_replay step reports aligned with `runs` (or None)."""
    seconds = dict.fromkeys(STAGES, 0.); angle = {label: 0. for _, label in ANGLE_BINS}; times = []
    for r, s in zip(runs, steps):
        if s is None: continue
        for k in STAGES: seconds[k] += s['seconds'][k]
        for k, v in s['near_angle_s'].items(): angle[k] += v
        if r['success']: times.append(s['total_s'])
    total = sum(seconds.values())
    return dict(runs=len(runs), success=sum(bool(r['success']) for r in runs),
                median_s=statistics.median(times) if times else None, step_s_success=times, seconds=seconds, total_s=total,
                shares={k: v / total for k, v in seconds.items()} if total else None,
                reacquire_share=seconds['near'] / total if total else None, near_angle_s=angle,
                missing_stage_report=[i for i, s in enumerate(steps) if s is None],
                stop_contract_failures=[i for i, r in enumerate(runs)
                                        if r['release'] not in RELEASED or r['remaining_live_pids'] != []])


def verdict(official, student):
    """Better only if success is not lower, the reacquire share is at most half, and the median step is shorter."""
    o, s = official, student
    checks = dict(success_not_lower=s['success'] >= o['success'],
                  reacquire_half_or_less=(s['reacquire_share'] is not None and o['reacquire_share'] is not None
                                          and s['reacquire_share'] <= o['reacquire_share'] / 2),
                  median_shorter=s['median_s'] is not None and o['median_s'] is not None and s['median_s'] < o['median_s'])
    return dict(better=all(checks.values()), checks=checks)


def compare(arm_dirs, *, root):
    out = dict(arms={}, verdicts={}, measure='success: server inventory; seconds: server ticks / 20 of the iron step',
               privileged='stage seconds use fixture positions and the server pose (analysis only)')
    for arm, dirs in arm_dirs.items():
        runs, steps = [run_record(d) for d in dirs], []
        for d in dirs:
            report = build_report([d], root=root)
            iron = [s for s in report['steps'] if s['target'] == 'iron_ore']
            steps.append(iron[0] if len(iron) == 1 else None)
        for r, s in zip(runs, steps): r['step_s'] = None if s is None else s['total_s']
        out['arms'][arm] = dict(arm_stats(runs, steps), run_records=runs)
    if 'official' in out['arms']:
        out['verdicts'] = {a: verdict(out['arms']['official'], v) for a, v in out['arms'].items() if a != 'official'}
    return out


def report_markdown(report):
    lines = ['# 돌아서기 학습 비교', '', '- ' + report['measure'] + '.', '- ' + report['privileged'] + '.', '',
             '| arm | 성공 | 성공 단계 중앙값(s) | 먼 곳 | 다가감 | 돌아서기(near) | 조준 | 채굴 | 회수 | near 중 90도 이상 | STOP 계약 실패 |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, a in report['arms'].items():
        sh = a['shares'] or dict.fromkeys(STAGES, 0.)
        near = sum(a['near_angle_s'].values()) or 1.
        median = '-' if a['median_s'] is None else '%.0f' % a['median_s']
        lines.append('| %s | %d/%d | %s | %s | %.0f%% | %d |' % (
            arm, a['success'], a['runs'], median,
            ' | '.join('%.0f%%' % (100 * sh[k]) for k in ('far', 'closing', 'near', 'aimed', 'mining', 'collect')),
            100 * a['near_angle_s']['>=90'] / near, len(a['stop_contract_failures'])))
    if report['verdicts']:
        lines += ['', '## 미리 정한 판정 (공식 대비)', '']
        for arm, v in report['verdicts'].items():
            checks = ', '.join('%s %s' % (k, 'O' if ok else 'X') for k, ok in v['checks'].items())
            lines.append('- %s: %s (%s)' % (arm, '나아짐' if v['better'] else '나아졌다고 쓰지 않음', checks))
    return '\n'.join(lines) + '\n'
