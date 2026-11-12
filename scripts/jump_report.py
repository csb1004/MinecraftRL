"""Jump course report (design docs/specs/2026-11-13-jump-course-design.md §5; no game, GPU or API).

Per arm of a batch: runs, successes (server ground contact on the goal platform), falls, time-outs, the furthest platform
stood on, the median game seconds of the successful course steps (policy decisions / 20; one decision per tick), and per
jump (gap, height change) how often the arm stood on that jump's landing platform when it got the chance.
Usage: jump_report.py --batch B --arms A... --out DIR
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from minecraftrl.jump_course import generate_course  # noqa: E402


def arm_report(folder):
    runs = sorted(d for d in Path(folder).iterdir() if (d / 'summary.json').exists())
    out = dict(runs=len(runs), success=0, fell=0, other=Counter(), best=Counter(), seconds=[], jumps={})
    for run in runs:
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        stage = (summary.get('stage_results') or [{}])[-1]
        status = stage.get('status'); best = int(stage.get('best_platform') or 0)
        course = generate_course(int(run.name.split('-')[0][1:]))['platforms']
        if summary.get('execution_valid') and summary.get('all_steps_completed'):
            out['success'] += 1; out['seconds'].append(stage.get('steps', 0) / 20.); best = len(course) - 1
        elif status == 'course_fell': out['fell'] += 1
        else: out['other'][str(status or summary.get('status'))] += 1
        out['best'][best] += 1
        for i in range(1, min(best + 1, len(course) - 1) + 1):      # jumps tried: up to one past the furthest platform
            key = 'gap %d, %+d' % (course[i]['gap'], course[i]['top'] - course[i - 1]['top'])
            tried, made = out['jumps'].get(key, (0, 0))
            out['jumps'][key] = (tried + 1, made + (i <= best))
    out['median_s'] = statistics.median(out['seconds']) if out['seconds'] else None
    out['other'] = dict(out['other']); out['best'] = {str(k): v for k, v in sorted(out['best'].items())}
    return out


def markdown(batch, report):
    lines = ['# 점프 코스 결과 (%s)' % Path(batch).name, '',
             '| 갈래 | 성공 | 떨어짐 | 그 밖 | 성공 시간 중앙값(게임 초) | 가장 멀리 선 발판(실행 수) |', '|---|---|---|---|---|---|']
    for arm, r in report.items():
        lines.append('| %s | %d/%d | %d | %s | %s | %s |' % (
            arm, r['success'], r['runs'], r['fell'], ', '.join('%s %d' % kv for kv in r['other'].items()) or '-',
            '-' if r['median_s'] is None else '%.1f' % r['median_s'], ', '.join('%s: %d' % kv for kv in r['best'].items())))
    lines += ['', '| 갈래 | 점프(틈, 높이) | 시도 | 착지 |', '|---|---|---|---|']
    for arm, r in report.items():
        for key, (tried, made) in sorted(r['jumps'].items()): lines.append('| %s | %s | %d | %d |' % (arm, key, tried, made))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True); p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    report = {arm: arm_report(args.batch / arm) for arm in args.arms}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'jump.json').write_bytes((json.dumps(report, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    md = markdown(args.batch, report); (args.out / 'jump.md').write_bytes(md.encode('utf-8'))
    print(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
