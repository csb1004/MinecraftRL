"""Building report (design docs/specs/2026-11-14-build-design.md §4; no game, GPU or API).

Per arm of a batch: runs, successes (the server placement record covers the blueprint), other endings, blueprint cells
filled (all runs), wrong placements (off the blueprint), the median game seconds of the successful steps (decisions /
20), and success by blueprint shape.
Usage: build_report.py --batch B --arms A... --out DIR
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from minecraftrl.build_task import SHAPES, generate_build  # noqa: E402


def arm_report(folder):
    runs = sorted(d for d in Path(folder).iterdir() if (d / 'summary.json').exists())
    out = dict(runs=len(runs), success=0, other=Counter(), filled=0, cells=0, wrong=0, seconds=[], by_shape={})
    for run in runs:
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        stage = (summary.get('stage_results') or [{}])[-1]
        ok = bool(summary.get('execution_valid') and summary.get('all_steps_completed'))
        task = generate_build(int(run.name.split('-')[0][1:]))
        out['filled'] += int(stage.get('filled') or 0); out['cells'] += len(task['cells']); out['wrong'] += int(stage.get('wrong') or 0)
        if ok: out['success'] += 1; out['seconds'].append(stage.get('steps', 0) / 20.)
        else: out['other'][str(stage.get('status') or summary.get('status'))] += 1
        tried, made = out['by_shape'].get(task['shape'], (0, 0)); out['by_shape'][task['shape']] = [tried + 1, made + ok]
    out['median_s'] = statistics.median(out['seconds']) if out['seconds'] else None
    out['other'] = dict(out['other'])
    return out


def markdown(batch, report):
    shapes = sorted(SHAPES)
    lines = ['# 건축 결과 (%s)' % Path(batch).name, '',
             '| 갈래 | 성공 | 그 밖 | 채운 칸 | 잘못 놓음 | 성공 시간 중앙값(게임 초) | ' + ' | '.join(shapes) + ' |',
             '|---|---|---|---|---|---|' + '---|' * len(shapes)]
    for arm, r in report.items():
        g = r['by_shape']
        lines.append('| %s | %d/%d | %s | %d/%d | %d | %s | ' % (
            arm, r['success'], r['runs'], ', '.join('%s %d' % kv for kv in r['other'].items()) or '-',
            r['filled'], r['cells'], r['wrong'], '-' if r['median_s'] is None else '%.1f' % r['median_s'])
            + ' | '.join('%d/%d' % (g[k][1], g[k][0]) if k in g else '-' for k in shapes) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True); p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    report = {arm: arm_report(args.batch / arm) for arm in args.arms}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'build.json').write_bytes((json.dumps(report, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    md = markdown(args.batch, report); (args.out / 'build.md').write_bytes(md.encode('utf-8'))
    print(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
