"""Multi-block experiment reports (design 2026-11-08 §6, §9, §10; no game, GPU or API).

survey: per training block of one arm - runs, server-inventory successes, attack while the crosshair is on the target,
        mining attempts and how they ended, and the R0 gate (design §6).
iron:   the iron comparison - success and iron-step game seconds (reacquire_compare) plus pooled iron mining attempts per
        arm, and the pre-set verdict for the multi-block arm (design §9; --iron-only omitted = the final form, §10).
Usage:  multi_block_report.py survey --batch B --arm official --out DIR
        multi_block_report.py iron --batch B --official official --multi multi [--iron-only hold-on] --out DIR
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from minecraftrl.mining_attempts import (attack_share, destroyed_share, dev_verdict, pool, r0_gate,  # noqa: E402
                                         run_block)


def finished(folder):
    return sorted(d for d in Path(folder).iterdir() if (d / 'summary.json').exists())


def write(out, name, value):
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_bytes(value.encode('utf-8') if isinstance(value, str)
                             else (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def pct(x): return '-' if x is None else '%.0f%%' % (100 * x)


def survey(batch, arm):
    blocks = {}
    for run in finished(Path(batch) / arm):
        block, success, stats = run_block(run)
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        b = blocks.setdefault(block, dict(runs=[], stats=[]))
        b['runs'].append(dict(run=run.name, success=success, elapsed_control_s=summary.get('elapsed_control_s'),
                              ends=stats['ends'], streaks=stats['streaks']))
        b['stats'].append(stats)
    per_block, report = {}, dict(batch=str(batch), arm=arm, blocks={})
    for block, b in sorted(blocks.items()):
        pooled = pool(b['stats']); per_block[block] = pooled
        times = [r['elapsed_control_s'] for r in b['runs'] if r['success'] and r['elapsed_control_s'] is not None]
        report['blocks'][block] = dict(runs=len(b['runs']), success=sum(r['success'] for r in b['runs']),
                                       attack_share=attack_share(pooled), ends=pooled['ends'],
                                       destroyed_share=destroyed_share(pooled),
                                       median_attempt=statistics.median(pooled['lengths']) if pooled['lengths'] else None,
                                       median_success_control_s=statistics.median(times) if times else None,
                                       streaks_before_destroy=pooled['streaks'], run_records=b['runs'])
    report['gate'] = r0_gate(per_block)
    lines = ['# 블록별 채굴 시도 (%s, %s)' % (Path(batch).name, arm), '',
             '| 블록 | 성공 | 조준 중 공격 | 시도 | 파괴 | 대상 위에서 뗌 | 조준 벗어남 | 파괴 비율 | 성공 제어 시간 중앙값(s) |',
             '|---|---|---|---|---|---|---|---|---|']
    for block, b in report['blocks'].items():
        e = b['ends']
        t = '-' if b['median_success_control_s'] is None else '%.0f' % b['median_success_control_s']
        lines.append('| %s | %d/%d | %s | %d | %d | %d | %d | %s | %s |' % (
            block, b['success'], b['runs'], pct(b['attack_share']), sum(e.values()), e['destroyed'], e['released'], e['left'],
            pct(b['destroyed_share']), t))
    g = report['gate']
    lines += ['', 'R0 관문(파괴 비율 %.0f%% 미만 블록 %d개 이상): %s, 실패 블록 %s' % (
        100 * g['threshold'], g['min_failing'], '통과' if g['passed'] else '불통과', ', '.join(g['failing_blocks']) or '없음')]
    return report, '\n'.join(lines) + '\n'


def iron(batch, arms):
    from minecraftrl.reacquire_compare import compare, report_markdown
    batch = Path(batch)
    dirs = {name: finished(batch / arm) for name, arm in arms.items() if arm}
    base = compare({arms[k]: v for k, v in dirs.items()}, root=ROOT)
    summary = {}
    for name, runs in dirs.items():
        stats = []
        for run in runs:
            block, _, s = run_block(run)
            if block != 'iron_ore': raise ValueError('%s does not mine iron_ore' % run)
            stats.append(s)
        a = base['arms'][arms[name]]
        summary[name] = dict(arm=arms[name], success=a['success'], runs=a['runs'], median_s=a['median_s'], attempts=pool(stats))
    verdict = dev_verdict(summary['official'], summary['multi'], summary.get('iron_only'))
    report = dict(batch=str(batch), arms=summary, verdict=verdict, compare=base)
    lines = [report_markdown(base), '## 철광석 채굴 시도', '',
             '| 갈래 | 성공 | 조준 중 공격 | 시도 | 파괴 | 대상 위에서 뗌 | 조준 벗어남 | 파괴 비율 |', '|---|---|---|---|---|---|---|---|']
    for name, s in summary.items():
        e = s['attempts']['ends']
        lines.append('| %s | %d/%d | %s | %d | %d | %d | %d | %s |' % (
            s['arm'], s['success'], s['runs'], pct(attack_share(s['attempts'])), sum(e.values()), e['destroyed'],
            e['released'], e['left'], pct(destroyed_share(s['attempts']))))
    checks = ', '.join('%s %s' % (k, 'O' if ok else 'X') for k, ok in verdict['conditions'].items())
    lines += ['', '미리 정한 판정(설계 §9/§10): %s (%s)' % ('나아짐' if verdict['improved'] else '나아졌다고 쓰지 않음', checks)]
    return report, '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('survey'); s.add_argument('--batch', type=Path, required=True); s.add_argument('--arm', required=True)
    s.add_argument('--out', type=Path, required=True)
    i = sub.add_parser('iron'); i.add_argument('--batch', type=Path, required=True)
    i.add_argument('--official', required=True); i.add_argument('--multi', required=True); i.add_argument('--iron-only')
    i.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'survey': report, md = survey(args.batch, args.arm); name = 'survey-%s' % args.arm
    else:
        report, md = iron(args.batch, dict(official=args.official, multi=args.multi, iron_only=args.iron_only))
        name = 'iron'
    write(args.out, name + '.json', report); write(args.out, name + '.md', md)
    print(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
