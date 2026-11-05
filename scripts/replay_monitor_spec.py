"""Replay a goal monitor spec over recorded runs (plan 2026-11-05 M1+M2): fit check times on dev runs, evaluate on others.

Usage: python scripts/replay_monitor_spec.py --spec SPEC.json --fit RUN_DIR... --eval RUN_DIR... --out NEW_DIR
       [--margins 0.2,0.5,1.0] [--min-records 3]
No game, GPU or API. Records are read only; the output folder must not exist. Shared folders or seeds exit with 2.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minecraftrl.pipeline.monitor_replay import ReplayRejected, build_report, report_markdown  # noqa: E402
from minecraftrl.pipeline.monitor_spec import SpecError  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--fit', nargs='+', type=Path, required=True, help='dev runs: check times come from these only')
    p.add_argument('--eval', nargs='+', type=Path, required=True, help='runs to evaluate (other seed)')
    p.add_argument('--margins', default='0.2,0.5,1.0')
    p.add_argument('--min-records', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        print(f'rejected: output folder exists: {a.out}', file=sys.stderr); return 2
    try:
        report = build_report(a.spec, a.fit, a.eval, margins=tuple(float(m) for m in a.margins.split(',')),
                              min_records=a.min_records)
    except (ReplayRejected, SpecError) as e:
        print(f'rejected: {e}', file=sys.stderr); return 2
    a.out.mkdir(parents=True)
    (a.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (a.out / 'report.md').write_text(report_markdown(report), encoding='utf-8')
    print(f'wrote {a.out / "report.json"} and report.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
