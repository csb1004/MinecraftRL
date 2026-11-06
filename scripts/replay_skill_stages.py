"""Split recorded policy steps into finer skill stages (plan 2026-11-07 task A): far, closing, near, aimed, mining,
collect, gui, in server-tick seconds, from each run's own fixture geometry.

Usage: python scripts/replay_skill_stages.py --runs RUN_DIR... --out NEW_DIR [--root PROJECT_DIR]
No game, GPU or API. Records are read only; the output folder must not exist (exit 2).
Fixture positions are privileged server geometry for offline analysis only.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from minecraftrl.pipeline.stage_replay import build_report, report_markdown  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=ROOT, help='project folder for fixture paths recorded on another mount')
    a = p.parse_args(argv)
    if a.out.exists():
        print(f'rejected: output folder exists: {a.out}', file=sys.stderr); return 2
    report = build_report(a.runs, root=a.root)
    a.out.mkdir(parents=True)
    (a.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (a.out / 'report.md').write_text(report_markdown(report), encoding='utf-8')
    print(f'wrote {a.out / "report.json"} and report.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
