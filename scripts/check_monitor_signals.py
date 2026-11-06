"""Check a monitor spec's signals against completed dev records and write a calibrated copy (task D 2026-11-07).

Usage: python scripts/check_monitor_signals.py --spec SPEC.json --fit RUN_DIR... --out NEW_DIR --version NAME
       [--waste-max 0.5] [--anomaly-max 0.2]
Flags waste signals that cover at least waste-max of the rows, and anomaly signals that fire in at least anomaly-max
of the completed steps of their sub-goal (normal behaviour, plan §7.2). Writes report.json and spec.json (the
calibrated copy). Dev records only; no game, GPU or API. The output folder must not exist (exit 2).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minecraftrl.pipeline.monitor_replay import load_run  # noqa: E402
from minecraftrl.pipeline.monitor_signals import calibrated_spec, flag_signals, signal_rates  # noqa: E402
from minecraftrl.pipeline.monitor_spec import SpecError, parse_spec  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--fit', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--waste-max', type=float, default=.5)
    p.add_argument('--anomaly-max', type=float, default=.2)
    a = p.parse_args(argv)
    if a.out.exists():
        print(f'rejected: output folder exists: {a.out}', file=sys.stderr); return 2
    raw = json.loads(a.spec.read_text(encoding='utf-8'))
    try: spec = parse_spec(raw)
    except SpecError as e:
        print(f'rejected: {e}', file=sys.stderr); return 2
    runs = [load_run(d) for d in a.fit if (d / 'timeline.jsonl').exists() and (d / 'telemetry.jsonl').exists()]
    rates = signal_rates(runs, spec)
    flags = flag_signals(rates, waste_max=a.waste_max, anomaly_max=a.anomaly_max)
    calibrated = calibrated_spec(raw, flags, version=a.version)
    parse_spec(calibrated)
    a.out.mkdir(parents=True)
    report = dict(spec=str(a.spec), version=spec.version, calibrated_version=a.version, fit=[str(r.path) for r in runs],
                  limits=dict(waste_max=a.waste_max, anomaly_max=a.anomaly_max), rates=rates, flags=flags)
    (a.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (a.out / 'spec.json').write_text(json.dumps(calibrated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(flags)} flagged; wrote {a.out / "report.json"} and spec.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
