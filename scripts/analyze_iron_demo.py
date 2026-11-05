"""Summarise iron-demo runs per condition (spec 5.12); success is recomputed from server telemetry.

Usage: python scripts/analyze_iron_demo.py --condition NAME RUN_DIR [RUN_DIR ...] [--condition NAME ...] [--out analysis.json]
Conditions are never merged into one success rate.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minecraftrl.pipeline.analysis import analyze_batch  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--condition', nargs='+', action='append', required=True, metavar=('NAME', 'RUN_DIR'))
    p.add_argument('--out', type=Path)
    p.add_argument('--success-status', action='store_true', help='success = all steps completed (goal is not an item)')
    a = p.parse_args(argv)
    conditions = {group[0]: [Path(r) for r in group[1:]] for group in a.condition}
    result = analyze_batch(conditions, by_status=a.success_status)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out: a.out.write_text(text + '\n', encoding='utf-8')
    for name, c in result.items():
        print('%s: success %d/%d, valid %d, guarded losses %d, masks %d, API %d (%d tokens), failures %s'
              % (name, c['success'], c['runs'], c['valid'], c['guarded_block_losses'], c['attack_masks'], c['api_calls'], c['tokens'],
                 c['failure_categories']))


if __name__ == '__main__':
    main()
