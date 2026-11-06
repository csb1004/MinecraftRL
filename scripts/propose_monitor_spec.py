"""Ask the LLM for a goal monitor spec and check it (plan 2026-11-05 §1, §4; task D 2026-11-07).

Usage: python scripts/propose_monitor_spec.py --facts FACTS.json --version NAME --out NEW_DIR --allow-paid
       [--margin 0.2] [--repairs 1] [--fit DEV_RUN_DIR... [--waste-max 0.5] [--anomaly-max 0.2]]
Paid: one Responses API call per attempt (at most 1 + repairs). Needs OPENAI_MODEL and OPENAI_API_KEY in the
environment; the key is never printed or written. Writes NEW_DIR/receipt.json, and NEW_DIR/spec.json only when the
proposal passes the fact rules and parse_spec. Exit 0 ok, 3 rejected after repairs, 2 refused.
Constraints are copied from the human spec named in the facts file (the LLM writes none).
With --fit, the accepted spec also gets the record check (monitor_signals): flagged signals go into the receipt and
NEW_DIR/calibrated.json is the copy without them (version NAME-cal). A person picks which spec to use.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from minecraftrl.pipeline.monitor_replay import load_run  # noqa: E402
from minecraftrl.pipeline.monitor_signals import calibrated_spec, flag_signals, signal_rates  # noqa: E402
from minecraftrl.pipeline.monitor_spec import parse_spec  # noqa: E402
from minecraftrl.pipeline.monitor_spec_llm import SubgoalFact, propose_spec, system_prompt  # noqa: E402
from minecraftrl.pipeline.openai_client import ResponsesClient  # noqa: E402


def main(argv=None, client=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--facts', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--margin', type=float, default=.2)
    p.add_argument('--repairs', type=int, default=1, choices=(0, 1, 2))
    p.add_argument('--allow-paid', action='store_true')
    p.add_argument('--fit', nargs='+', type=Path, default=[], help='dev runs for the record check')
    p.add_argument('--waste-max', type=float, default=.5)
    p.add_argument('--anomaly-max', type=float, default=.2)
    a = p.parse_args(argv)
    if not a.allow_paid:
        print('refused: --allow-paid is required (paid API call)', file=sys.stderr); return 2
    if a.out.exists():
        print(f'refused: output folder exists: {a.out}', file=sys.stderr); return 2
    raw = json.loads(a.facts.read_text(encoding='utf-8'))
    if raw.get('schema') != 'monitor_spec_facts_v1':
        print('refused: facts schema must be monitor_spec_facts_v1', file=sys.stderr); return 2
    facts = tuple(SubgoalFact(**f) for f in raw['subgoals'])
    source = Path(raw['constraints_from'])
    source = source if source.is_absolute() else ROOT / source
    constraints = json.loads(source.read_text(encoding='utf-8')).get('constraints', [])
    if client is None:
        client = ResponsesClient(os.getenv('OPENAI_MODEL'), os.getenv('OPENAI_API_KEY'), allow_paid=True,
                                 max_calls=a.repairs + 1)
    result = propose_spec(client, raw['goal'], facts, version=a.version, margin=a.margin, constraints=constraints,
                          repairs=a.repairs)
    a.out.mkdir(parents=True)
    receipt = dict(ok=result['ok'], reasons=result['reasons'], attempts=result['attempts'], model=client.model,
                   calls=len(result['attempts']), version=a.version, margin=a.margin,
                   facts=str(a.facts), facts_sha256=hashlib.sha256(a.facts.read_bytes()).hexdigest(),
                   constraints_from=str(source), prompt_sha256=hashlib.sha256(system_prompt().encode('utf-8')).hexdigest(),
                   source_sha256=hashlib.sha256((ROOT / 'src/minecraftrl/pipeline/monitor_spec_llm.py').read_bytes()).hexdigest(),
                   record_check=None)
    calibrated = None
    if result['ok'] and a.fit:
        runs = [load_run(d) for d in a.fit if (d / 'timeline.jsonl').exists() and (d / 'telemetry.jsonl').exists()]
        flags = flag_signals(signal_rates(runs, parse_spec(result['spec'])), waste_max=a.waste_max,
                             anomaly_max=a.anomaly_max)
        calibrated = calibrated_spec(result['spec'], flags, version=a.version + '-cal'); parse_spec(calibrated)
        receipt['record_check'] = dict(fit=[str(r.path) for r in runs], flags=flags,
                                       limits=dict(waste_max=a.waste_max, anomaly_max=a.anomaly_max))
    (a.out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not result['ok']:
        print('rejected after repairs: ' + '; '.join(result['reasons']), file=sys.stderr); return 3
    (a.out / 'spec.json').write_text(json.dumps(result['spec'], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if calibrated is not None:
        (a.out / 'calibrated.json').write_text(json.dumps(calibrated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {a.out / "spec.json"} ({len(result["attempts"])} call(s))')
    return 0


if __name__ == '__main__':
    sys.exit(main())
