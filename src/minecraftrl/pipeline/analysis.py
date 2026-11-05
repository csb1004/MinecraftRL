"""Evaluation summary of demo runs (spec 5.12, plan Task 11): success is recomputed from server telemetry.

Conditions (empty KB / inherited KB) are reported separately and never merged into one success rate.
"""
import json
import statistics
from pathlib import Path

from .viewer import read_jsonl

GOAL = 'iron_ore'
STOP_KINDS = ('stop_file',)


def failure_category(summary, success):
    """None on success. 'technical': invalid execution or a failed API call; 'motor': the policy ran out of time, failed a
    step, or kept repeating the same anomaly until the retry limit; 'knowledge': the slow path got no usable knowledge or
    plan (evaluation round 1 showed that a hold alone does not say which)."""
    if success: return None
    if not summary.get('execution_valid'): return 'technical'
    if summary.get('status') != 'anomaly_hold': return 'motor'
    slow = ((summary.get('stage_results') or [{}])[-1].get('slow_path')) or {}
    if slow.get('llm') == 'failed': return 'technical'
    if slow.get('route') in ('retry_limit', 'retry_stage'): return 'motor'
    if 'not enough spare placeable blocks' in str(slow.get('reason')): return 'motor'  # blocks spent off the pillar
    return 'knowledge'


def analyze_run(run_dir, *, by_status=False):
    """by_status: success = all planned steps completed (scenes whose goal is not an item, e.g. the ceiling escape)."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    samples = read_jsonl(run_dir / 'telemetry.jsonl'); timeline = read_jsonl(run_dir / 'timeline.jsonl')
    iron = max((int((s.get('inventory') or {}).get(GOAL, 0)) for s in samples), default=0)
    # Goal blocks broken without harvest = irreversible losses the guard is there to prevent (target 0).
    losses, seen = 0, 0
    for s in samples:
        demo = s.get('demo') or {}; destroys = int(demo.get('demo_destroys', 0) or 0)
        if destroys > seen:
            if demo.get('demo_last_destroyed_block') == GOAL and not demo.get('demo_last_destroy_harvestable'): losses += destroys - seen
            seen = destroys
    holds = [e for e in timeline if e['kind'] == 'hold']
    usage = (summary.get('oversight') or {}).get('usage') or []
    success = bool(summary.get('execution_valid')) and (summary.get('status') == 'completed' if by_status else iron >= 1)
    return dict(run=run_dir.name, success=success, iron_ore=iron, status=summary.get('status'),
                execution_valid=summary.get('execution_valid'), release=summary.get('release'),
                elapsed_control_s=summary.get('elapsed_control_s'), steps=summary.get('steps'),
                api_calls=(summary.get('oversight') or {}).get('api_calls', 0),
                tokens=sum(int((u or {}).get('total_tokens', 0)) for u in usage),
                hold_latency_ms=[e.get('hold_latency_ms') for e in holds],
                guard_mining_ticks=[e.get('mining_ticks_at_hold') for e in holds if str(e.get('reason', '')).startswith('guard_')],
                hold_reasons=[e.get('reason') for e in holds],
                detect_to_resume_s=[e.get('detect_to_resume_s') for e in timeline if e['kind'] == 'resume'],
                attack_masks=sum(int(s.get('attack_masks') or 0) for s in summary.get('stage_results') or []),
                guarded_block_losses=losses, failure_category=failure_category(summary, success),
                stages=[dict(id=s.get('id'), status=s.get('status'), steps=s.get('steps')) for s in summary.get('stage_results') or []])


def _quantiles(values):
    values = sorted(v for v in values if v is not None)
    if not values: return None
    pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]
    return dict(n=len(values), p50=pick(.5), p95=pick(.95), max=values[-1], mean=statistics.fmean(values))


def analyze_batch(conditions, *, by_status=False):
    """{condition: [run dirs]} -> per-condition summary; conditions stay separate."""
    out = {}
    for condition, runs in conditions.items():
        rows = [analyze_run(r, by_status=by_status) for r in runs if (Path(r) / 'summary.json').exists()]
        out[condition] = dict(
            runs=len(rows), success=sum(r['success'] for r in rows), valid=sum(bool(r['execution_valid']) for r in rows),
            guarded_block_losses=sum(r['guarded_block_losses'] for r in rows), attack_masks=sum(r['attack_masks'] for r in rows),
            api_calls=sum(r['api_calls'] for r in rows), tokens=sum(r['tokens'] for r in rows),
            failure_categories={c: sum(r['failure_category'] == c for r in rows) for c in ('knowledge', 'motor', 'technical')},
            hold_latency_ms=_quantiles([v for r in rows for v in r['hold_latency_ms']]),
            detect_to_resume_s=_quantiles([v for r in rows for v in r['detect_to_resume_s']]),
            elapsed_control_s=_quantiles([r['elapsed_control_s'] for r in rows if r['success']]), per_run=rows)
    return out
