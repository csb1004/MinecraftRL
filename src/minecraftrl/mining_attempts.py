"""Mining attempts per run, the R0 gate and the iron verdict (multi-block design 2026-11-08 §6, §9; no game, GPU or API).

An attempt is a stretch of consecutive policy rows whose server mining block is the target family. It ends in one of:
- destroyed: the server destroy count rose and the last destroyed block is the target family;
- released: the server stopped mining while the crosshair is still on the target (attack let go);
- left: the server stopped mining because the crosshair left the target.
Only policy rows count. Success itself is the probe's server inventory check, never these counts.
"""
import json
import statistics
from pathlib import Path

from .pipeline.steps import block_matches

ENDS = ('destroyed', 'released', 'left')
FAILING_SHARE = .5          # R0 gate: a block "fails" when fewer than half of its attempts end destroyed
GATE_MIN_FAILING = 2


def attempts(rows, block):
    """Counts for one run's telemetry rows: crosshair-on-target rows, attack on them, attempt ends and lengths, and the
    attack streak just before each target destroy."""
    ends = dict.fromkeys(ENDS, 0); lengths, streaks = [], []
    on_rows = attack_rows = 0; streak = attempt = 0; prev = None
    for r in rows:
        if r.get('controller') != 'policy' or 'demo' not in r: continue
        d = r['demo']
        on = block_matches(d.get('demo_hit_block', 'air'), block)
        mining = block_matches(d.get('demo_mining_block', 'air'), block)
        attack = int((r.get('action') or {}).get('attack', 0)) == 1
        if on: on_rows += 1; attack_rows += int(attack)
        destroyed = prev is not None and d['demo_destroys'] > prev and block_matches(d.get('demo_last_destroyed_block', 'air'), block)
        if destroyed:
            streaks.append(streak)
            if attempt: ends['destroyed'] += 1; lengths.append(attempt)
            attempt = 0
        elif attempt and not mining:
            ends['released' if on else 'left'] += 1; lengths.append(attempt); attempt = 0
        if mining and not destroyed: attempt += 1      # the destroy row's stale mining block starts no new attempt
        streak = streak + 1 if attack else 0
        prev = d['demo_destroys']
    return dict(on_rows=on_rows, attack_rows=attack_rows, ends=ends, lengths=lengths, streaks=streaks)


def pool(stats):
    out = dict(on_rows=0, attack_rows=0, ends=dict.fromkeys(ENDS, 0), lengths=[], streaks=[])
    for s in stats:
        out['on_rows'] += s['on_rows']; out['attack_rows'] += s['attack_rows']
        for k in ENDS: out['ends'][k] += s['ends'][k]
        out['lengths'] += s['lengths']; out['streaks'] += s['streaks']
    return out


def destroyed_share(stats):
    """Share of attempts that ended destroyed; 0 when the policy never mined the target."""
    n = sum(stats['ends'].values())
    return stats['ends']['destroyed'] / n if n else 0.


def attack_share(stats):
    return stats['attack_rows'] / stats['on_rows'] if stats['on_rows'] else None


def r0_gate(per_block):
    """Continue only if at least GATE_MIN_FAILING blocks have a destroyed share below FAILING_SHARE under the official
    policy: when every block is already mined well, the teacher labels equal the official actions."""
    failing = sorted(b for b, s in per_block.items() if destroyed_share(s) < FAILING_SHARE)
    return dict(passed=len(failing) >= GATE_MIN_FAILING, failing_blocks=failing, threshold=FAILING_SHARE,
                min_failing=GATE_MIN_FAILING)


def dev_verdict(official, multi, iron_only=None):
    """Design §9 (and §10 without `iron_only`). Each arm: dict(success, median_s, attempts=pooled stats)."""
    so, sm = destroyed_share(official['attempts']), destroyed_share(multi['attempts'])
    share_ok = sm >= 2 * so and sm > so
    if iron_only is not None: share_ok = share_ok and sm > destroyed_share(iron_only['attempts'])
    if multi['median_s'] is None: time_ok = False
    elif official['median_s'] is None: time_ok = True
    else: time_ok = multi['median_s'] <= official['median_s']
    conditions = dict(success_not_lower=multi['success'] >= official['success'], destroyed_share=share_ok,
                      median_not_longer=time_ok)
    return dict(improved=all(conditions.values()), conditions=conditions,
                shares=dict(official=so, multi=sm, iron_only=None if iron_only is None else destroyed_share(iron_only['attempts'])))


def hold_verdict(official, baseline, new):
    """Press design 2026-11-10 §5: the new student holds better than the baseline student when its success is not
    lower than the baseline's, its destroyed share is higher than the baseline's, and its median success time is not
    longer than the official policy's. Each arm: dict(success, median_s, attempts=pooled stats)."""
    so, sb, sn = (destroyed_share(a['attempts']) for a in (official, baseline, new))
    if new['median_s'] is None: time_ok = False
    elif official['median_s'] is None: time_ok = True
    else: time_ok = new['median_s'] <= official['median_s']
    conditions = dict(success_not_lower=new['success'] >= baseline['success'], destroyed_share_higher=sn > sb,
                      median_not_longer=time_ok)
    return dict(improved=all(conditions.values()), conditions=conditions, shares=dict(official=so, baseline=sb, new=sn))


def run_block(run_dir):
    """(block, success, attempts) of one probe run folder: the plan's one step names the block."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    request = json.loads((run_dir / 'request.json').read_text(encoding='utf-8'))
    plan = request.get('plan')
    if not isinstance(plan, list) or len(plan) != 1: raise ValueError('expected a one-step plan in %s' % run_dir)
    block = plan[0].get('block') or plan[0]['target']
    rows = [json.loads(line) for line in (run_dir / 'telemetry.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    return block, bool(summary.get('execution_valid') and summary.get('all_steps_completed')), attempts(rows, block)


def median_or_none(values):
    return statistics.median(values) if values else None
