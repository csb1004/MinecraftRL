"""Record check of the signals a monitor spec chose (plan 2026-11-05 §7.2; task D 2026-11-07).

A signal is useful only if it is not normal behaviour for its sub-goal. On completed dev steps this measures:
- waste: the mean share of policy rows showing the signal (monitor_replay.waste_fractions);
- anomaly: the share of steps where it fired, from the replayed stall detector for stall reasons and from recorded
  monitor events or holds for the other signals.
A signal at or above the caller's limits is flagged; `calibrated_spec` writes a copy without the flagged signals.
Fit (dev) records only; offline, no game, GPU or API.
"""
import copy
import statistics

from .monitor_replay import policy_rows, stall_rows, waste_fractions
from .viewer import read_jsonl

STALL_REASONS = ('no_progress', 'repeated_cycle', 'stationary_repetition')


def _recorded_kinds(timeline, step):
    return {(e.get('event') or {}).get('kind') for e in timeline
            if e.get('kind') in ('monitor_event', 'anomaly') and e.get('step_id') == step.id
            and step.start_t <= e['t'] <= step.end_t}


def signal_rates(runs, spec):
    acc = {key: dict(completed_steps=0, waste={s: [] for s in sub.waste}, anomaly={s: 0 for s in sub.anomaly})
           for key, sub in spec.subgoals.items()}
    for run in runs:
        timeline = None
        for step in run.steps:
            sub = spec.for_step(step.skill, step.target)
            if sub is None or step.status != 'completed': continue
            a = acc[sub.key]; a['completed_steps'] += 1
            rows = policy_rows(run, step)
            for name, share in waste_fractions(rows, sub.waste).items():
                if share is not None: a['waste'][name].append(share)
            stalls = {reason for _, reason in stall_rows(run, step, rows)} if set(sub.anomaly) & set(STALL_REASONS) else set()
            if set(sub.anomaly) - set(STALL_REASONS):
                timeline = read_jsonl(run.path / 'timeline.jsonl') if timeline is None else timeline
                recorded = _recorded_kinds(timeline, step)
            else: recorded = set()
            for name in sub.anomaly: a['anomaly'][name] += name in (stalls if name in STALL_REASONS else recorded)
    out = {}
    for key, a in acc.items():
        n = a['completed_steps']
        out[key] = dict(completed_steps=n,
                        waste={s: statistics.mean(v) if v else None for s, v in a['waste'].items()},
                        anomaly={s: {'rate': hits / n if n else None,
                                     'source': 'replayed_stall_detector' if s in STALL_REASONS else 'recorded_events'}
                                 for s, hits in a['anomaly'].items()})
    return out


def flag_signals(rates, *, waste_max, anomaly_max):
    """Signals that are normal in completed steps: waste share >= waste_max, anomaly rate >= anomaly_max."""
    flags = []
    for key, r in rates.items():
        if not r['completed_steps']: continue
        for name, share in r['waste'].items():
            if share is not None and share >= waste_max:
                flags.append(dict(key=key, role='waste', signal=name, value=share, completed_steps=r['completed_steps'],
                                  reason=f'{name} covers {share:.0%} of rows in completed {key} steps'))
        for name, a in r['anomaly'].items():
            if a['rate'] is not None and a['rate'] >= anomaly_max:
                flags.append(dict(key=key, role='anomaly', signal=name, value=a['rate'], completed_steps=r['completed_steps'],
                                  reason=f"{name} fires in {a['rate']:.0%} of completed {key} steps ({a['source']})"))
    return flags


def calibrated_spec(raw, flags, *, version):
    """A copy of the raw spec without the flagged signals, under a new version."""
    out = copy.deepcopy(raw); out['version'] = version
    for f in flags:
        names = out['subgoals'][f['key']].get(f['role'], [])
        out['subgoals'][f['key']][f['role']] = [n for n in names if n != f['signal']]
    return out
