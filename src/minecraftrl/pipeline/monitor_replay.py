"""Offline replay of monitor specs over recorded pipeline runs (plan 2026-11-05 M1+M2; no game, GPU or API).

Steps are cut from `timeline.jsonl` (`step_started` / `step_finished`); signals are recomputed from `telemetry.jsonl`
through the live adapter `rm.sample_from`. Records are read only. The live pipeline is not changed.
"""
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from ..stall_detection import StallDetector, StallSample
from .rm import sample_from
from .steps import block_matches, item_count
from .viewer import read_jsonl

AFTER_FOUND = ':after_found'
MOVE_KEYS = ('forward', 'back', 'left', 'right')
STUCK_TICKS, STUCK_DISTANCE = 40, .5   # moving_but_stuck: under half a block in the last 40 server ticks
LOOK_DOWN_PITCH = 60.                  # looking_down: pitch above 60 degrees
REVISIT_MOVE, CELL = .1, 2.            # revisiting_area: a move into a 2-block cell visited and left before


@dataclass(frozen=True)
class StepRecord:
    run: str
    id: str
    skill: str
    target: str | None
    block: str | None
    count: int
    start_t: float
    end_t: float
    start_step: int
    end_step: int | None
    status: str


@dataclass
class Run:
    name: str
    path: Path
    seed: int | None
    steps: list
    rows: list


def _steps(name, timeline):
    """A restarted id is a new step; a step with no `step_finished` is 'unfinished' and ends at the last record."""
    steps, open_steps = [], {}
    last_t = timeline[-1]['t'] if timeline else 0.
    for e in timeline:
        if e.get('kind') == 'step_started': open_steps[e['id']] = e
        elif e.get('kind') == 'step_finished' and e.get('id') in open_steps:
            s = open_steps.pop(e['id'])
            steps.append(StepRecord(name, s['id'], s['skill'], s.get('target'), s.get('block'), int(s.get('count', 1)),
                                    float(s['t']), float(e['t']), int(s['start_step']), int(e['step']), e['status']))
    for s in open_steps.values():
        steps.append(StepRecord(name, s['id'], s['skill'], s.get('target'), s.get('block'), int(s.get('count', 1)),
                                float(s['t']), float(last_t), int(s['start_step']), None, 'unfinished'))
    return sorted(steps, key=lambda s: (s.start_t, s.start_step))


def load_run(run_dir):
    path = Path(run_dir)
    request = path / 'request.json'
    seed = json.loads(request.read_text(encoding='utf-8')).get('seed') if request.exists() else None
    return Run(path.name, path, seed, _steps(path.name, read_jsonl(path / 'timeline.jsonl')),
               read_jsonl(path / 'telemetry.jsonl'))


def sample_from_row(row):
    """The live adapter on a recorded row: telemetry keeps the server inventory under `inventory`."""
    return sample_from({'workshop': {**row, 'server_inventory': row.get('inventory') or {}}, 'demo': row.get('demo') or {}},
                       row.get('action') or {}, row['t'])


def policy_rows(run, step):
    return [r for r in run.rows if r.get('controller') == 'policy' and r['step'] >= step.start_step
            and (step.end_step is None or r['step'] < step.end_step)]


@dataclass(frozen=True)
class TimeEntry:
    """Seconds per unit over completed dev records: best is the target time, worst sets the check time."""
    n: int
    best: float
    worst: float


def unit_count(step, sub):
    return 1 if sub.unit == 'step' else step.count


def found_time(rows, step):
    """The first recorded moment the crosshair is on the wanted block (server-observed hit block)."""
    wanted = step.block or step.target
    for r in rows:
        s = sample_from_row(r)
        if s.valid and block_matches(s.hit_block, wanted): return r['t']
    return None


def fit_times(runs, spec, *, min_records=3):
    """Bounded sub-goals: duration per unit. until_found: only the part after the target was seen, under
    '<key>:after_found' (the search itself has no time). Fewer than `min_records` completions make no entry."""
    samples = {}
    for key, sub in spec.subgoals.items(): samples[key + AFTER_FOUND if sub.ending == 'until_found' else key] = []
    for run in runs:
        for step in run.steps:
            sub = spec.for_step(step.skill, step.target)
            if sub is None or step.status != 'completed': continue
            units = unit_count(step, sub)
            if sub.ending == 'bounded': samples[sub.key].append((step.end_t - step.start_t) / units); continue
            found = found_time(policy_rows(run, step), step)
            if found is not None: samples[sub.key + AFTER_FOUND].append((step.end_t - found) / units)
    table = {k: TimeEntry(len(v), min(v), max(v)) for k, v in samples.items() if len(v) >= min_records}
    return table, {k: len(v) for k, v in samples.items() if len(v) < min_records}


def first_stall(run, step, rows, until_t, anomaly):
    """The live search detector (FastMonitor fields, invalid samples skipped) over policy rows before `until_t`;
    the first stall whose reason the spec names, as (seconds into the step, reason)."""
    detector, prev = StallDetector(), None
    for r in rows:
        if until_t is not None and r['t'] >= until_t: break
        s = sample_from_row(r)
        if not s.valid: continue
        removed = s.last_destroyed_pos if prev is not None and s.destroys > prev.destroys else None
        prev = s
        report = detector.update(StallSample(tick=s.tick, position=s.position, yaw=s.yaw, stage='search',
                                             action=s.action_keys, inventory_count=item_count(s.inventory, step.target),
                                             block_removed=removed, context_id=run.name))
        if report['event'] == 'stalled' and report['reason'] in anomaly: return r['t'] - step.start_t, report['reason']
    return None


def waste_fractions(rows, names):
    """Share of valid policy rows showing each named waste signal (None when the step has no valid rows)."""
    samples = [s for s in map(sample_from_row, rows) if s.valid]
    if not samples: return {name: None for name in names}
    counts = dict.fromkeys(names, 0)
    history, left, prev = deque(), set(), None
    for s in samples:
        x, z = s.position[0], s.position[2]
        while len(history) > 1 and history[1].tick <= s.tick - STUCK_TICKS: history.popleft()
        ref = history[0] if history and history[0].tick <= s.tick - STUCK_TICKS else None
        history.append(s)
        moving = any(k in s.action_keys for k in MOVE_KEYS) and 'attack' not in s.action_keys
        if moving and ref is not None and math.hypot(x - ref.position[0], z - ref.position[2]) < STUCK_DISTANCE:
            counts['moving_but_stuck'] = counts.get('moving_but_stuck', 0) + 1
        if s.pitch > LOOK_DOWN_PITCH: counts['looking_down'] = counts.get('looking_down', 0) + 1
        cell = (math.floor(x / CELL), math.floor(z / CELL))
        if prev is not None:
            prev_cell = (math.floor(prev.position[0] / CELL), math.floor(prev.position[2] / CELL))
            if cell != prev_cell: left.add(prev_cell)
            if math.hypot(x - prev.position[0], z - prev.position[2]) >= REVISIT_MOVE and cell in left:
                counts['revisiting_area'] = counts.get('revisiting_area', 0) + 1
        prev = s
    return {name: counts[name] / len(samples) for name in names}


def replay_step(run, step, sub, table, margin):
    duration = step.end_t - step.start_t
    out = dict(run=run.name, step=step.id, skill=step.skill, status=step.status, ending=sub.ending if sub else None,
               duration_s=duration, found_s=None, check_s=None, check_reason=None, saved_s=None, waste={}, note=None)
    if sub is None: return dict(out, note='no_spec')
    rows = policy_rows(run, step); units = unit_count(step, sub)
    out['waste'] = waste_fractions(rows, sub.waste)
    candidates, start = [], 0.
    if sub.ending == 'until_found':
        found = found_time(rows, step)
        out['found_s'] = None if found is None else found - step.start_t
        stall = first_stall(run, step, rows, found, sub.anomaly)
        if stall: candidates.append(stall)
        entry, start = table.get(sub.key + AFTER_FOUND), out['found_s']
    else: entry = table.get(sub.key)
    if entry is None: out['note'] = 'no_records'
    elif start is not None:
        limit = start + entry.worst * (1 + margin) * units
        if duration > limit: candidates.append((limit, 'over_check_time'))
    if candidates:
        check, reason = min(candidates)
        out.update(check_s=check, check_reason=reason, saved_s=duration - check)
    return out
