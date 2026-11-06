"""Offline replay of monitor specs over recorded pipeline runs (plan 2026-11-05 M1+M2; no game, GPU or API).

Steps are cut from `timeline.jsonl` (`step_started` / `step_finished`); signals are recomputed from `telemetry.jsonl`
through the live adapter `rm.sample_from`. Records are read only. The live pipeline is not changed.
"""
import hashlib
import json
import math
import statistics
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path

from ..stall_detection import StallDetector, StallSample
from .monitor_spec import load_spec
from .rm import sample_from
from .steps import block_matches, item_count
from .viewer import read_jsonl

AFTER_FOUND = ':after_found'
FOUND_MODES = ('first_hit', 'engaged')
CLOCKS = ('wall', 'ticks')
MOVE_KEYS = ('forward', 'back', 'left', 'right')
STUCK_TICKS, STUCK_DISTANCE = 40, .5   # moving_but_stuck: under half a block in the last 40 server ticks
LOOK_DOWN_PITCH = 60.                  # looking_down: pitch above 60 degrees
REVISIT_MOVE, CELL = .1, 2.            # revisiting_area: a move into a 2-block cell visited and left before


@dataclass(frozen=True)
class ReplayOptions:
    """Replay v2 choices (plan 2026-11-07 task B/C). The defaults are the M2 (v1) replay.

    found: 'first_hit' starts the after-found clock at the first crosshair contact; 'engaged' times each span of
      contact (crosshair or mining on the target) and ends a span after `lost_ticks` without contact.
    clock: 'wall' uses recorded seconds; 'ticks' uses server ticks / 20 (holds freeze the world, so they drop out).
    separate_goal_budget: the stall detector reports a position stall even after the goal budget expired.
    """
    found: str = 'first_hit'
    lost_ticks: int = 100
    clock: str = 'wall'
    separate_goal_budget: bool = False

    def __post_init__(self):
        if self.found not in FOUND_MODES: raise ValueError(f'found must be one of {FOUND_MODES}')
        if self.clock not in CLOCKS: raise ValueError(f'clock must be one of {CLOCKS}')
        if isinstance(self.lost_ticks, bool) or not isinstance(self.lost_ticks, int) or self.lost_ticks < 1:
            raise ValueError('lost_ticks must be a positive integer')


V1 = ReplayOptions()


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


def step_rows(run, step, controller=None):
    """Rows recorded inside the step (one controller, or all). A hold freezes the step counter while the slow path
    runs, so the next step starts at the same number: the recorded clock decides, and the row that ended a step
    stays with it."""
    def mine(r): return controller is None or r.get('controller') == controller
    if step.end_step is None:
        return [r for r in run.rows if mine(r) and r['step'] >= step.start_step and r['t'] >= step.start_t]
    return [r for r in run.rows if mine(r) and step.start_step <= r['step'] <= step.end_step
            and step.start_t <= r['t'] <= step.end_t]


def policy_rows(run, step):
    return step_rows(run, step, 'policy')


def row_clock(row, clock):
    """A row's time on the given clock: recorded seconds, or server ticks / 20."""
    return row['t'] if clock == 'wall' else int(row.get('world_time', 0)) / 20.


def step_span(run, step, options):
    """(start, end, clock used). The tick clock spans the step's first and last valid rows; a step without valid
    rows keeps the wall clock."""
    if options.clock == 'ticks':
        ticks = [int(r.get('world_time', 0)) for r in step_rows(run, step) if sample_from_row(r).valid]
        if ticks: return ticks[0] / 20., ticks[-1] / 20., 'ticks'
    return step.start_t, step.end_t, 'wall'


def engagements(rows, wanted, lost_ticks):
    """Spans of contact with the wanted block (crosshair or mining) as (first contact row, last contact row,
    row where the loss was seen or None if still open). A gap of `lost_ticks` without contact ends a span."""
    spans, cur = [], None
    for r in rows:
        s = sample_from_row(r)
        if not s.valid: continue
        contact = block_matches(s.hit_block, wanted) or block_matches(s.mining_block, wanted)
        if cur is not None and s.tick - int(cur[1].get('world_time', 0)) >= lost_ticks:
            spans.append((cur[0], cur[1], r)); cur = None
        if contact:
            if cur is None: cur = [r, r]
            else: cur[1] = r
    if cur is not None: spans.append((cur[0], cur[1], None))
    return spans


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


def after_found_start(rows, step, options):
    """The row that starts the after-found clock of a completed step: the first contact (v1) or the start of the
    last contact span ('engaged'); None if the target was never touched."""
    wanted = step.block or step.target
    if options.found == 'first_hit':
        return next((r for r in rows if (lambda s: s.valid and block_matches(s.hit_block, wanted))(sample_from_row(r))),
                    None)
    spans = engagements(rows, wanted, options.lost_ticks)
    return spans[-1][0] if spans else None


def fit_times(runs, spec, *, min_records=3, options=V1):
    """Bounded sub-goals: duration per unit. until_found: only the part after the target was seen, under
    '<key>:after_found' (the search itself has no time). Fewer than `min_records` completions make no entry."""
    samples = {}
    for key, sub in spec.subgoals.items(): samples[key + AFTER_FOUND if sub.ending == 'until_found' else key] = []
    for run in runs:
        for step in run.steps:
            sub = spec.for_step(step.skill, step.target)
            if sub is None or step.status != 'completed': continue
            units = unit_count(step, sub)
            start, end, clock = step_span(run, step, options)
            if sub.ending == 'bounded': samples[sub.key].append((end - start) / units); continue
            found = after_found_start(policy_rows(run, step), step, options)
            if found is not None: samples[sub.key + AFTER_FOUND].append((end - row_clock(found, clock)) / units)
    table = {k: TimeEntry(len(v), min(v), max(v)) for k, v in samples.items() if len(v) >= min_records}
    return table, {k: len(v) for k, v in samples.items() if len(v) < min_records}


def stall_rows(run, step, rows, until_t=None, separate_goal_budget=False):
    """The live search detector (FastMonitor fields, invalid samples skipped) over policy rows before `until_t`;
    yields (row, reason) for each stall."""
    detector, prev = StallDetector(separate_goal_budget=separate_goal_budget), None
    for r in rows:
        if until_t is not None and r['t'] >= until_t: return
        s = sample_from_row(r)
        if not s.valid: continue
        removed = s.last_destroyed_pos if prev is not None and s.destroys > prev.destroys else None
        prev = s
        report = detector.update(StallSample(tick=s.tick, position=s.position, yaw=s.yaw, stage='search',
                                             action=s.action_keys, inventory_count=item_count(s.inventory, step.target),
                                             block_removed=removed, context_id=run.name))
        if report['event'] == 'stalled': yield r, report['reason']


def stall_events(run, step, rows, until_t=None):
    """(recorded time, reason) for each stall of the live detector (v1)."""
    for r, reason in stall_rows(run, step, rows, until_t): yield r['t'], reason


def first_stall(run, step, rows, until_t, anomaly):
    """The first stall whose reason the spec names, as (seconds into the step, reason)."""
    return next(((t - step.start_t, reason) for t, reason in stall_events(run, step, rows, until_t) if reason in anomaly),
                None)


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


def replay_step(run, step, sub, table, margin, options=V1):
    """When a check would have fired, in seconds from the step start on the chosen clock (None if never)."""
    t0, t1, clock = step_span(run, step, options)
    duration = t1 - t0
    out = dict(run=run.name, step=step.id, skill=step.skill, status=step.status, ending=sub.ending if sub else None,
               duration_s=duration, found_s=None, check_s=None, check_reason=None, saved_s=None, waste={}, note=None,
               clock=clock)
    if sub is None: return dict(out, note='no_spec')
    rows = policy_rows(run, step); units = unit_count(step, sub)
    out['waste'] = waste_fractions(rows, sub.waste)
    entry = table.get(sub.key + AFTER_FOUND if sub.ending == 'until_found' else sub.key)
    if entry is None: out['note'] = 'no_records'
    allowed = None if entry is None else entry.worst * (1 + margin) * units
    candidates = []
    if sub.ending == 'bounded':
        if allowed is not None and duration > allowed: candidates.append((allowed, 'over_check_time'))
    else:
        wanted = step.block or step.target
        first = after_found_start(rows, step, V1)
        out['found_s'] = None if first is None else row_clock(first, clock) - t0
        if options.found == 'first_hit':
            windows = [] if first is None else [(out['found_s'], duration)]
            until_t = None if first is None else first['t']
        else:
            windows = [(row_clock(a, clock) - t0, duration if c is None else row_clock(c, clock) - t0)
                       for a, _, c in engagements(rows, wanted, options.lost_ticks)]
            until_t = None
        stall = next(((row_clock(r, clock) - t0, reason)
                      for r, reason in stall_rows(run, step, rows, until_t, options.separate_goal_budget)
                      if reason in sub.anomaly and not any(a <= row_clock(r, clock) - t0 <= b for a, b in windows)), None)
        if stall: candidates.append(stall)
        if allowed is not None:
            over = next((a + allowed for a, b in windows if b - a > allowed), None)
            if over is not None: candidates.append((over, 'over_check_time'))
    if candidates:
        check, reason = min(candidates)
        out.update(check_s=check, check_reason=reason, saved_s=duration - check)
    return out


STALL_KINDS = ('no_progress', 'repeated_cycle', 'stationary_repetition', 'no_goal_progress')
FIDELITY_S = 2.
CATEGORIES = ('problem', 'success', 'other')


def run_label(run):
    return f'{run.path.parent.name}/{run.name}'


def category(r):
    """problem: hit the run time limit. success: completed, or an until_found step that saw its target.
    A step the record cut off (no step_finished) is neither."""
    if r['status'] == 'unfinished': return 'other'
    if r['status'] == 'time_limit': return 'problem'
    if r['status'] == 'completed' or (r['ending'] == 'until_found' and r['found_s'] is not None): return 'success'
    return 'other'


def _summary(rows):
    out = {}
    for cat in CATEGORIES:
        group = [r for r in rows if category(r) == cat]
        checked = [r for r in group if r['check_s'] is not None]
        entry = {'n': len(group)}
        if cat == 'problem':
            saved = [r['saved_s'] for r in checked]
            entry.update(detected=len(checked), saved_s_median=statistics.median(saved) if saved else None,
                         saved_s_total=sum(saved), reasons=dict(Counter(r['check_reason'] for r in checked)))
        elif cat == 'success':
            entry.update(false_checks=len(checked), reasons=dict(Counter(r['check_reason'] for r in checked)))
        out[cat] = entry
    return out


def _replay_all(runs, spec, table, margin, options):
    return [dict(replay_step(run, step, spec.for_step(step.skill, step.target), table, margin, options), run=run_label(run))
            for run in runs for step in run.steps]


def evaluate(runs, spec, table, margins, options=V1):
    """Per margin: detections and saved time on problem steps, false checks on successes, per skill too.
    `steps` lists every replayed step at the spec's own margin."""
    result = {'margins': {}, 'steps': _replay_all(runs, spec, table, spec.margin, options)}
    for margin in margins:
        rows = result['steps'] if margin == spec.margin else _replay_all(runs, spec, table, margin, options)
        skills = sorted({r['skill'] for r in rows})
        result['margins'][f'{margin:g}'] = dict(_summary(rows),
                                                by_skill={k: _summary([r for r in rows if r['skill'] == k]) for k in skills})
    return result


def stall_fidelity(run):
    """Recorded stall events (monitor_event, or anomaly for hold kinds) that the replayed live detector reproduces:
    same step, same kind, within FIDELITY_S seconds."""
    timeline = read_jsonl(run.path / 'timeline.jsonl')
    out = dict(recorded=0, reproduced=0, replayed=0, missed=[])
    for step in run.steps:
        events = list(stall_events(run, step, policy_rows(run, step)))
        out['replayed'] += len(events)
        for e in timeline:
            kind = (e.get('event') or {}).get('kind')
            if e.get('kind') not in ('monitor_event', 'anomaly') or e.get('step_id') != step.id or kind not in STALL_KINDS:
                continue
            if not step.start_t <= e['t'] <= step.end_t: continue
            out['recorded'] += 1
            if any(k == kind and abs(t - e['t']) <= FIDELITY_S for t, k in events): out['reproduced'] += 1
            else: out['missed'].append(dict(step=step.id, kind=kind, at_s=e['t'] - step.start_t))
    return out


class ReplayRejected(ValueError):
    pass


def _usable(dirs):
    keep, skipped = [], []
    for d in map(Path, dirs):
        (keep if d.is_dir() and (d / 'timeline.jsonl').exists() and (d / 'telemetry.jsonl').exists() else skipped).append(d)
    return keep, [str(d) for d in skipped]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inputs(runs):
    return [dict(dir=str(r.path), seed=r.seed, steps=len(r.steps), timeline_sha256=sha256(r.path / 'timeline.jsonl'),
                 telemetry_sha256=sha256(r.path / 'telemetry.jsonl')) for r in runs]


def _fidelity(runs):
    per_run = {run_label(r): stall_fidelity(r) for r in runs}
    total = {k: sum(v[k] for v in per_run.values()) for k in ('recorded', 'reproduced', 'replayed')}
    return dict(total, runs=per_run)


def build_report(spec_path, fit_dirs, eval_dirs, *, margins=(0.2, 0.5, 1.0), min_records=3, options=V1):
    """Times are fitted on the fit runs only and applied to the eval runs. Shared folders or seeds are refused."""
    spec = load_spec(spec_path)
    overlap = {Path(d).resolve() for d in fit_dirs} & {Path(d).resolve() for d in eval_dirs}
    if overlap: raise ReplayRejected('folders in both fit and eval: ' + ', '.join(sorted(map(str, overlap))))
    fit_paths, skipped_fit = _usable(fit_dirs); eval_paths, skipped_eval = _usable(eval_dirs)
    if not fit_paths or not eval_paths: raise ReplayRejected('need at least one fit run and one eval run with records')
    fit_runs, eval_runs = [load_run(p) for p in fit_paths], [load_run(p) for p in eval_paths]
    shared = {r.seed for r in fit_runs} & {r.seed for r in eval_runs}
    if shared: raise ReplayRejected('seeds in both fit and eval: ' + ', '.join(map(str, sorted(shared, key=str))))
    table, insufficient = fit_times(fit_runs, spec, min_records=min_records, options=options)
    margins = tuple(sorted(set(margins) | {spec.margin}))
    here = Path(__file__).resolve().parent
    return dict(
        spec=dict(path=str(spec_path), version=spec.version, author=spec.author, margin=spec.margin),
        options=asdict(options),
        inputs=dict(fit=_inputs(fit_runs), eval=_inputs(eval_runs), skipped=skipped_fit + skipped_eval),
        source={'monitor_spec.py': sha256(here / 'monitor_spec.py'), 'monitor_replay.py': sha256(here / 'monitor_replay.py'),
                'spec': sha256(spec_path)},
        min_records=min_records, times={k: asdict(v) for k, v in sorted(table.items())}, insufficient=insufficient,
        evaluation=evaluate(eval_runs, spec, table, margins, options),
        fidelity=dict(fit=_fidelity(fit_runs), eval=_fidelity(eval_runs)))


def _num(v, digits=1):
    return '-' if v is None else f'{v:.{digits}f}'


def _seeds(runs):
    return ', '.join(str(s) for s in sorted({r['seed'] for r in runs}, key=str))


def report_markdown(report):
    spec, ev, inputs = report['spec'], report['evaluation'], report['inputs']
    lines = [f"# 감시 명세 재생 결과 ({spec['version']})", '',
             f"- 개발 기록 {len(inputs['fit'])}개 (seed {_seeds(inputs['fit'])}): 기준 시간만 정한다.",
             f"- 평가 기록 {len(inputs['eval'])}개 (seed {_seeds(inputs['eval'])}): 감지와 오탐만 잰다.",
             f"- 건너뛴 폴더 {len(inputs['skipped'])}개 (기록 파일 없음). 명세 여유 {spec['margin']:g}.",
             '- 재생 선택: ' + ', '.join(f'{k}={v}' for k, v in report.get('options', asdict(V1)).items()), '',
             '## 여유별 요약', '',
             '| 여유 | 문제 단계 | 감지 | 아낄 수 있던 시간 중앙값(s) | 합계(s) | 성공 단계 | 오탐 | 기타 |',
             '|---|---|---|---|---|---|---|---|']
    for m, s in ev['margins'].items():
        p, ok = s['problem'], s['success']
        lines.append(f"| {m} | {p['n']} | {p['detected']} | {_num(p['saved_s_median'])} | {_num(p['saved_s_total'])} | "
                     f"{ok['n']} | {ok['false_checks']} | {s['other']['n']} |")
    own = ev['margins'][f"{spec['margin']:g}"]
    lines += ['', f"## 스킬별 (여유 {spec['margin']:g})", '', '| 스킬 | 문제 | 감지 | 성공 | 오탐 | 기타 |',
              '|---|---|---|---|---|---|']
    for skill, s in own['by_skill'].items():
        lines.append(f"| {skill} | {s['problem']['n']} | {s['problem']['detected']} | {s['success']['n']} | "
                     f"{s['success']['false_checks']} | {s['other']['n']} |")
    lines += ['', '점검 사유: 문제 ' + json.dumps(own['problem']['reasons'], ensure_ascii=False) + ', 오탐 ' +
              json.dumps(own['success']['reasons'], ensure_ascii=False), '',
              '## 기준 시간 (개발 기록, 단위당 초)', '', '| 키 | n | 최고 | 최악 |', '|---|---|---|---|']
    lines += [f"| {k} | {v['n']} | {v['best']:.1f} | {v['worst']:.1f} |" for k, v in report['times'].items()]
    lines += ['', f"## 기록 부족 (완료 {report['min_records']}개 미만, 기준 없음)", '']
    lines += [f'- {k}: {n}개' for k, n in sorted(report['insufficient'].items())] or ['- 없음']
    lines += ['', '## 재현도 (기록된 정체 이벤트를 다시 계산해 ±2초 안에 같은 종류가 나온 수)', '']
    for name, label in (('fit', '개발'), ('eval', '평가')):
        f = report['fidelity'][name]
        lines.append(f"- {label}: 기록 {f['recorded']}개 중 재현 {f['reproduced']}개 (재생에서 나온 정체 {f['replayed']}개)")
    return '\n'.join(lines) + '\n'
