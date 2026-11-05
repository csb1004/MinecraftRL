"""Offline replay of monitor specs over recorded pipeline runs (plan 2026-11-05 M1+M2; no game, GPU or API).

Steps are cut from `timeline.jsonl` (`step_started` / `step_finished`); signals are recomputed from `telemetry.jsonl`
through the live adapter `rm.sample_from`. Records are read only. The live pipeline is not changed.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from .rm import sample_from
from .viewer import read_jsonl


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
