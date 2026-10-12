"""Preserve observed limits of the handoff fix; no gameplay or training."""
import json
from collections import Counter
from pathlib import Path

from minecraftrl.runtime import write_json
from minecraftrl.stall_replay import wood_sample
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root / 'results/fresh-target-handoff-game-20261009'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
work = read(base / 'work.json')
assert work['completed'] and work['error'] is None
run = Path(work['run'])
rows = read(run / 'goal-recovery-rows.json')
attempt = read(run / 'recovery-attempts.json')[0]
assert attempt['next_phase'] == 'resumed'
start = attempt['end_step']
resumed = rows[start:]
samples = [wood_sample(r['server_after'], r['action'], str(run)) for r in resumed]
last = read(run / 'resumed-stall-decisions.json')[-1]
assert last['event'] == 'stalled'
summary = read(run / 'summary.json')
result = dict(
    source_run=str(run), handoff_step=start, resumed_transitions=len(resumed),
    raw_stages=Counter(s.stage for s in samples),
    reachable_observations=sum(r['server_after']['wood_v3_reachable'] == 1 for r in resumed),
    accepted_log_mining_observations=sum(r['server_after'].get('wood_server_mining', 0) == 1 for r in resumed),
    attack_commands=sum('attack' in s.action for s in samples),
    movement_commands=sum(any(k in s.action for k in ('forward', 'back', 'left', 'right')) for s in samples),
    target_distance_min=min(s.distance for s in samples if s.distance is not None),
    target_distance_last=samples[-1].distance,
    restall=last, log_gain=work['log_gain'],
    control_s=summary['elapsed_control_s'], cleanup_s=summary['elapsed_cleanup_s'],
    snapshots=dict(handoff_input='frames/000478.png', last_action_input='frames/000741.png'),
    observation='Fresh target acquisition was followed by 264 approach observations, without a reachable or mining/collection state.',
    interpretation='Observed target is not proof of a feasible direct path; the policy remained against the wall. Generalized detour behavior is unresolved.',
    scope='One privileged-observation development run; no new gameplay, API calls or optimizer updates.',
    source_sha256={name: digest(run / name) for name in (
        'goal-recovery-rows.json', 'recovery-attempts.json', 'resumed-stall-decisions.json',
        'summary.json', 'frames/000478.png', 'frames/000741.png')},
    script_sha256=digest(Path(__file__)),
)
destination = base / 'analysis/failure-evidence.json'
if destination.exists():
    raise FileExistsError(destination)
write_json(destination, result)
print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'restall')}))
