"""Read-only adapter from recorded wood episodes to stagnation evidence."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

from .approach_recovery import ApproachRecovery
from .rl_core import LOGS, ShapedWoodRewardV3
from .stall_detection import StallDetector, StallSample


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def action_signature(action):
    pressed = []
    for key, value in sorted(action.items()):
        if key == 'camera':
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError('invalid camera action')
            for axis, angle in zip(('pitch', 'yaw'), value):
                if not isinstance(angle, (int, float)) or not math.isfinite(angle):
                    raise ValueError('invalid camera angle')
                if abs(angle) >= 1.:
                    pressed.append(axis + ('+' if angle > 0 else '-'))
        else:
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            if value not in (0, 1):
                raise ValueError('invalid button action')
            if value:
                pressed.append(key)
    return tuple(pressed)


def log_count(obs):
    values = [obs['server_inventory'].get(k, 0) for k in LOGS]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or int(v) != v for v in values):
        raise ValueError('invalid server log inventory')
    return int(sum(values))


def wood_sample(obs, action, context_id, *, previous=None, terminal=False):
    # render_ready means the global chunk update queue is empty, not fresh telemetry.
    valid = obs.get('server_available') == 1 and obs.get('error', 0) == 0
    if not valid:
        return StallSample(tick=obs.get('server_tick', 0), position=(0., 0., 0.), yaw=0.,
                           stage='search', action=(), context_id=context_id, valid=False, terminal=terminal)
    stage = ShapedWoodRewardV3.state(obs)[0]  # raw sensor, not v4's overridden search context
    target = None
    if obs['wood_v3_kind'] in (1, 2):
        target = tuple(obs['wood_v3_target_' + a] for a in 'xyz')
    removed = None
    if previous and previous.get('wood_v3_kind') == 1 and obs['wood_v3_kind'] == 2:
        old = tuple(previous['wood_v3_target_' + a] for a in 'xyz')
        if old == target:
            removed = target
    return StallSample(tick=obs['server_tick'], position=tuple(obs['server_' + a] for a in 'xyz'),
                       yaw=obs['yaw'], stage=stage, action=action_signature(action), target=target,
                       distance=obs['wood_v3_distance'] if target else None, inventory_count=log_count(obs),
                       block_removed=removed, context_id=context_id, terminal=terminal)


def replay_run(run, output):
    """Never import an environment/LLM/torch; preserve original files and STOP."""
    run = Path(run)
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    initial = json.loads((run / 'initial-state.json').read_text(encoding='utf-8'))['server']
    context = str(run)
    detector, legacy = StallDetector(), ApproachRecovery()
    first = wood_sample(initial, {}, context)
    detector.update(first)
    legacy.update(initial, first.stage)
    initial_count = log_count(initial)
    previous = initial
    alarms, legacy_entries, gaps = [], [], []
    stages, progress, timings = Counter(), Counter(), []
    transitions = 0
    stopped = False
    first_time = None
    mining_observations = 0
    with Path(output).open('x', encoding='utf-8') as dst, (run / 'events.jsonl').open(encoding='utf-8') as src:
        for line in src:
            event = json.loads(line)
            if event['event'] == 'stop_acknowledged':
                stopped = True
                detector.stop()
            if event['event'] != 'transition':
                continue
            if stopped:
                raise ValueError('recorded transition after STOP')
            transitions += 1
            if event['step'] != transitions:
                raise ValueError('nonsequential transition steps')
            obs = event['info']['workshop']
            terminal = bool(event['done'] or log_count(obs) > initial_count)
            sample = wood_sample(obs, event['action'], context, previous=previous, terminal=terminal)
            started = perf_counter()
            decision = detector.update(sample)
            timings.append((perf_counter() - started) * 1000.)
            if first_time is None:
                first_time = event['monotonic_s']
            decision.update(step=transitions, elapsed_from_first_transition_s=event['monotonic_s']-first_time,
                            stage=sample.stage, tick_from_initial=sample.tick-first.tick)
            dst.write(json.dumps(decision, ensure_ascii=False) + '\n')
            stages[sample.stage] += 1
            progress.update(decision['progress'])
            mining_observations += int(obs.get('wood_server_mining') == 1)
            if decision['event'] == 'stalled':
                alarms.append(decision)
            if decision['event'] == 'observation_gap':
                gaps.append(decision)
            old = legacy.update(obs, sample.stage, terminal=terminal)
            if old['event'] == 'entered':
                legacy_entries.append(dict(step=transitions, tick_from_initial=sample.tick-first.tick))
            previous = obs
    if transitions != summary['steps'] or not stopped:
        raise ValueError('incomplete original episode log')
    timings.sort()
    acquired = log_count(previous) - initial_count
    if bool(acquired > 0) != bool(summary['server_target_reached']):
        raise ValueError('server acquisition disagrees with summary')
    return dict(transitions=transitions, actual_inventory_delta=acquired, original_status=summary['status'],
                alarms=alarms, legacy_entries=legacy_entries, observation_gaps=gaps, stages=dict(stages),
                progress=dict(progress), mining_observations_without_damage_sensor=mining_observations,
                decision_ms=dict(median=timings[len(timings)//2], p95=timings[int((len(timings)-1)*.95)]),
                source_stop_seen=stopped, new_game_actions=0, model_updates=0,
                scope='offline counterfactual detector; no planner or environment intervention')
