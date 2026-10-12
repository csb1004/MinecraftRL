"""Measured single-episode execution. No API-policy fallback."""
import json
import math
import threading
import time
import traceback
from pathlib import Path


def jsonable(value):
    if hasattr(value, 'tolist'): return value.tolist()
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def quantile(values, q):
    if not values: return None
    ordered = sorted(values); position = (len(ordered) - 1) * q
    lo = int(position); hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def distribution(values):
    return {'count': len(values), 'median': quantile(values, .5), 'p95': quantile(values, .95)}


def inventory_counts(obs):
    inv = obs.get('inventory')
    if not isinstance(inv, dict): raise ValueError('inventory observation missing or not a mapping')
    result = {}
    for item, value in inv.items():
        if hasattr(value, 'item'): value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError('invalid inventory count: ' + str(item))
        result[str(item).removeprefix('minecraft:') if hasattr(str, 'removeprefix') else str(item).replace('minecraft:', '', 1)] = int(value)
    return result


class StopGate:
    """Stop is latched. An already-dispatched step may finish; no next step starts.

    The return of stop() acknowledges quiescence of dispatch. Neutral release is
    cleanup, not a policy action. Never use this gate for OS-global key injection.
    """
    def __init__(self):
        self.requested = threading.Event(); self._lock = threading.RLock(); self.reason = None

    def stop(self, reason='cancelled'):
        self.requested.set()
        with self._lock:
            if self.reason is None: self.reason = reason

    def dispatch(self, fn):
        with self._lock:
            if self.requested.is_set(): return False, None
            return True, fn()


def save_png(frame, path):
    from PIL import Image
    Image.fromarray(frame).save(path)


def run_episode(policy, env, output, *, seed, goal, target_items=(), target_count=1,
                limit_s=180., gate=None, stop_file=None, clock=time.monotonic,
                save_screen=save_png, synchronize=lambda: None, gpu_metrics=lambda: {}, completion_check=None,
                completion_status='target_reached',on_control_closed=None):
    if completion_status not in ('target_reached','craft_ready_for_recovery'): raise ValueError('invalid completion status')
    if not 0 < limit_s <= 180: raise ValueError('limit_s must be in (0, 180]')
    if target_count < 1: raise ValueError('target_count must be positive')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frames = output / 'frames'; frames.mkdir(exist_ok=True)
    gate = gate or StopGate()
    report = {'seed': seed, 'goal': goal, 'controller': 'policy', 'status': 'initializing',
              'steps': 0, 'execution_valid': False, 'target_observed': False,
              'inventory_peak_gain': {}, 'inventory_final_delta': {}, 'release': 'not_attempted',
              'stop_contract': 'no new policy dispatch after stop acknowledgment; an in-flight step may finish'}
    inference = []; context_encoding = []; cycles = []; baseline = {}; peak = {}; latest = {}; started = None
    log = (output / 'events.jsonl').open('w', encoding='utf-8')
    def emit(event, **fields):
        log.write(json.dumps(jsonable({'event': event, 'monotonic_s': clock(), **fields}), ensure_ascii=False, allow_nan=False) + '\n'); log.flush()
    def stop_reason():
        if stop_file and Path(stop_file).exists(): gate.stop('stop_file')
        if started is not None and clock() - started >= limit_s: gate.stop('time_limit')
        return gate.reason if gate.requested.is_set() else None
    try:
        env.seed(seed)
        obs = env.reset()
        policy.reset(goal)
        baseline = inventory_counts(obs); latest = dict(baseline)
        started = clock(); emit('episode_started', seed=seed, goal=goal, baseline_inventory=baseline)
        while not gate.requested.is_set():
            if stop_reason(): break
            cycle_start = clock()
            prepare = getattr(policy, 'prepare_control', None)
            if prepare is not None:
                prepare(elapsed=lambda: clock()-started, limit_s=limit_s,
                        should_stop=stop_reason, request_stop=gate.stop, emit=emit)
                if stop_reason(): break
            context_only = getattr(policy, 'next_action_kind', 'policy') == 'context_only'
            event_prefix = 'context_encoding' if context_only else 'inference'
            frame_path = frames / ('%06d.png' % report['steps'])
            save_screen(obs['pov'], frame_path)
            synchronize()
            emit(event_prefix+'_started', step=report['steps'] + 1)
            before = clock()
            action = policy.act(obs['pov'])
            synchronize(); inference_ended = clock()
            inference_ms = (inference_ended - before) * 1000
            (context_encoding if context_only else inference).append(inference_ms)
            emit(event_prefix+'_finished', step=report['steps'] + 1,
                 call_started_s=before, call_finished_s=inference_ended)
            if stop_reason():
                emit('action_discarded', reason=gate.reason); break
            controller = getattr(policy, 'controller_name', 'policy')
            emit('action_ready', action=action, frame=str(frame_path.relative_to(output)), controller=controller)
            # Check deadline immediately before dispatch, including logging overhead.
            if stop_reason():
                emit('action_discarded', reason=gate.reason); break
            def step_environment():
                dispatch_prefix = 'control_dispatch' if context_only else 'policy_dispatch'
                emit(dispatch_prefix+'_started', step=report['steps'] + 1, controller=controller)
                step_started = clock()
                transition = env.step(action)
                step_finished = clock()
                emit(dispatch_prefix+'_finished', step=report['steps'] + 1, controller=controller,
                     call_started_s=step_started, call_finished_s=step_finished)
                return transition
            dispatched, transition = gate.dispatch(step_environment)
            if not dispatched: break
            report['steps'] += 1
            obs, reward, done, info = transition
            latest = inventory_counts(obs)
            for item in set(baseline) | set(latest):
                peak[item] = max(peak.get(item, 0), latest.get(item, 0) - baseline.get(item, 0))
            emit('transition', step=report['steps'], controller=controller, action=action, reward=reward, done=done,
                 observation={k: v for k, v in obs.items() if k != 'pov'}, info=info,
                 inference_ms=None if context_only else inference_ms,
                 context_encoding_ms=inference_ms if context_only else None, gpu=gpu_metrics())
            # Includes capture, inference, environment step and transition log flush.
            cycles.append((clock() - cycle_start) * 1000)
            emit('cycle_timing', step=report['steps'], cycle_ms=cycles[-1])
            if completion_check is not None and completion_check(obs):
                report['status']=completion_status
                gate.stop(completion_status)
                emit(completion_status,step=report['steps'])
                break
            if done:
                report['status'] = 'environment_done'; break
        if report['status'] == 'initializing':
            report['status'] = gate.reason if gate.reason in ('time_limit','planning_paused') else 'cancelled'
        if 'pov' in obs: save_screen(obs['pov'], frames / 'final.png')
        report['execution_valid'] = report['steps'] > 0 and report['status'] in ('environment_done', 'time_limit', 'target_reached','craft_ready_for_recovery')
    except BaseException as exc:
        # A policy may observe STOP after the loop's check and before inference.
        # Its guard is expected cancellation, not a backend failure. Preserve real
        # interruptions without an observed stop as errors, with their traceback.
        reason=stop_reason() if isinstance(exc,InterruptedError) else None
        if reason:
            report['status']=reason if reason in ('time_limit','planning_paused') else 'cancelled'
            report['control_interrupted']=dict(type=type(exc).__name__,reason=reason)
        else:
            report['status'] = 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'error'
            report['error'] = type(exc).__name__ + ': ' + str(exc)
            (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        control_ended = clock()
        gate.stop(report['status'])
        emit('stop_acknowledged', reason=gate.reason, steps=report['steps'])
        try:
            release_status = env.release(); report['release'] = release_status or 'neutral_command_sent'
            emit('input_release', controller='cleanup', outcome=report['release'])
        except BaseException as exc:
            report['release'] = 'failed: ' + repr(exc)
            report['execution_valid'] = False
        resources=([('policy',policy),('environment',env)] if on_control_closed is not None else
                   [('environment',env),('policy',policy)])
        for label, resource in resources:
            try: resource.close()
            except BaseException as exc:
                report[label + '_close_error'] = repr(exc); report['execution_valid'] = False
            if (label=='policy' and on_control_closed is not None and 'policy_close_error' not in report
                    and report['release']=='neutral_command_sent' and 'error' not in report):
                try:on_control_closed(report)
                except Exception as exc:
                    report['finalization_marker_error']=repr(exc);report['execution_valid']=False
        report.update(elapsed_control_s=None if started is None else control_ended - started,
                      elapsed_cleanup_s=clock() - control_ended,
                      inference_ms=distribution(inference), context_encoding_ms=distribution(context_encoding),
                      control_cycle_ms=distribution(cycles),
                      inventory_peak_gain={k: v for k, v in peak.items() if v},
                      inventory_final_delta={k: latest.get(k, 0) - baseline.get(k, 0) for k in set(latest) | set(baseline)
                                             if latest.get(k, 0) != baseline.get(k, 0)}, stop_reason=gate.reason)
        report['target_observed'] = bool(target_items) and sum(peak.get(k, 0) for k in target_items) >= target_count
        report['gpu'] = gpu_metrics()
        emit('episode_finished', status=report['status']); log.close()
        write_json(output / 'summary.json', report)
    return report
