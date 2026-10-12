"""Planner-facing evidence, without new rewards, RM mutation or action authority.

Stagnation is a reason to consider another plan, not proof that a task is
impossible. This projects the existing detector and raw server observations;
it does not create a second progress detector or call an LLM.
"""
import math

from .runtime import jsonable
from .stall_detection import StallDetector
from .stall_replay import wood_sample


def _sample(obs):
    sample = wood_sample(obs,{},'recovery_evidence')
    if not sample.valid:
        raise ValueError('server observation unavailable')
    StallDetector().update(sample)  # Existing validation, no alternate thresholds.
    return sample


def _result(classification, **fields):
    return dict(classification=classification,replan_candidate=False,task_impossible=False,
                execution_authorized=False,rm_effect='preserve',goal_completed=False,**fields)


def _baseline(value):
    if type(value) is not int or value < 0:
        raise ValueError('authoritative nonnegative inventory baseline required')


def assess_stall(obs, decision, *, baseline_count, stopped=False):
    """An instantaneous event is advisory; scheduler must still check freshness."""
    _baseline(baseline_count)
    if stopped:
        return _result('stopped')
    try:
        sample = _sample(obs)
        if decision['tick'] != sample.tick or not decision.get('observation_valid'):
            raise ValueError('stale/gapped detector evidence')
    except (KeyError,ValueError,TypeError):
        return _result('observation_unreliable')
    if sample.inventory_count > baseline_count:
        result = _result('goal_completed'); result['goal_completed'] = True
        return result
    result = _result('progressing',facts=dict(tick=sample.tick,position=list(sample.position),
        yaw=sample.yaw,raw_stage=sample.stage,target=jsonable(sample.target),distance=sample.distance,
        goal_inventory=sample.inventory_count,mining_active=bool(obs.get('wood_server_mining',0)),
        mining_damage_observed=sample.mining_progress is not None),
        uncertainties=['path_feasibility_unmeasured','block_hardness_unmeasured'])
    if sample.stage == 'mine' and sample.mining_progress is None:
        result['uncertainties'].append('mining_damage_unmeasured')
    if decision.get('event') == 'stalled' and decision.get('stalled'):
        if decision.get('reason') not in ('no_progress','no_goal_progress','stationary_repetition','repeated_cycle'):
            raise ValueError('unknown stagnation reason')
        result.update(classification='stagnation_suspected',replan_candidate=True,
            evidence={k:jsonable(decision[k]) for k in ('tick','reason','no_progress_ticks',
                'no_goal_progress_ticks','window_ticks','cycle','action_fraction','position_span',
                'evidence_start_tick') if k in decision})
    return result


def assess_attempt(before, observations, *, reason, baseline_count, budget_ticks):
    """Assess a committed recovery window; movement alone never means success.

The caller supplies the actual boundary reason. `control_budget`/STOP cannot
authorize another plan; a skill budget can nominate replanning while the
scheduler separately checks remaining episode time, capability and identity.
"""
    _baseline(baseline_count)
    if type(budget_ticks) is not int or budget_ticks <= 0:
        raise ValueError('positive server-tick budget required')
    if reason not in ('tick_budget','planning_time_budget','wood_target_acquired','wood_interaction_ready','control_budget','stop','goal_completed','environment_done'):
        raise ValueError('unknown recovery boundary')
    if reason == 'stop':
        return _result('stopped')
    try:
        start = _sample(before); samples = [_sample(s) for s in observations]
        if not samples or len(samples) > budget_ticks:
            raise ValueError('empty/over-budget recovery window')
        previous = start
        for sample in samples:
            if sample.tick-previous.tick != 1:
                raise ValueError('noncontiguous server evidence')
            previous = sample
    except (ValueError,KeyError,TypeError):
        return _result('observation_unreliable')
    last = samples[-1]
    previous = samples[-2] if len(samples)>1 else start
    fresh_target = previous.target is None and last.target is not None and last.stage in ('approach','mine')
    distances = [s.distance for s in samples if s.target == start.target and s.distance is not None]
    approach = bool(start.target is not None and start.distance is not None and distances
                    and min(distances) <= start.distance-.5)
    observed_target = start.target is None and any(s.target is not None for s in samples)
    result = _result('recovery_incomplete',same_target_approach_progress=approach,
        current_same_target_approach_progress=bool(start.target is not None and last.target==start.target
            and start.distance is not None and last.distance is not None and last.distance<=start.distance-.5),
        target_became_observed=observed_target,
        max_horizontal_displacement=max(math.hypot(s.position[0]-start.position[0],s.position[2]-start.position[2]) for s in samples),
        max_yaw_delta=max(abs((s.yaw-start.yaw+180)%360-180) for s in samples),
        interaction_observed=any(s.stage in ('mine','collect') for s in samples),
        current_target_tracked=last.target is not None,
        target_acquired_on_last_transition=fresh_target,
        final_raw_stage=last.stage,committed_ticks=len(samples),boundary_reason=reason,
        actual_log_gain=last.inventory_count-baseline_count,
        scope='Privileged observed window; path feasibility and mining damage are not measured')
    if last.inventory_count > baseline_count:
        result.update(classification='goal_completed',goal_completed=True)
    elif reason == 'environment_done':
        result['classification'] = 'environment_terminated'
    elif reason == 'control_budget':
        result['classification'] = 'control_budget_exhausted'
    elif reason == 'wood_target_acquired':
        if not fresh_target:
            raise ValueError('target handoff requires fresh acquisition, not a persisted server lock')
        result['classification'] = 'target_acquired'
    elif reason == 'wood_interaction_ready':
        if last.stage not in ('mine','collect'):
            raise ValueError('interaction handoff lacks current target evidence')
        result['classification'] = 'interaction_ready'
    elif reason in ('tick_budget','planning_time_budget'):
        if reason == 'tick_budget' and len(samples) != budget_ticks:
            raise ValueError('skill budget was not exhausted')
        result.update(classification=('recovery_incomplete_with_progress' if approach else
            'recovery_incomplete_with_target_evidence' if observed_target else 'recovery_budget_exhausted'),
                      replan_candidate=True)
    elif reason == 'goal_completed':
        raise ValueError('goal completion lacks authoritative inventory gain')
    return result
