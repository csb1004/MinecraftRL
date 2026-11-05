"""Bounded natural-terrain evaluation and attempted-goal accounting."""
import math


def _point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError('finite absolute XYZ point required')
    return list(value)


def natural_targets(world, points):
    if world.get('kind') != 'natural_navigation' or world.get('experimental_split') not in ('train','development'):
        raise ValueError('separate natural navigation train/development world required')
    contract = world.get('navigation_goals', {})
    if contract.get('schema') != 'natural_navigation_goals_v1' or contract.get('coordinates') != 'absolute_xyz':
        raise ValueError('explicit natural XYZ goal contract required')
    if (contract.get('source_archive_sha256') != world.get('sha256') or
            not isinstance(world.get('sha256'), str) or len(world['sha256']) != 64):
        raise ValueError('natural goals are not bound to the saved world')
    for key, value in dict(privileged_targets=True,terrain_modified=False,inventory_modified=False,
                          route_given_to_policy=False,geometry_only=True,learned_route_verified=False).items():
        if contract.get(key) is not value:
            raise ValueError('natural evaluation semantics changed: '+key)
    for key, value in dict(goal_count=6,horizontal_radius=8,vertical_radius=8,actions_per_goal=100).items():
        if type(contract.get(key)) is not int or contract[key] != value:
            raise ValueError('natural evaluation bounds changed: '+key)
    if not isinstance(points, list) or len(points) != 6 or points != contract.get('points'):
        raise ValueError('natural goal schedule differs from frozen world contract')
    initial = _point(contract.get('initial_position'))
    if initial != _point(world.get('position')):
        raise ValueError('natural saved start differs from goal contract')
    result = [_point(point) for point in points]
    previous = initial
    for point in result:
        if (max(abs(point[i]-initial[i]) for i in (0,2)) > 8 or
                abs(point[1]-initial[1]) > 8 or not 1 <= point[1] <= 253):
            raise ValueError('natural goal outside inspected region')
        if not 1.5 <= math.hypot(point[0]-previous[0],point[2]-previous[2]) <= 4 or abs(point[1]-previous[1]) > 1:
            raise ValueError('natural goal exceeds declared local distance or height change')
        previous = point
    return result


def validate_natural_trial(world, config):
    """No flat teacher or PPO may silently enter the first natural evaluation."""
    targets = natural_targets(world, config.get('points'))
    if (config.get('schema') != 'navigation_trial_config_v2' or
            config.get('goal_contract') != 'natural_navigation_goals_v1' or
            config.get('mode') != 'evaluate' or config.get('action_selection') != 'delivered_mode' or
            config.get('pose_source') != 'client' or type(config.get('arrival_hold_ticks')) is not int or
            config['arrival_hold_ticks'] != 12 or config.get('student_ticks',32) != 32):
        raise ValueError('natural trial requires learned-only settled client-pose evaluation')
    if config.get('split') != world['experimental_split'] or config.get('world_sha256') != world['sha256']:
        raise ValueError('natural trial split/world binding mismatch')
    return targets


def attempted_goal_outcomes(summary, rows, goal_count):
    """Keep early termination distinct from goals the policy never attempted."""
    if type(goal_count) is not int or not 1 <= goal_count <= 6:
        raise ValueError('bounded natural goal count required')
    attempted = sorted({row['index'] for row in rows})
    if attempted != list(range(len(attempted))) or len(attempted) > goal_count:
        raise ValueError('natural goal attempts must be an ordered schedule prefix')
    if summary['navigation_finished'] and len(attempted) != goal_count:
        raise ValueError('finished natural schedule must have attempted every goal')
    receipts = summary['navigation_receipts']
    if any(r['index'] not in attempted for r in receipts):
        raise ValueError('completion receipt without an attempted goal')
    completed = {r['index']: r for r in receipts}
    if len(completed) != len(receipts):
        raise ValueError('duplicate natural goal receipt')
    outcomes = []
    for index in range(goal_count):
        if index not in attempted:
            outcome = 'not_attempted'
        elif index in completed:
            receipt = completed[index]
            if receipt['outcome'] not in ('completed','failed'):
                raise ValueError('unsupported natural goal receipt')
            if (type(receipt['student_success']) is not bool or
                    receipt['student_success'] != (receipt['outcome'] == 'completed')):
                raise ValueError('natural outcome and learned arrival evidence disagree')
            outcome = 'arrived' if receipt['student_success'] else 'budget_exhausted'
        else:
            if index != attempted[-1] or summary['navigation_finished']:
                raise ValueError('missing terminal natural goal evidence')
            outcome = 'interrupted'
        outcomes.append(dict(index=index,outcome=outcome,
            actions=sum(row['index']==index for row in rows)))
    return dict(goals=outcomes,attempted=len(attempted),not_attempted=goal_count-len(attempted),
        arrived=sum(r['outcome']=='arrived' for r in outcomes),
        budget_exhausted=sum(r['outcome']=='budget_exhausted' for r in outcomes),
        interrupted=sum(r['outcome']=='interrupted' for r in outcomes),
        termination_status=summary['status'])


def validate_transfer_admission(evidence, baseline_sha256, candidate_sha256):
    """Admit exploratory transfer only after audited flat arrival retention."""
    pipeline=evidence['pipeline'];ready=evidence['mode_readiness']
    calibration=evidence['calibration'];summary=evidence['calibration_summary']
    comparison=evidence['flat_comparison'];selection=evidence['flat_selection']['plan']
    if pipeline.get('completed') is not True or pipeline.get('active') is not None or pipeline.get('error') is not None:
        raise ValueError('source pipeline is incomplete or failed')
    if (ready.get('completed') is not True or ready.get('train_calibration_ready') is not True or
            ready.get('counts')!={'near':4,'far':4} or ready.get('actor_sha256')!=candidate_sha256):
        raise ValueError('source mode readiness is unbound or failed')
    if (calibration.get('completed') is not True or calibration.get('passed') is not True or
            summary.get('actor_sha256')!=candidate_sha256 or summary.get('execution_valid') is not True or
            summary.get('worker_exit_code')!=0 or summary.get('navigation_finished') is not True or
            calibration.get('arrivals')!=summary.get('learned_arrivals') or not 5<=summary.get('learned_arrivals',0)<=6):
        raise ValueError('source actual calibration is unbound or failed')
    if (comparison.get('completed') is not True or
            selection.get('baseline_sha256')!=baseline_sha256 or selection.get('candidate_sha256')!=candidate_sha256):
        raise ValueError('source flat comparison is unbound or incomplete')
    pairs=comparison.get('pairs',[])
    if len(pairs)!=2 or len({p['case'] for p in pairs})!=2:
        raise ValueError('two distinct flat development comparisons required')
    arrivals=0
    for pair in pairs:
        old=pair['reports']['baseline']['arrivals'];new=pair['reports']['candidate']['arrivals']
        if (pair.get('initial_conditions_equal') is not True or type(old) is not int or type(new) is not int or
                not 0<=old<=new<=6):
            raise ValueError('source flat arrival regression or unequal initial conditions')
        arrivals+=new
    if arrivals<10:raise ValueError('source flat arrival coverage insufficient')
    return dict(verified=True,flat_arrivals=arrivals,flat_improvement_gate_passed=comparison.get('gate_passed'),
                speed_improvement_required=False,scope='Exploratory natural evaluation admission only')
