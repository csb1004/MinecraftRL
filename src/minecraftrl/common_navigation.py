"""Reproducible goal variation for one reusable, privileged local-movement skill."""
import math
import random
import json

POLICY_FILES=('navigation_trial.py','navigation_model.py','navigation_refinement.py','waypoints.py',
              'rl_policy.py','rl_collect.py','runtime.py','policies.py','rl_core.py')


def initial_navigation_conditions(run):
    from .world_store import digest
    read=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))
    initial=read(run/'initial-state.json')['server']
    condition=read(run/'rl-condition.json')
    jar=read(run/'runtime-jar-cache.json')
    protocol=read(run.parent/'protocol.json')
    policy_files=POLICY_FILES+('navigation_arrival.py',) if protocol.get('arrival_hold_ticks',1)>1 else POLICY_FILES
    if protocol.get('pose_source','server')=='client':
        policy_files+=('navigation_observation.py','workshop_monitor.py')
    if protocol.get('action_selection','sample')!='sample':policy_files+=('navigation_action_likelihood.py',)
    if protocol.get('goal_contract')=='natural_navigation_goals_v1':policy_files+=('natural_navigation.py',)
    if protocol.get('goal_contract')=='step_navigation_goals_v1':policy_files+=('step_navigation.py','fixture_probe.py')
    if protocol.get('startup_control'):policy_files+=('startup_control.py',)
    result=dict(memory=condition['initial_recurrent_sha256'],
        action_selection=protocol.get('action_selection','sample'),
        conditioning={k:condition[k] for k in ('embedding_sha256','prompt','cond_scale',
            'backbone_tail_sha256','backbone_temporal_sha256','recurrent_commit_mode','reward_config')},
        runtime_jar={k:jar[k] for k in ('sha256','launcher_sha256')},
        pose={k:initial[k] for k in ('server_x','server_y','server_z','yaw','pitch')},
        client_pose={k:initial[k] for k in ('client_x','client_y','client_z','client_tick')} if protocol.get('pose_source','server')=='client' else None,
        pixels=digest(run/'frames/000000.png'),
        policy_sources={f:digest(run.parent/'source/src/minecraftrl'/f) for f in policy_files})
    if protocol.get('goal_contract')=='natural_navigation_goals_v1':
        result['natural_world_state']={k:initial[k] for k in ('server_health','server_food','server_alive',
            'world_time','server_tick','server_on_ground','held_key_count')}
        result['natural_inventory']={k:v for k,v in initial['server_inventory'].items() if v}
    if protocol.get('startup_control'):
        result['startup_control']=protocol['startup_control']
    return result


def validate_navigation_reference(protocol,world,sampling_seed,actor_sha256):
    expected=dict(mode='evaluate',split='development',sampling_seed=sampling_seed,
        actor_sha256=actor_sha256,world_sha256=world['sha256'],points=world['points'])
    if protocol.get('arrival_hold_ticks',1)!=world.get('arrival_hold_ticks',1):
        raise ValueError('reference run differs in arrival contract')
    if protocol.get('pose_source','server')!=world.get('pose_source','server'):
        raise ValueError('reference run differs in control pose source')
    if protocol.get('action_selection','sample')!=world.get('action_selection','sample'):
        raise ValueError('reference run differs in action selection')
    if world['split']!='development' or any(protocol.get(k)!=v for k,v in expected.items()):
        raise ValueError('reference run differs in policy/world/goals/sampling seed/split')


def navigation_training_arguments(plan):
    method=plan.get('training_method','bc_continue')
    if method=='ppo':
        if plan.get('collection_mode')!='ppo' or not any(c['split']=='train' for c in plan['cases']):
            raise ValueError('PPO requires new on-policy training collection')
        if plan.get('replay') or plan.get('label_monitor') or plan.get('query_expert') or plan.get('goal_variations'):
            raise ValueError('PPO cannot reuse demonstrations or query expert labels')
        if any(k in plan for k in ('learning_rate','epochs','kl_limit','distance_scale','bc_loss')):
            raise ValueError('PPO retains its declared optimizer and trust-region defaults')
        return method,[]
    if method!='bc_continue':raise ValueError('unsupported common navigation training method')
    if not plan.get('label_monitor'):raise ValueError('BC requires a separate label monitor')
    if 'distance_scale' in plan and (type(plan['distance_scale']) is not int or plan['distance_scale'] not in (1,16)):
        raise ValueError('distance conditioning experiment permits only scale 1 or 16')
    if plan.get('bc_loss','canonical') not in ('canonical','delivered'):
        raise ValueError('unknown supervised navigation objective')
    if plan.get('goal_variations') and not plan.get('query_expert'):
        raise ValueError('goal variations require explicit expert supervision')
    options=['--learning-rate',str(plan['learning_rate'])]
    if plan.get('query_expert'):options.append('--query-expert')
    if plan.get('goal_variations'):options.append('--goal-variations')
    for key in ('epochs','kl_limit','distance_scale','bc_loss'):
        if key in plan:options.extend(['--'+key.replace('_','-'),str(plan[key])])
    return method,options


def navigation_trial_arguments(plan,mode):
    if mode not in ('correction','evaluate','ppo'):raise ValueError('unknown navigation trial mode')
    hold=plan.get('arrival_hold_ticks',1);prefix=plan.get('student_ticks')
    if type(hold) is not int or not 1<=hold<=20:raise ValueError('arrival hold must be 1..20 ticks')
    if prefix is not None and (type(prefix) is not int or not 1<=prefix<=60):
        raise ValueError('correction prefix must be 1..60 ticks')
    options=['--arrival-hold-ticks',str(hold)] if hold>1 else []
    source=plan.get('pose_source','server')
    if source not in ('server','client'):raise ValueError('unknown control pose source')
    if source=='client':options+=['--pose-source','client']
    selection=plan.get('action_selection','sample')
    if selection not in ('sample','delivered_mode') or (selection!='sample' and mode!='evaluate'):
        raise ValueError('delivered mode is evaluation only')
    if selection!='sample':options+=['--action-selection',selection]
    if mode=='correction' and prefix is not None:options+=['--student-ticks',str(prefix)]
    return options


def validate_collection_quality(report, minimum_completed):
    """Execution validity alone cannot admit failed demonstrations to training."""
    if type(minimum_completed) is not int or not 1<=minimum_completed<=6:
        raise ValueError('collection completion gate must be 1..6 goals')
    if (not report.get('execution_valid') or not report.get('navigation_finished') or
            report.get('learned_arrivals',0)+report.get('teacher_arrivals',0)<minimum_completed):
        raise ValueError('collection did not meet the declared completed-goal quality gate')


def project_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('path outside project')
    return path


def validate_prepared_worlds(selection, manifest):
    expected = {case['label']: case for case in selection['cases']}
    worlds = manifest['worlds']
    if not manifest.get('completed') or set(worlds) != set(expected):
        raise ValueError('incomplete or changed curriculum cases')
    for name, world in worlds.items():
        if any(world[key] != expected[name][key] for key in ('seed', 'split', 'points')):
            raise ValueError('prepared goals/split differ from predeclared selection: ' + name)


def curriculum_cases(plan, sources):
    if plan.get('schema') != 'common_navigation_curriculum_v1':
        raise ValueError('unknown common skill curriculum')
    cases = plan['cases']
    if not cases or len({c['label'] for c in cases}) != len(cases):
        raise ValueError('unique nonempty case labels required')
    allowed = {'train': set(sources['train_seeds']), 'development': set(sources['validation_seeds'])}
    if not allowed['train'].isdisjoint(allowed['development']):
        raise ValueError('train/development seed leakage')
    reserved = set(sources['reserved_final_seeds'])
    rng = random.Random(plan['generation_seed'])
    result = []
    for case in cases:
        if (case['split'] not in allowed or case['seed'] not in allowed[case['split']] or
                case['seed'] in reserved):
            raise ValueError('source seed violates declared split')
        if not case['label'].replace('-', '').isalnum():
            raise ValueError('simple case label required')
        start = [round(rng.uniform(2., 5.), 3), round(rng.uniform(2., 11.), 3)]
        previous, points = start, []
        for index in range(6):
            low, high = (.65, 1.3) if index % 2 else (3., 6.)
            for _ in range(10000):
                point = [round(rng.uniform(1.7, 5.3), 3), round(rng.uniform(1.7, 12.3), 3)]
                if low <= math.dist(previous, point) <= high:
                    break
            else:
                raise ValueError('cannot sample local goal')
            points.append(point)
            previous = point
        result.append(dict(case, start_xz=start, points=points))
    return result
