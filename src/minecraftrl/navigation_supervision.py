"""Separate expert targets for real training states; never change behavior actions."""
import math
from collections import Counter
from .navigation_refinement import local_targets, validate_dataset
from .fixture_probe import navigate
from .runtime import jsonable
from .navigation_observation import position


def expert_requests(data, observed):
    """Bind actual tensor rows before choosing the explicitly declared teacher."""
    if data.get('world',{}).get('kind')=='natural_navigation':
        raise ValueError('flat expert cannot label natural terrain')
    validate_dataset(data, 'bc')
    if data['world'].get('experimental_split') != 'train':
        raise ValueError('expert supervision requires training worlds')
    targets = local_targets(data['world'], data['config']['points'])
    if len(observed) != len(data['rows']):
        raise ValueError('observation/trajectory count mismatch')
    ox, oz = (c * 16 for c in data['world']['chunk'])
    pose_source=data['config'].get('pose_source','server')
    step=data['world'].get('kind')=='step_navigation'
    requests, excluded = [], Counter()
    for index, (row, actual) in enumerate(zip(data['rows'], observed)):
        if (row['owner'] != actual['owner'] or row['index'] != actual['index'] or
                row['motion_goal'] != actual['goal'] or row['env_action'] != actual['action'] or
                row.get('interrupted') or actual.get('interrupted')):
            raise ValueError('expert query is not bound to actual behavior')
        if actual['target'] != targets[row['index']]:
            raise ValueError('expert target differs from frozen schedule')
        if row['owner'] == 'neutral':
            excluded['neutral_handoff'] += 1
            continue
        if row['owner'] not in ('student', 'teacher'):
            raise ValueError('unknown behavior owner')
        g = actual['server_before']
        if step:
            from .step_navigation import step_teacher
            changes,_=step_teacher(g,actual['target'],data['world'],pose_source=pose_source)
            requests.append(dict(row=index,owner=row['owner'],changes=changes))
            continue
        current=position(g,pose_source)
        keys = ('server_x', 'server_y', 'server_z', 'yaw', 'pitch')
        if not all(math.isfinite(g[k]) for k in keys):
            raise ValueError('nonfinite expert observation')
        # No assumption that a straight-line teacher can handle walls, ledges or GUI.
        if (g.get('server_available') != 1 or g.get('error') != 0 or g.get('screen') != 0 or
                abs(current[1] - 201.) > .05 or not 1.0 < current[0] - ox < 6.7 or
                not .5 < current[2] - oz < 15.5):
            excluded['outside_expert_domain'] += 1
            continue
        changes, _ = navigate(g, actual['target'][0], actual['target'][2],pose_source=pose_source)
        requests.append(dict(row=index, owner=row['owner'], changes=changes))
    return requests, dict(excluded)


def label_training_states(data, observed, mapper, transformer):
    """Official quantized labels, with teacher parity and independent decode checks."""
    import numpy as np
    import torch
    from steve1.VPT.lib.actions import Buttons
    requests, excluded = expert_requests(data, observed)
    selected, labels, owners = [], [], Counter()
    for request in requests:
        index = request['row']
        original = data['rows'][index]
        # Match the scalar no-op used by the actual MineRL teacher, including all buttons.
        desired = {button: 0 for button in Buttons.ALL}
        desired['camera'] = np.zeros(2)
        desired.update(request['changes'])
        factored = transformer.env2policy({k: np.asarray(v) for k, v in desired.items()})
        mapped = mapper.from_factored({k: v[None] for k, v in factored.items()})
        decoded = transformer.policy2env(mapper.to_factored(mapped))
        roundtrip = mapper.from_factored(transformer.env2policy(decoded))
        if any(not np.array_equal(mapped[k], roundtrip[k]) for k in mapped):
            raise ValueError('expert action failed official mapping roundtrip')
        action = {k: torch.as_tensor(v).unsqueeze(1) for k, v in mapped.items()}
        if original['owner'] == 'teacher' and any(
                not torch.equal(action[k], original['action'][k]) for k in action):
            raise ValueError('queried expert differs from recorded teacher action')
        # Shallow row copy; never overwrite actual action, logp, owner or features.
        selected.append(dict(original, bc_action=action))
        if data['world'].get('kind')=='step_navigation':
            selected[-1].update(bc_source='procedural_step',bc_step_jump=bool(request['changes'].get('jump',0)))
        labels.append(dict(row=index, behavior_owner=original['owner'],
            policy_action=jsonable(action), quantized_env_action=jsonable(decoded)))
        owners[original['owner']] += 1
    if not selected:
        raise ValueError('no eligible expert labels')
    return selected, dict(schema='navigation_expert_queries_v1', labels=labels,
        owners=dict(owners), excluded=excluded, actual_actions_modified=False,
        expert='step_navigation.step_teacher' if data['world'].get('kind')=='step_navigation' else 'fixture_probe.navigate',
        source='privileged_'+data['config'].get('pose_source','server')+'_before',
        teacher_parity_verified=True, official_mapping_roundtrip_verified=True)
