"""Counterfactual labels restricted to declared goals in static procedural step arenas."""
from .navigation_supervision import expert_requests
from .navigation_goal_queries import supervision_goal
from .step_navigation import step_targets,step_teacher,terrain_action_count
from .waypoints import goal_features
from .runtime import jsonable


def step_goal_requests(data,observed):
    if data.get('world',{}).get('kind')!='step_navigation':
        raise ValueError('declared step goal queries require procedural step terrain')
    actual,excluded=expert_requests(data,observed)
    if terrain_action_count(observed):raise ValueError('step goal queries require unchanged terrain')
    targets=step_targets(data['world'],data['config']['points']);requests=[]
    for entry in actual:
        index=entry['row'];row=data['rows'][index];record=observed[index];g=record['server_before']
        original=supervision_goal(row);expected=goal_features(g,record['target'],pose_source='client')
        if any(abs(a-b)>1e-7 for a,b in zip(original[:6],expected)):
            raise ValueError('step original goal differs from observed geometry')
        for target_index,target in enumerate(targets):
            if target_index==row['index']:continue
            changes,_=step_teacher(g,target,data['world'],pose_source='client')
            requests.append(dict(row=index,target_index=target_index,target=list(target),
                motion_goal=goal_features(g,target,pose_source='client')+list(original[6:9])+[1.],
                changes=changes,behavior_owner=row['owner']))
    if not requests:raise ValueError('no eligible step goal queries')
    assert len(requests)==5*len(actual)
    return requests,dict(selected_rows=[r['row'] for r in actual],excluded=excluded,pose_source='client',
        domain='declared_static_step_goals_v1',counterfactual_budget_ticks=100,
        goals_per_state=5,original_goal_excluded=True,world_sha256=data['world']['sha256'])


def label_step_goal_variations(data,observed,mapper,transformer):
    import numpy as np
    import torch
    from steve1.VPT.lib.actions import Buttons
    requests,selection=step_goal_requests(data,observed);labeled=[];labels=[]
    for request in requests:
        original=data['rows'][request['row']]
        desired={button:0 for button in Buttons.ALL};desired['camera']=np.zeros(2)
        desired.update(request['changes'])
        factored=transformer.env2policy({k:np.asarray(v) for k,v in desired.items()})
        mapped=mapper.from_factored({k:v[None] for k,v in factored.items()})
        decoded=transformer.policy2env(mapper.to_factored(mapped))
        roundtrip=mapper.from_factored(transformer.env2policy(decoded))
        if any(not np.array_equal(mapped[k],roundtrip[k]) for k in mapped):
            raise ValueError('step goal query failed official action mapping roundtrip')
        action={k:torch.as_tensor(v).unsqueeze(1) for k,v in mapped.items()}
        labeled.append(dict(original,bc_action=action,bc_motion_goal=list(request['motion_goal']),
            bc_source='procedural_step',bc_step_jump=bool(request['changes'].get('jump',0)),bc_step_goal_query=True))
        labels.append(dict({k:v for k,v in request.items() if k!='changes'},
            policy_action=jsonable(action),quantized_env_action=jsonable(decoded)))
    return labeled,dict(schema='step_navigation_goal_queries_v1',selection=selection,labels=labels,
        source='privileged_client_before',actual_actions_modified=False,executed_counterfactual_goals=0,
        official_mapping_roundtrip_verified=True,gradient_split='train',world_sha256=data['world']['sha256'])
