"""Counterfactual goal supervision on real train states, never executed trajectories."""
import math
from collections import Counter
from .navigation_supervision import expert_requests
from .navigation_observation import position
from .waypoints import goal_features
from .fixture_probe import navigate
from .runtime import jsonable

BEARINGS=(-170.,-135.,-45.,-8.,0.,8.,45.,135.,170.)
DISTANCES=(.10,.18,.60,1.50,3.)


def flat_body_clearance(world,xyz):
    ox,oz=(16*c for c in world['chunk']);x,y,z=xyz
    if not all(math.isfinite(v) for v in xyz) or abs(y-201.)>.05:return False
    # Feet occupy a .6-wide box. Column 0/15 are perimeter walls; column x7 is the inner wall.
    occupied=lambda value: (math.floor(value-.3+1e-7),math.floor(value+.3-1e-7))
    left,right=occupied(x-ox);near,far=occupied(z-oz)
    return 1<=left<=right<=6 and 1<=near<=far<=14


def supervision_goal(row):
    goal=row.get('bc_motion_goal',row['motion_goal'])
    if len(goal)!=10 or not all(math.isfinite(v) for v in goal):raise ValueError('invalid supervised motion goal')
    if 'bc_motion_goal' in row and 'bc_action' not in row:raise ValueError('counterfactual goal requires a separate action label')
    return goal


def goal_variation_requests(data,observed,coverage='sparse'):
    if data.get('world',{}).get('kind')=='step_navigation':raise ValueError('flat counterfactual goals cannot label step terrain')
    if coverage=='boundary':return approach_goal_variation_requests(data,observed)
    if coverage=='dense':return dense_goal_variation_requests(data,observed)
    if coverage!='sparse':raise ValueError('unknown goal coverage')
    actual,excluded=expert_requests(data,observed)
    source=data['config'].get('pose_source','server')
    grounded=[r for r in actual if observed[r['row']]['server_before'].get(source+'_on_ground')==1]
    eligible=[r for r in grounded if flat_body_clearance(data['world'],position(observed[r['row']]['server_before'],source))]
    counts=Counter(excluded);counts['unverified_ground']=len(actual)-len(grounded)
    counts['source_body_outside_open_area']=len(grounded)-len(eligible)
    count=min(12,len(eligible))
    selected=[eligible[i*(len(eligible)-1)//(count-1)]['row'] for i in range(count)] if count>1 else [r['row'] for r in eligible]
    requests=[]
    for index in selected:
        row=data['rows'][index];record=observed[index];g=record['server_before'];xyz=position(g,source)
        original_goal=supervision_goal(row)
        expected=goal_features(g,record['target'],pose_source=source)
        if any(abs(a-b)>1e-7 for a,b in zip(original_goal[:6],expected)):
            raise ValueError('original goal does not match current observation geometry')
        for bearing in BEARINGS:
            angle=math.radians(g['yaw']+bearing);targets=[]
            for distance in DISTANCES:
                target=[xyz[0]-distance*math.sin(angle),201.,xyz[2]+distance*math.cos(angle)]
                if flat_body_clearance(data['world'],target):targets.append((distance,target))
                else:counts['target_outside_known_flat_domain']+=1
            if not any(distance>=.6 for distance,_ in targets):
                counts['ray_without_paired_far_goal']+=1
                continue
            for distance,target in targets:
                goal=goal_features(g,target,pose_source=source)+list(original_goal[6:9])+[1.]
                changes,_=navigate(g,target[0],target[2],pose_source=source)
                requests.append(dict(row=index,target=target,bearing=bearing,distance=distance,
                    motion_goal=goal,changes=changes,behavior_owner=row['owner']))
    return requests,dict(selected_rows=selected,excluded=dict(counts),pose_source=source,
        domain='bedrock_fixture_body_clearance_v1',counterfactual_budget_ticks=100)


def label_goal_variations(data,observed,mapper,transformer,coverage='sparse'):
    import numpy as np
    import torch
    from steve1.VPT.lib.actions import Buttons
    requests,selection=goal_variation_requests(data,observed,coverage)
    labeled=[];labels=[]
    for request in requests:
        original=data['rows'][request['row']]
        desired={button:0 for button in Buttons.ALL};desired['camera']=np.zeros(2)
        desired.update(request['changes'])
        factored=transformer.env2policy({k:np.asarray(v) for k,v in desired.items()})
        mapped=mapper.from_factored({k:v[None] for k,v in factored.items()})
        decoded=transformer.policy2env(mapper.to_factored(mapped))
        roundtrip=mapper.from_factored(transformer.env2policy(decoded))
        if any(not np.array_equal(mapped[k],roundtrip[k]) for k in mapped):
            raise ValueError('goal query failed official action mapping roundtrip')
        action={k:torch.as_tensor(v).unsqueeze(1) for k,v in mapped.items()}
        labeled.append(dict(original,bc_action=action,bc_motion_goal=list(request['motion_goal'])))
        labels.append(dict({k:v for k,v in request.items() if k!='changes'},
            policy_action=jsonable(action),quantized_env_action=jsonable(decoded)))
    if not labeled:raise ValueError('no eligible paired goal queries')
    schema={'sparse':'navigation_goal_queries_v1','dense':'navigation_goal_queries_v2','boundary':'navigation_goal_queries_v3'}[coverage]
    return labeled,dict(schema=schema,selection=selection,labels=labels,
        source='privileged_'+selection['pose_source']+'_before',actual_actions_modified=False,
        executed_counterfactual_goals=0,official_mapping_roundtrip_verified=True,gradient_split='train')


def dense_goal_variation_requests(data,observed):
    if data.get('world',{}).get('kind')=='step_navigation':raise ValueError('flat counterfactual goals cannot label step terrain')
    """Every eligible logged state gets a paired near/far ray and another far bearing."""
    import hashlib
    actual,excluded=expert_requests(data,observed);source=data['config'].get('pose_source','server')
    counts=Counter(excluded);requests=[];selected=[]
    for record in actual:
        index=record['row'];row=data['rows'][index];observation=observed[index];g=observation['server_before']
        if g.get(source+'_on_ground')!=1:counts['unverified_ground']+=1;continue
        xyz=position(g,source)
        if not flat_body_clearance(data['world'],xyz):counts['source_body_outside_open_area']+=1;continue
        original=supervision_goal(row);expected=goal_features(g,observation['target'],pose_source=source)
        if any(abs(a-b)>1e-7 for a,b in zip(original[:6],expected)):
            raise ValueError('original goal does not match current observation geometry')
        rays={}
        for bearing in BEARINGS:
            angle=math.radians(g['yaw']+bearing)
            points={distance:[xyz[0]-distance*math.sin(angle),201.,xyz[2]+distance*math.cos(angle)] for distance in DISTANCES}
            clear={distance:target for distance,target in points.items() if flat_body_clearance(data['world'],target)}
            far=[distance for distance in clear if distance>=.6]
            near=[distance for distance in clear if distance<.22]
            if far and near:rays[bearing]=(clear,near,max(far))
        if not rays:counts['no_paired_ray']+=1;continue
        identity=f"dense-v1:{data['world']['sha256']}:{index}".encode()
        selector=int.from_bytes(hashlib.sha256(identity).digest()[:8],'big')
        bearings=list(rays);primary=0. if 0. in rays else bearings[selector%len(bearings)]
        clear,near,far=rays[primary];near_distance=near[(selector>>8)%len(near)]
        choices=[(primary,near_distance,clear[near_distance]),(primary,far,clear[far])]
        other=[bearing for bearing in bearings if bearing!=primary]
        if other:
            bearing=other[(selector>>16)%len(other)];clear,_,far=rays[bearing]
            choices.append((bearing,far,clear[far]))
        else:counts['only_one_paired_bearing']+=1
        selected.append(index)
        for bearing,distance,target in choices:
            goal=goal_features(g,target,pose_source=source)+list(original[6:9])+[1.]
            changes,_=navigate(g,target[0],target[2],pose_source=source)
            requests.append(dict(row=index,target=target,bearing=bearing,distance=distance,motion_goal=goal,
                changes=changes,behavior_owner=row['owner']))
    return requests,dict(selected_rows=selected,excluded=dict(counts),pose_source=source,
        domain='bedrock_fixture_body_clearance_v1',counterfactual_budget_ticks=100,
        coverage='all_eligible_logged_states_v1',pairing='one near and far on the same ray; one distinct far bearing when valid',
        primary_bearing='current heading when body-clear, otherwise deterministic valid bearing',
        far_distance='largest verified distance among .6,1.5,3 blocks',selector='sha256 world identity and logged row')


def approach_goal_variation_requests(data,observed):
    if data.get('world',{}).get('kind')=='step_navigation':raise ValueError('flat counterfactual goals cannot label step terrain')
    """Keep dense supervision and bridge its .18-to-.6-block approach gap uniformly."""
    import hashlib
    requests,selection=dense_goal_variation_requests(data,observed)
    source=selection['pose_source']
    # One added target per eligible observation, using its already verified near/far ray.
    # This changes BC supervision only; the teacher and actual arrival threshold stay fixed.
    near=[r for r in requests if r['distance']<.22]
    assert len(near)==len(selection['selected_rows'])
    extra=[]
    for anchor in near:
        index=anchor['row'];row=data['rows'][index];g=observed[index]['server_before']
        selector=int.from_bytes(hashlib.sha256(f"approach-v1:{data['world']['sha256']}:{index}".encode()).digest()[:8],'big')
        distance=(.26,.34,.46)[selector%3];bearing=anchor['bearing']
        xyz=position(g,source);angle=math.radians(g['yaw']+bearing)
        target=[xyz[0]-distance*math.sin(angle),201.,xyz[2]+distance*math.cos(angle)]
        if not flat_body_clearance(data['world'],target):raise ValueError('intermediate goal lacks body clearance')
        goal=goal_features(g,target,pose_source=source)+list(supervision_goal(row)[6:9])+[1.]
        changes,_=navigate(g,target[0],target[2],pose_source=source)
        extra.append(dict(row=index,target=target,bearing=bearing,distance=distance,motion_goal=goal,
            changes=changes,behavior_owner=row['owner']))
    selection=dict(selection,coverage='all_eligible_logged_states_boundary_v1',
        intermediate_distances=[.26,.34,.46],intermediate_goal_rule='one deterministic intermediate point on each existing paired ray',
        teacher_stop_distance_unchanged=True)
    return requests+extra,selection
