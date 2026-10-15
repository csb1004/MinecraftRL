"""Bound instances of one learned waypoint executor and observation-based receipts."""
from copy import deepcopy
import math
from .waypoints import arrived,goal_features

NAMES=tuple('navigate_waypoint_'+str(i) for i in range(4))
COMPLETION='waypoint_arrived'


def capability(index,policy_ref,actor_sha256,embedding_sha256,target):
    from .replanning import collection_capability
    collection_capability('wood',policy_ref)
    if type(index) is not int or not 0<=index<4:raise ValueError('invalid waypoint instance')
    for sha in (actor_sha256,embedding_sha256):
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('invalid waypoint weight binding')
    if (not isinstance(target,list) or len(target)!=3 or
            any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>30000000 for v in target)):
        raise ValueError('invalid waypoint target')
    return dict(skill_id=NAMES[index],goal='wood',policy_ref=policy_ref,
        arguments=dict(actor_sha256=actor_sha256,embedding_sha256=embedding_sha256,target=deepcopy(target)),
        max_budget_s=30.,max_budget_ticks=100,completion=COMPLETION,
        scope='Experimental learned coordinate following; unknown traversability; arrival is not wood access or root success')


def validate_capability(value):
    if value.get('skill_id') not in NAMES:raise ValueError('unknown waypoint instance')
    expected=capability(NAMES.index(value['skill_id']),value['policy_ref'],**value['arguments'])
    if value!=expected:raise ValueError('changed waypoint contract')
    return expected


def position(g):return [float(g['server_'+a]) for a in 'xyz']


def distance(p,q):return math.hypot(p[0]-q[0],p[2]-q[2])


def scalar(action,key):
    value=action.get(key,0)
    return float(value[0] if isinstance(value,list) else value)


def assess_navigation(cap,rows,*,reason):
    validate_capability(cap)
    if reason not in ('predicate_satisfied','budget_exhausted','stop','control_budget','environment_done','subgoal_handoff','observed_blockage'):
        raise ValueError('unknown navigation termination')
    result=dict(outcome='unrun',diagnosis='not_executed',committed_steps=len(rows),reason=reason,
        target=deepcopy(cap['arguments']['target']),terrain_proven=False,partial_progress=False,
        retry_blocking=False,root_completed=False,evidence={},entry=None,end=None)
    if not rows:return result
    target=result['target'];before=rows[0]['server_before'];after=rows[-1]['server_after']
    # Observed interruption has precedence over an accidental arrival on the last step.
    interrupted=reason in ('stop','control_budget','environment_done','subgoal_handoff')
    try:
        from .stall_replay import log_count
        previous=before;best=distance(position(before),target);moved=0.;directed=misaligned=streak=longest=0
        for row in rows:
            a,b=row['server_before'],row['server_after']
            if a!=previous or b['server_tick']!=a['server_tick']+1:raise ValueError('noncontiguous navigation evidence')
            goal_features(a,target);goal_features(b,target)
            if a.get('screen')!=0 or b.get('screen')!=0:raise ValueError('unobserved navigation through GUI')
            p,q=position(a),position(b);step=distance(p,q);moved+=step;best=min(best,distance(q,target))
            action=row['action'];yaw=math.radians(a['yaw'])
            f=scalar(action,'forward')-scalar(action,'back');left=scalar(action,'left')-scalar(action,'right')
            dx,dz=-math.sin(yaw)*f+math.cos(yaw)*left,math.cos(yaw)*f+math.sin(yaw)*left
            length=math.hypot(dx,dz);delta=distance(p,target)
            aligned=length>0 and delta>0 and (dx*(target[0]-p[0])+dz*(target[2]-p[2]))/(length*delta)>=math.cos(math.pi/4)
            directed+=bool(length);misaligned+=bool(length and not aligned)
            streak=streak+1 if aligned and step<=.025 and abs(p[1]-q[1])<=.25 else 0
            longest=max(longest,streak);previous=b
        start=distance(position(before),target);end=distance(position(after),target)
        windowed=None
        if reason=='observed_blockage':
            from .motion_blockage import target_blockage
            windowed=target_blockage(rows,target)
            if windowed is None:raise ValueError('missing fresh target-aligned blockage evidence')
        result.update(entry=dict(position=position(before),yaw=before['yaw'],reachable=before['wood_v3_reachable']),
            end=dict(position=position(after),yaw=after['yaw'],reachable=after['wood_v3_reachable']),
            root_completed=log_count(after)>log_count(before),
            partial_progress=best<=start-.25,
            evidence=dict(start_distance=start,end_distance=end,minimum_distance=best,path_blocks=moved,
                movement_commands=directed,misaligned_commands=misaligned,aligned_low_motion_streak=longest))
        if interrupted:result.update(outcome='interrupted',diagnosis=reason)
        elif arrived(after,target):result.update(outcome='completed',diagnosis='observed_arrival')
        else:
            diagnosis=('blocked_direction_suspected' if longest>=12 or windowed is not None else
                'arrival_precision' if best<=1. else 'heading_misaligned' if directed>=8 and misaligned/directed>=.6 else
                'insufficient_directed_motion' if directed<8 else 'unresolved_navigation_failure')
            result.update(outcome='failed',diagnosis=diagnosis,retry_blocking=True)
        if windowed is not None:result['evidence']['windowed_blockage']=windowed
    except (KeyError,TypeError,ValueError,OverflowError):
        result.update(outcome='interrupted' if interrupted else 'unverified',diagnosis='observation_unreliable')
    return result


def same_failed_route(cap,observation,receipt,*,world_id,episode_id):
    """Local method equivalence, not a permanent claim that a path is impassable."""
    if (receipt.get('method',{}).get('skill_id') not in NAMES or
            receipt.get('before',{}).get('world_id')!=world_id or receipt['before'].get('episode_id')!=episode_id):return False
    old=receipt['method'];r=receipt.get('assessment',{})
    if not r.get('retry_blocking') or not r.get('entry'):return False
    if (old['policy_ref']!=cap['policy_ref'] or any(old['arguments'][k]!=cap['arguments'][k]
            for k in ('actor_sha256','embedding_sha256'))):return False
    p=position(observation);origin=r['entry']['position']
    if distance(p,origin)>2.5 or abs(p[1]-origin[1])>1.5:return False
    if observation.get('wood_v3_reachable')!=r['end']['reachable']:return False
    if r['diagnosis'] in ('heading_misaligned','insufficient_directed_motion'):
        rotation=abs((observation['yaw']-r['end']['yaw']+180)%360-180)
        if rotation>=45:return False
    target=cap['arguments']['target'];prior=old['arguments']['target']
    a=[target[0]-p[0],target[2]-p[2]];b=[prior[0]-origin[0],prior[2]-origin[2]]
    n=math.hypot(*a)*math.hypot(*b)
    return n>0 and sum(x*y for x,y in zip(a,b))/n>=math.cos(math.radians(40))
