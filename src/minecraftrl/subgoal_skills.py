"""Observation-defined neural options; no action synthesis or new root rewards."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


SKILLS={
    'leave_stalled_area':dict(prompt='go explore',completion='displacement_3_blocks',
        scope='Horizontal displacement only; does not prove a passage or target access'),
    'locate_wood':dict(prompt='chop down the tree, gather wood, pick up wood, chop it down, break tree',
        completion='wood_target_known',scope='Server-tracked target location; not necessarily visible or reachable'),
    'approach_wood':dict(prompt='walk toward the tree',completion='wood_interaction_ready',
        scope='Experimental text-conditioned approach; requires an observed wood target; no guaranteed detour'),
}


def capability(skill_id,policy_ref,embedding_sha256,navigation_actor_sha256=None):
    from .replanning import collection_capability
    collection_capability('wood',policy_ref)
    if skill_id not in SKILLS or not isinstance(embedding_sha256,str) or len(embedding_sha256)!=64 or any(
            c not in '0123456789abcdef' for c in embedding_sha256):
        raise ValueError('invalid neural subgoal binding')
    spec=SKILLS[skill_id]
    result=dict(skill_id=skill_id,goal='wood',policy_ref=policy_ref,
        arguments=dict(embedding_sha256=embedding_sha256),max_budget_s=30.,max_budget_ticks=200,
        completion=spec['completion'],scope=spec['scope'])
    if navigation_actor_sha256 is not None:
        if (skill_id not in ('locate_wood','approach_wood') or len(navigation_actor_sha256)!=64
                or any(c not in '0123456789abcdef' for c in navigation_actor_sha256)):
            raise ValueError('invalid local navigation actor')
        result['arguments']['navigation_actor_sha256']=navigation_actor_sha256
        result['scope']=('Local passability route with learned navigation actor; '+
            ('new wood ray observation required' if skill_id=='locate_wood' else 'current server interaction readiness required'))
    return result


def validate_capability(value):
    expected=capability(value['skill_id'],value['policy_ref'],**value['arguments'])
    if value!=expected:raise ValueError('changed neural subgoal contract')
    return expected


def navigation_state(obs,baseline):
    from .replanning import number,integer
    from .stall_replay import log_count
    if (obs.get('server_available')!=1 or obs.get('error',0)!=0 or obs.get('server_alive')!=1
            or not integer(baseline) or not number(obs.get('screen',0))):
        raise ValueError('subgoal requires valid live server observation')
    gui_open=obs.get('screen',0)!=0
    position=[obs['server_'+axis] for axis in 'xyz']
    kind=obs['wood_v3_kind'];distance=obs['wood_v3_distance'];tick=obs['server_tick']
    if (any(not number(x,-30000000) for x in position) or kind not in (0,1,2)
            or obs['wood_v3_reachable'] not in (0,1) or not number(distance)
            or not number(tick) or int(tick)!=tick):raise ValueError('invalid navigation evidence')
    target=[obs['wood_v3_target_'+a] for a in 'xyz'] if kind else None
    if target and any(not number(v,-30000000) or int(v)!=v for v in target):
        raise ValueError('invalid navigation target')
    reachable=bool(kind and obs['wood_v3_reachable'])
    mining=bool(kind==1 and reachable and obs.get('wood_server_mining')==1
                and [obs.get('wood_mining_'+a) for a in 'xyz']==target)
    gain=log_count(obs)-baseline
    phase=('completed' if gain>=1 else 'gui_open' if gui_open else 'collecting' if kind==2 else 'mining' if mining else
           'reachable' if reachable else 'tracked_unreachable' if kind else 'unknown')
    return dict(position=position,tick=int(tick),target=target,kind=int(kind),target_known=bool(kind),
        distance=distance,reachable=reachable,mining=mining,inventory_gain=gain,phase=phase,
        interaction_ready=bool(not gui_open and (kind==2 or (kind==1 and reachable))),
        gui_open=gui_open,source='privileged_server')


def validate_navigation(value):
    from .replanning import number,integer
    fields={'position','tick','target','kind','target_known','distance','reachable','mining',
            'inventory_gain','phase','interaction_ready','gui_open','source'}
    if not isinstance(value,dict) or set(value)!=fields:raise ValueError('invalid abstract navigation state')
    if (not isinstance(value['position'],list) or len(value['position'])!=3
            or any(not number(x,-30000000) for x in value['position'])
            or not integer(value['tick']) or not integer(value['kind']) or value['kind']>2
            or not number(value['distance']) or not integer(value['inventory_gain'],-1000000)
            or any(type(value[k]) is not bool for k in ('target_known','reachable','mining','interaction_ready','gui_open'))
            or value['source']!='privileged_server'):raise ValueError('invalid abstract navigation values')
    kind=value['kind'];target=value['target']
    if ((kind==0 and target is not None) or (kind and (not isinstance(target,list) or len(target)!=3 or
            any(not number(v,-30000000) or int(v)!=v for v in target)))):
        raise ValueError('invalid abstract target')
    phase=('completed' if value['inventory_gain']>=1 else 'gui_open' if value['gui_open'] else 'collecting' if kind==2 else
           'mining' if value['mining'] else 'reachable' if value['reachable'] else
           'tracked_unreachable' if kind else 'unknown')
    if (value['target_known']!=bool(kind) or value['phase']!=phase or
            (value['reachable'] and not kind) or (value['mining'] and (kind!=1 or not value['reachable'])) or
            value['interaction_ready']!=bool(not value['gui_open'] and (kind==2 or (kind==1 and value['reachable'])))):
        raise ValueError('abstract phase contradicts observation')
    return deepcopy(value)


def completion(skill_id,before,current):
    validate_navigation(before);validate_navigation(current)
    if current['tick']<before['tick']:raise ValueError('subgoal observation predates entry')
    if current['gui_open']:return False
    if skill_id=='leave_stalled_area':
        return math.hypot(current['position'][0]-before['position'][0],
                          current['position'][2]-before['position'][2])>=3.-1e-6
    if skill_id=='locate_wood':return current['target_known']
    if skill_id=='approach_wood':return current['interaction_ready']
    raise ValueError('unsupported subgoal completion')


def load_catalogue(path,root):
    import numpy as np
    from .world_store import digest
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root/'results') or path.stat().st_size>16384:
        raise ValueError('project-local bounded skill catalogue required')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema')!=1 or set(manifest.get('skills',{}))!=set(SKILLS):
        raise ValueError('invalid skill catalogue')
    loaded={}
    for name,item in manifest['skills'].items():
        source=(root/item['path']).resolve()
        if (not source.is_relative_to(root/'results') or source.stat().st_size>16384
                or digest(source)!=item['file_sha256'] or item['prompt']!=SKILLS[name]['prompt']):
            raise ValueError('skill embedding provenance mismatch')
        value=np.load(source,allow_pickle=False)
        if value.shape!=(1,512) or value.dtype.kind!='f' or not np.isfinite(value).all():
            raise ValueError('invalid skill embedding')
        sha=hashlib.sha256(value.tobytes()).hexdigest()
        if sha!=item['embedding_sha256']:raise ValueError('skill tensor hash mismatch')
        loaded[name]=dict(embedding=value,sha256=sha,provenance=deepcopy(item))
    return loaded
