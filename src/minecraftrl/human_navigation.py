"""Human reference BC, with absent root RM and no invented completion labels."""
import json
import math
from pathlib import Path
from minecraftrl.world_store import digest


def reference_motion_goal(current,previous,target,remaining):
    if type(remaining) is not int or not 0<=remaining<=100:raise ValueError('invalid assigned motion budget')
    if current.get('isGuiOpen') is not False or previous.get('isGuiOpen') is not False:
        raise ValueError('human GUI transition is not movement supervision')
    if type(current['tick']) is not int or type(previous['tick']) is not int or current['tick']-previous['tick']!=1:
        raise ValueError('human observations must be consecutive integer ticks')
    values=[current[k] for k in ('xpos','ypos','zpos','yaw','pitch')]
    values += [previous[k] for k in ('xpos','ypos','zpos','yaw','pitch')]+list(target)
    if len(target)!=3 or not all(type(v) in (int,float) and math.isfinite(v) for v in values):
        raise ValueError('invalid human pose or reference point')
    if abs(current['pitch'])>90 or abs(previous['pitch'])>90:raise ValueError('invalid human pitch')
    dx,dy,dz=[target[i]-current[k] for i,k in enumerate(('xpos','ypos','zpos'))]
    yaw=math.radians(current['yaw']);angle=math.atan2(-dx,dz)-yaw
    mx=current['xpos']-previous['xpos'];mz=current['zpos']-previous['zpos']
    bounded=lambda value:max(-1.,min(1.,value))
    return [math.sin(angle),math.cos(angle),min(math.hypot(dx,dz)/16,1.),bounded(dy/4),current['pitch']/90,1.,
        bounded(mx*math.cos(yaw)+mz*math.sin(yaw)),bounded(-mx*math.sin(yaw)+mz*math.cos(yaw)),
        ((current['yaw']-previous['yaw']+180)%360-180)/180,remaining/100.]


def checked_file(root,entry):
    path=(root/entry['path']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('invalid human data path')
    if digest(path)!=entry['sha256']:raise ValueError('human data hash mismatch')
    return path


def load_reference_dataset(root,contract,actor,mapper,transformer):
    import numpy as np
    import torch
    from minecraftrl.contractor_actions import ContractorActions
    if contract.get('schema')!='human_navigation_reference_bc_v1':raise ValueError('unsupported human BC contract')
    mixed_pause=contract.get('supervision')=='far_and_observed_pause'
    if contract.get('supervision') not in ('far_reference_only','far_and_observed_pause') or contract.get('minimum_distance')!=1.:
        raise ValueError('unsupported human reference objective')
    if mixed_pause and contract.get('pause_reference_schema')!='observed_movement_pause_v1':
        raise ValueError('explicit observed-pause evidence contract required')
    if actor.get('schema') not in ('navigation_refinement_v3','navigation_refinement_v4') or actor.get('root_context_mode')!='ignored':
        raise ValueError('human RM is unobserved; navigation must structurally ignore RM')
    if actor.get('schema')=='navigation_refinement_v4' and (actor.get('visual_adapter_width')!=64 or actor.get('kind')!='mlp'):
        raise ValueError('unsupported version4 human navigation adapter')
    files={key:checked_file(root,contract[key]) for key in ('feature_report','features','reference_proposal','pair_manifest','visual_review','action_projection')}
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    report=read(files['feature_report']);proposal=read(files['reference_proposal']);pairs=read(files['pair_manifest'])['pairs']
    if not report['completed'] or report['actual_runtime_parity']['maximum_feature_error']!=0:
        raise ValueError('human encoder lineage unqualified')
    if report['features_sha256']!=digest(files['features']):raise ValueError('feature receipt mismatch')
    if (proposal['features_sha256']!=digest(files['features']) or
        proposal['feature_report_sha256']!=digest(files['feature_report']) or
        proposal['pair_manifest_sha256']!=digest(files['pair_manifest']) or
        proposal['visual_review_sha256']!=digest(files['visual_review'])):
        raise ValueError('human reference provenance mismatch')
    if proposal['source_features_modified'] or proposal['source_actions_modified']:
        raise ValueError('human originals must be preserved')
    bundle=torch.load(files['features'],map_location='cpu')
    if bundle['schema']!='common_visual_features_qualification_v1' or bundle['rm_context_observed'] or bundle['human_goal_intent_observed']:
        raise ValueError('unexpected human feature semantics')
    for key in ('parent_sha256','model_sha256','embedding_sha256'):
        if bundle['lineage'][key]!=actor[key] or bundle['lineage'][key]!=report['lineage'][key]:
            raise ValueError('human feature lineage differs from navigation actor: '+key)
    indexed={}
    for row in bundle['rows']:
        key=(row['session'],row['clip_start'],row['frame'])
        if key in indexed:raise ValueError('duplicate human feature')
        indexed[key]=row
    review=read(files['visual_review']);projection=read(files['action_projection'])
    if not projection['verified']:raise ValueError('official action projection unverified')
    result={'train':[],'development':[]};sessions={key:set() for key in result};receipts=[];used=set();goal_error=0.;pause_receipts=[]
    for clip in proposal['clips']:
        if not clip['labels']:continue
        identity=clip['session'];split=clip['split'];start,end=clip['start'],clip['end']
        if split not in result:raise ValueError('unsupported human split')
        if any(clip[k] for k in ('ground_observed','human_intent_observed','safe_route_verified','stable_stop_label')):
            raise ValueError('unavailable human evidence cannot be invented')
        pair=next(p for p in pairs if p['session']==identity)
        if pair['split']!=split or pair['jsonl_sha256']!=clip['jsonl_sha256']:raise ValueError('human split/provenance mismatch')
        source=checked_file(root,dict(path=pair['jsonl'],sha256=pair['jsonl_sha256']))
        checked_file(root,dict(path=pair['video'],sha256=pair['video_sha256']))
        raw=[json.loads(line) for line in source.open(encoding='utf-8')]
        target=[raw[end][k] for k in ('xpos','ypos','zpos')]
        if clip['target_frame']!=end or clip['target']!=target:raise ValueError('reference is not the recorded future point')
        if any(abs(raw[i]['ypos']-target[1])>1e-6 for i in range(end-3,end+1)):
            raise ValueError('unstable reference height; not an ordinary movement target')
        visual=next(w for w in review['windows'] if w['id'] in identity and w['start']==start)
        if visual['split']!=split or visual['end']!=end:raise ValueError('visual review window mismatch')
        projected=next(p for p in projection['reports'] if p['session']==identity and p['start']==start)
        if projected['end']!=end or projected['source_jsonl_sha256']!=clip['jsonl_sha256']:raise ValueError('action review provenance mismatch')
        actions={a['frame']:a for a in projected['actions']};converter=ContractorActions();converted={}
        for index in range(end):converted[index]=converter.convert(raw[index])
        reference_kind=clip.get('reference_kind','far')
        goal_source='hindsight_recorded_future_position'
        if reference_kind=='observed_pause':
            if not mixed_pause:raise ValueError('pause labels cannot enter the far-only contract')
            from minecraftrl.human_pause import observed_pause_evidence
            evidence=observed_pause_evidence(raw,converted,start,end)
            if clip.get('observed_pause_evidence')!=evidence:raise ValueError('pause evidence does not reproduce')
            expected=set(range(start+1,end));goal_source='hindsight_recorded_pause_position'
            pause_receipts.append(dict(session=identity,split=split,start=start,end=end,labels=len(expected),evidence=evidence))
        elif reference_kind=='far' and clip.get('observed_pause_evidence') is None:
            expected={i for i in range(start+1,end) if math.hypot(raw[i]['xpos']-target[0],raw[i]['zpos']-target[2])>=1.}
        else:raise ValueError('unknown or ambiguous human reference kind')
        if {label['frame'] for label in clip['labels']}!=expected or len(clip['labels'])!=len(expected):
            raise ValueError('human reference mask differs from declared rule')
        for label in clip['labels']:
            index=label['frame'];key=(identity,start,index)
            if (identity,index) in used:raise ValueError('human action reused in multiple training labels')
            used.add((identity,index));stored=indexed[key];current=raw[index]
            if stored['split']!=split or stored['rm_context'] is not None or stored['goal'] is not None:
                raise ValueError('human source context must stay unobserved')
            if stored['loss_eligible'] or stored['stable_stop_label'] or stored['first']:
                raise ValueError('unexpected original feature label or boundary')
            goal=reference_motion_goal(current,raw[index-1],target,100-(index-start))
            recorded=label['motion_goal']
            if len(recorded)!=10 or not all(type(v) in (int,float) and math.isfinite(v) for v in recorded):
                raise ValueError('invalid recorded human reference goal')
            error=max(abs(a-b) for a,b in zip(recorded,goal));goal_error=max(goal_error,error)
            if (label['previous_frame']!=index-1 or error>1e-12 or label['rm_context'] is not None or
                label['arrival_label'] is not None or label['terminal'] is not False or label['reference_budget_ticks']!=100 or
                label['loss_candidate'] is not True):
                raise ValueError('human reference labels do not reproduce')
            if label['goal_source']!=goal_source or label['pose_source']!='contractor_recorded_player_pose':
                raise ValueError('human reference source not explicit')
            values=stored['visual_features']
            if tuple(values.shape)!=(2,1,2048) or not torch.isfinite(values).all():raise ValueError('invalid frozen human features')
            converted_row=converted[index]
            if not converted_row['bc_eligible']:raise ValueError('unsupported human action')
            action={k:np.asarray(v) for k,v in converted_row['action'].items()}
            if np.abs(action['camera']).max()>10:raise ValueError('clipped human camera is not a label')
            if not actions[index]['physical_buttons_preserved']:raise ValueError('human action projection loses buttons')
            mapped=mapper.from_factored({k:v[None] for k,v in transformer.env2policy(action).items()})
            for name,value in mapped.items():
                if not torch.equal(torch.as_tensor(value).unsqueeze(1),stored['action'][name]):raise ValueError('human action reproduction mismatch')
                if value.tolist()!=actions[index]['mapped_action'][name]:raise ValueError('human action review mismatch')
            result[split].append(dict(features=values,motion_goal=None,bc_motion_goal=recorded,
                action=stored['action'],bc_action=stored['action'],owner='human',bc_source='hindsight_reference',
                human_session=identity,human_frame=index,arrival_label=None,rm_context=None))
            if reference_kind=='observed_pause':result[split][-1]['human_reference_kind']='observed_pause'
        sessions[split].add(identity);receipts.append(dict(session=identity,split=split,start=start,end=end,labels=len(expected)))
    if not result['train'] or not result['development'] or sessions['train']&sessions['development']:
        raise ValueError('human train/development split is empty or overlapping')
    receipt=dict(verified=True,sessions=receipts,counts={k:len(v) for k,v in result.items()},
        maximum_goal_reconstruction_error=goal_error,goal_reconstruction_tolerance=1e-12,
        source_files={k:digest(v) for k,v in files.items()},human_intent_observed=False,stop_demonstrations=0,
        actual_game_successes=0,rm_context_mode='ignored',scope='Human action imitation toward assigned far references only')
    if mixed_pause:
        receipt.update(observed_pause_references=pause_receipts,
            observed_pause_labels={split:sum(r['labels'] for r in pause_receipts if r['split']==split) for split in result},
            pause_source_sha256=digest(Path(__file__).with_name('human_pause.py')),
            scope='Far-reference and observed movement-pause imitation; no ground, intention, safety, terminal or task-completion labels')
    return result,receipt
