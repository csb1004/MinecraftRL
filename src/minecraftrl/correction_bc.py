"""Admission and loss masks for complete student -> neutral -> teacher episodes."""


def validate_correction_plan(plan):
    if plan.get('schema')!='student_correction_v1' or plan.get('masked_loss_required') is not True:
        raise ValueError('explicit masked correction contract required')
    if plan.get('parent')!=plan.get('checkpoint') or plan.get('reward_mode')!='wood-potential-v4':
        raise ValueError('correction parent/context mismatch')
    for s in plan['sessions']:
        allowed=(s['split']=='train' and s.get('seed') in (104743,104759)) or (
            s['split']=='development' and s.get('seed')==104761)
        if not allowed or not s.get('eligible_for_future_masked_bc'):
            raise ValueError('unapproved correction split/source')
        if len(s.get('handoffs',[]))!=1:raise ValueError('one verified handoff required')
        h=s['handoffs'][0];start=h['step']
        if (type(start) is not int or not 1<start<s['end'] or not h.get('verified') or
            h.get('held_key_count')!=0 or not h.get('teacher_started') or
            h.get('trigger',{}).get('step')!=start-1 or h['trigger'].get('event')!='stalled'):
            raise ValueError('invalid neutral/teacher boundary')
        frames=s.get('frames',[])
        if len(frames)!=s['end'] or s.get('expert_labels')!=s['end']-start:
            raise ValueError('incomplete correction sequence')
        for i,f in enumerate(frames):
            if f.get('path')!=f'frames/{i:06d}.png' or type(f.get('loss_eligible')) is not bool or f['loss_eligible']!=(i>=start):
                raise ValueError('correction mask/frame mismatch')
            if f['sha256']!=s['frame_sha256'][i]:raise ValueError('frame hash mismatch')


def correction_owner(session,index):
    handoff=session['handoffs'][0]['step']
    return 'student' if index<handoff-1 else 'handoff_neutral' if index==handoff-1 else 'teacher'


def supervised_windows(rows,size=8):
    from .temporal_bc import windows
    return [(start,end) for start,end in windows(len(rows),size)
            if any(r.get('loss_eligible',True) for r in rows[start:end])]


def supervised_batch(rows,features):
    """Select targets only after recurrent replay; context has no action label."""
    import torch
    indices=[i for i,r in enumerate(rows) if r.get('loss_eligible',True)]
    if features.shape[1]!=len(rows) or not indices:
        raise ValueError('empty or mismatched supervision mask')
    selected=[rows[i] for i in indices]
    if any(not isinstance(r.get('action'),dict) for r in selected):raise ValueError('missing expert action')
    action={k:torch.cat([r['action'][k] for r in selected],1).to(features.device) for k in selected[0]['action']}
    base=torch.cat([r['features'] for r in selected],1).to(features.device)
    return features[:,indices],base,action
