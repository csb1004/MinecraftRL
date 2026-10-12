"""Contracts for a separate, goal-conditioned behavioral cloning pilot."""
import math
import re


def validate_plan(plan):
    if not isinstance(plan.get('goal'),str) or not plan['goal'].strip():raise ValueError('missing source goal')
    if type(plan.get('frame_offset')) is not int or plan['frame_offset']!=0:raise ValueError('unverified frame offset')
    teacher=plan.get('data_source') in ('local_teacher','student_correction')
    if plan.get('rm_context')!=('recorded' if teacher else 'unobserved_frozen'):raise ValueError('source/RM mismatch')
    sessions=plan.get('sessions',[]);seen=set();splits=set();groups={}
    for s in sessions:
        if not isinstance(s.get('id'),str) or not s['id'] or s['id'] in seen:raise ValueError('duplicate/missing session')
        seen.add(s['id']);splits.add(s['split'])
        if s['split'] not in ('train','development'):raise ValueError('evaluation data is forbidden')
        # Five-minute contractor chunks retain the same recording identifier.
        match=re.fullmatch(r'.+-([0-9a-f]{12})-\d{8}-\d{6}',s['id'])
        group=match.group(1) if match else s['id']
        if group in groups and groups[group]!=s['split']:raise ValueError('recording session crosses splits')
        groups[group]=s['split']
        if not all(type(s[k]) is int for k in ('context_start','start','end')):raise ValueError('invalid frame bounds')
        if teacher:
            if not s['context_start']==s['start']==0 or s['end']<=0:raise ValueError('teacher must retain complete episode')
        elif not 0<=s['context_start']<s['start']<s['end']:raise ValueError('missing recurrent warmup')
    if splits!={'train','development'}:raise ValueError('both disjoint splits required')
    if plan.get('data_source')=='student_correction':
        from .correction_bc import validate_correction_plan
        validate_correction_plan(plan)


def loss_eligible(row,*,button_loss):
    return (row['bc_eligible'] and not row['gui'] and not button_loss and
            all(math.isfinite(x) and abs(x)<=10 for x in row['action']['camera']))


def comparison_gate(before_nll,after_nll,reference_kl,weight_delta):
    vals=(before_nll,after_nll,reference_kl,weight_delta)
    return (all(math.isfinite(x) for x in vals) and after_nll<=before_nll and
            -.000001<=reference_kl<=.03 and weight_delta>0)
