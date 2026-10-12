"""Offline, bounded Korean intent parser and independent commitment boundary.

This is an executable subset of the research dialogue interface, not an LLM.
Unrecognized language never gets converted to a best-guess game action.
"""
import re
from .tasks import validate_command


_COLLECT=re.compile(r'(?:지금 )?(원목|나무|흙) ([0-9]{1,3})개 (?:모아줘|모아 주세요|수집해줘|수집해 주세요)[.!]?')
_MODIFY=re.compile(r'목표를 (원목|나무|흙) ([0-9]{1,3})개로 (?:바꿔줘|바꿔 주세요)[.!]?')
_CONTROLS={'잠시 멈춰줘':'pause','일시 중단해줘':'pause',
           '다시 계속해줘':'resume','작업 재개해줘':'resume',
           '작업 취소해줘':'cancel','작업 취소해 주세요':'cancel'}
_LABELS={'request':'TASK_REQUEST','modify':'TASK_MODIFY','pause':'TASK_PAUSE',
         'resume':'TASK_RESUME','cancel':'TASK_CANCEL'}


def _parse(text):
    # Full-sentence matching keeps quotation, negation and additional clauses
    # outside the authorization grammar. Do not remove them by normalization.
    match=_COLLECT.fullmatch(text)
    op='request'
    if match is None:
        match=_MODIFY.fullmatch(text); op='modify'
    if match:
        goal='dirt' if match[1]=='흙' else 'wood'
        return _LABELS[op],dict(op=op,goal=goal,count=int(match[2]))
    op=_CONTROLS.get(text.rstrip('.!'))
    if op: return _LABELS[op],dict(op=op)
    if text.endswith('?'): return 'QUESTION',None
    if '나중에' in text or text.endswith(('해보자.','해보자','모아보자.','모아보자')):
        return 'TASK_PROPOSAL',None
    if '예쁘' in text or '좋겠' in text: return 'PREFERENCE',None
    if text in ('안녕','고마워','잘했어'): return 'CHAT',None
    return 'UNRESOLVED',None


def decide(message, *, candidate=None):
    """Compile only literal supported authorization; candidate has no authority.

    Context comes from the caller's acknowledged task/revision. The worker
    validates it again against live state before changing any task.
    """
    allowed={'kind','message_id','text','task_id','expected_revision'}
    if not isinstance(message,dict) or set(message)-allowed or message.get('kind')!='dialogue':
        raise ValueError('invalid dialogue envelope')
    for key in ('message_id','task_id'):
        if not isinstance(message.get(key),str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',message[key]):
            raise ValueError('invalid '+key)
    text=message.get('text')
    if not isinstance(text,str) or not 0<len(text)<=2048: raise ValueError('invalid utterance length')
    if 'expected_revision' in message and (type(message['expected_revision']) is not int or message['expected_revision']<1):
        raise ValueError('invalid expected_revision')
    label,fields=_parse(text.strip())
    result=dict(parser='offline-ko-v1',utterance_type=label,command=None,
                status='no_action',reason='outside_supported_authorization')
    if fields is None: return result
    command=dict(command_id=message['message_id'],task_id=message['task_id'],**fields)
    if fields['op']!='request':
        if 'expected_revision' not in message:
            result.update(status='needs_context',reason='acknowledged_revision_required'); return result
        command['expected_revision']=message['expected_revision']
    elif 'expected_revision' in message:
        result.update(status='needs_clarification',reason='use_explicit_modify_for_existing_task'); return result
    try: validate_command(command)
    except ValueError as exc:
        result.update(status='needs_clarification',reason=str(exc)); return result
    if candidate is not None and (not isinstance(candidate,dict) or candidate!=command):
        result.update(status='rejected',reason='semantic_candidate_disagrees_with_authorization'); return result
    result.update(command=command,status='eligible',reason='worker_state_validation_required')
    return result
