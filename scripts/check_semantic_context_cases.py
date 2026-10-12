"""Offline contract checks using expected candidates, NOT model accuracy evaluation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from minecraftrl.semantic import PLANNER, OpenAISemantic, Conversation, OPS
from minecraftrl.semantic_evaluation import evaluate


def check(cases):
    evidence=[]
    for case in cases:
        sent=[]
        def fixture(request,timeout):
            payload=json.loads(request.data); context=json.loads(payload['input'][-1]['content'])
            digest=hashlib.sha256(json.dumps(context,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
            if context!=case['context'] or digest!=case['input_sha256']:
                raise ValueError('fixture_input_context_changed')
            sent.append(digest)
            return {'status':'completed','output':[{'type':'message','content':[
                {'type':'output_text','text':json.dumps(case['expected'])}]}]}
        provider=OpenAISemantic('offline-contract-fixture','not-a-key',allow_paid=True,transport=fixture,max_calls=1)
        context=case['context']; task=copy.deepcopy(context['current_task']); original=copy.deepcopy(task)
        candidate=provider.parse(context['current_utterance'],context['recent_dialogue'],task,
                                 summary=context['dialogue_summary'])
        conversation=Conversation(); decision=conversation.propose(context['current_utterance'],candidate,task)
        if decision['command'] is not None or task!=original: raise AssertionError('Proposal changed authoritative state')
        op=OPS.get(candidate['intent'])
        if op:
            if decision['status']!='needs_confirmation': raise AssertionError('Expected confirmation')
            token=decision['confirmation_id']
            confirmed=conversation.confirm('확인 '+token,task)['command']
            if confirmed['op']!=op or confirmed['expected_revision']!=task['revision']:
                raise AssertionError('Wrong task binding')
            proposal=conversation.propose(context['current_utterance'],candidate,task)
            stale=conversation.confirm('확인 '+proposal['confirmation_id'],dict(task,revision=task['revision']+1))
            if stale['status']!='confirmation_expired': raise AssertionError('Old revision accepted')
        elif decision['status']!='no_action': raise AssertionError('Nonexecuting case became a proposal')
        evidence.append({'case_id':case['id'],'input_sha256':sent[0],'fixture_contract_passed':True,
                         'authoritative_task_unchanged':task==original,'decision_status':decision['status']})
    return {'mode':'synthetic_expected_candidate_contract_only','additional_paid_api_calls':0,
            'game_runs':0,'cases':evidence,
            'model_evaluation':evaluate(cases,[],current_prompt_sha256=hashlib.sha256(PLANNER.encode('utf-8')).hexdigest())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=check(json.loads(args.cases.read_text(encoding='utf-8')))
    with args.output.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'fixture_contract_passed':len(result['cases']),'actual_model_unrun':result['model_evaluation']['unrun']}))


if __name__=='__main__': main()
