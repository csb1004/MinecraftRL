"""Offline scoring of recorded responses; no model calls or game commands."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from .semantic import PLANNER, decode_response, validate_candidate


def evaluate(cases,records,*,current_prompt_sha256):
    indexed={}
    for case in cases:
        if case['id'] in indexed: raise ValueError('duplicate_case')
        validate_candidate(case['expected']); indexed[case['id']]=case
    recorded={}
    for record in records:
        identity=record.get('case_id')
        if identity not in indexed or identity in recorded: raise ValueError('unknown_or_duplicate_record')
        if not re.fullmatch('[0-9a-f]{64}',record.get('prompt_sha256','')): raise ValueError('missing_prompt_provenance')
        expected_input=indexed[identity].get('input_sha256')
        if expected_input is not None and record.get('input_sha256')!=expected_input:
            raise ValueError('input_context_mismatch')
        recorded[identity]=record
    report=dict(planned=len(cases),evaluated=0,unrun=0,response_errors=0,label_matches=0,candidate_matches=0,
                label_mismatches=0,current_prompt_evaluated=0,current_prompt_sha256=current_prompt_sha256,results=[])
    for identity,case in indexed.items():
        result={'case_id':identity,'expected':case['expected']}; report['results'].append(result)
        record=recorded.get(identity)
        if record is None:
            result['status']='unrun'; report['unrun']+=1; continue
        result['prompt_sha256']=record['prompt_sha256']
        if 'input_sha256' in case: result['input_sha256']=record['input_sha256']
        result['current_prompt']=record['prompt_sha256']==current_prompt_sha256
        report['evaluated']+=1; report['current_prompt_evaluated']+=int(result['current_prompt'])
        try: actual=decode_response(record.get('response'))
        except (ValueError,TypeError):
            result['status']='response_error'; report['response_errors']+=1; continue
        fields=['intent','goal','count']
        if 'steps' in actual or 'steps' in case['expected']: fields.append('steps')
        mismatch=[key for key in fields if actual.get(key)!=case['expected'].get(key)]
        result.update(actual=actual,mismatched_fields=mismatch,status='candidate_mismatch' if mismatch else 'candidate_match')
        report['label_mismatches' if 'intent' in mismatch else 'label_matches']+=1
        report['candidate_matches']+=int(not mismatch)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True,help='JSON list: case_id, prompt_sha256, response')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=evaluate(json.loads(args.cases.read_text(encoding='utf-8-sig')),
                    json.loads(args.records.read_text(encoding='utf-8-sig')),
                    current_prompt_sha256=hashlib.sha256(PLANNER.encode('utf-8')).hexdigest())
    # Exclusive creation preserves past evaluation evidence.
    with args.output.open('x',encoding='utf-8') as stream: json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))
    # Exit success means scoring completed, not that model accuracy was validated.
    return 0


if __name__=='__main__': raise SystemExit(main())
