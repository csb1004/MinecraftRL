"""Explicitly authorized, one-use two-call probe; never dispatches game input."""
import argparse
import json
import os
from pathlib import Path

from minecraftrl.recovery_draft import RecoveryDraftProvider
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-paid',action='store_true');a=p.parse_args()
    if not a.allow_paid: raise SystemExit('Paid recovery proposal probe requires explicit user approval; no request sent.')
    root=Path(__file__).resolve().parents[1];base=root/'results/recovery-draft-preparation-20261009'
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    manifest=read(base/'manifest.json')
    assert manifest['prepared'] and manifest['maximum_calls']==2 and len(manifest['cases'])==2
    for path,sha in manifest['source_sha256'].items(): assert digest(path)==sha,path
    key=os.environ.get('OPENAI_API_KEY');model=os.environ.get('OPENAI_MODEL')
    if not key or model!=manifest['model']:
        raise SystemExit('Process-local API key and the prepared model setting are required; no request sent.')
    # Directory claim is exclusive. A crash/failed call never permits silent retry.
    output=base/'paid-probe';output.mkdir(exist_ok=False)
    provider=RecoveryDraftProvider(model,key,allow_paid=True,max_calls=2)
    report=dict(completed=False,attempts=[],model=model,maximum_calls=2,retries=0,new_game_actions=0,
        execution_authorized=False,manifest_sha256=digest(base/'manifest.json'),scope=manifest['scope'])
    for case in manifest['cases']:
        path=Path(case['input']);assert digest(path)==case['sha256']
        attempt=dict(label=case['label'],request_claimed=True,status='started',input_sha256=case['sha256'])
        report['attempts'].append(attempt);write_json(output/'report.json',report)
        try:
            proposal=provider.propose(read(path));attempt.update(status='contract_valid',proposal=proposal)
        except Exception as exc:
            attempt.update(status='failed',error=type(exc).__name__+': '+str(exc))
        write_json(output/'report.json',report)
    report.update(completed=True,remaining_calls=provider.remaining)
    write_json(output/'report.json',report)
    print(json.dumps(dict(completed=True,calls=len(report['attempts']),
                         contract_valid=sum(a['status']=='contract_valid' for a in report['attempts']),new_game_actions=0)))


if __name__=='__main__':main()
