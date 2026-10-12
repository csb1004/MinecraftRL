"""One paid archived plan-generation call, only after separate explicit approval."""
import argparse
import json
import os
from pathlib import Path
import time

from minecraftrl.rm_planner import RMPlanProvider,validate_plan
from minecraftrl.semantic import response_text
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--authorization',type=Path,required=True);p.add_argument('--allow-paid',action='store_true')
    args=p.parse_args()
    if not args.allow_paid:raise SystemExit('Separate paid-call approval required; no request sent.')
    root=Path(__file__).resolve().parents[1];base=args.prepared.resolve()
    if not base.is_relative_to(root/'results'):raise ValueError('invalid preparation path')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(base/'manifest.json');authorization=read(args.authorization)
    if (authorization.get('approved') is not True or authorization.get('maximum_calls')!=1
            or authorization.get('manifest_sha256')!=digest(base/'manifest.json')):
        raise SystemExit('Approval does not match the concrete one-call preparation.')
    for name,sha in manifest['source_sha256'].items():
        if digest(root/name)!=sha:raise ValueError('prepared source/input changed: '+name)
    key=os.environ.get('OPENAI_API_KEY');model=os.environ.get('OPENAI_MODEL')
    if not key or model!=manifest['model']:raise SystemExit('Prepared process-local model/key setting required.')
    output=base/'paid-probe';output.mkdir(exist_ok=False)
    provider=RMPlanProvider(model,key,allow_paid=True,max_calls=1)
    original=provider.transport
    receipt=dict(network_calls=0,retries=0,new_game_actions=0,completed=False,
                 manifest_sha256=digest(base/'manifest.json'))
    def transport(request,timeout):
        receipt['network_calls']+=1;write_json(output/'receipt.json',receipt)
        response=original(request,timeout)
        write_json(output/'response.json',response);return response
    provider.transport=transport;started=time.monotonic()
    try:
        response=provider._call(read(base/'request-payload.json'))
        plan=validate_plan(json.loads(response_text(response)),read(base/'packet.json'))
        write_json(output/'proposal.json',plan)
        receipt.update(completed=True,status='contract_valid',usage=response.get('usage'))
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__)
    finally:
        receipt['elapsed_s']=time.monotonic()-started;write_json(output/'receipt.json',receipt)
    print(json.dumps(receipt))
    return 0 if receipt['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
