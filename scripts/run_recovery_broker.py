"""Windows proposal broker: explicit fixed fixture or separately authorized one-call API."""
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import time

from minecraftrl.memory import atomic_json
from minecraftrl.recovery_broker import serve_once
from minecraftrl.recovery_draft import validate_draft
from minecraftrl.recovery_live import LiveRecoveryProvider
from minecraftrl.rm_planner import RMPlanProvider,fixture_plan,validate_plan
from minecraftrl.semantic import _transport
from minecraftrl.world_store import digest


def make_api_provider(model,key,*,rm_plan,transport):
    provider=RMPlanProvider if rm_plan else LiveRecoveryProvider
    return provider(model,key,allow_paid=True,max_calls=1,transport=transport)


def fixture_reply(packet,choice,*,rm_plan):
    if rm_plan:
        reply=fixture_plan(packet)
        if choice=='remain_paused':reply.update(decision='pause',steps=[])
        return validate_plan(reply,packet)
    return validate_draft(dict(request_id=packet['request_id'],evidence_id=packet['evidence_id'],
        choice=choice,reason='Fixed Windows broker response; no model call.'),packet)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--exchange',type=Path,required=True)
    p.add_argument('--rm-plan',action='store_true',help='Use full RM subgoal plans; paid mode still requires separate approval')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fixture-choice',choices=['explore_then_resume','remain_paused'])
    mode.add_argument('--allow-paid',action='store_true')
    p.add_argument('--fixture-delay',type=float,default=5.)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];exchange=a.exchange.resolve()
    if not exchange.is_relative_to(root/'results'):raise SystemExit('Exchange must be inside project results')
    if not 0<=a.fixture_delay<=60:raise SystemExit('Invalid fixed response delay')
    stop=root/'results/STOP'
    def closed():return stop.exists() or (exchange/'closed.json').exists()
    class Fixture:
        def propose(self,packet):
            until=time.monotonic()+a.fixture_delay
            while time.monotonic()<until:
                if closed():raise InterruptedError('Fixture request closed')
                time.sleep(.02)
            return fixture_reply(packet,a.fixture_choice,rm_plan=a.rm_plan)
    def provider_factory():
        if not a.allow_paid:return Fixture()
        # Called only after live evidence validation and an exclusive request claim.
        from dotenv import dotenv_values
        mywife=Path('D:/DATA/Desktop/MyWife')
        with contextlib.redirect_stderr(io.StringIO()):settings=dotenv_values(mywife/'.env')
        default_model=None
        for node in ast.parse((mywife/'firefly/config.py').read_text(encoding='utf-8')).body:
            if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='DEFAULT_MODEL' for n in node.targets):
                default_model=ast.literal_eval(node.value)
        key=os.getenv('OPENAI_API_KEY') or settings.get('OPENAI_API_KEY')
        model=os.getenv('TOOL_PLANNER_MODEL') or settings.get('TOOL_PLANNER_MODEL') or default_model
        if not key or model!='gpt-5.3-codex':raise ValueError('Existing key/model does not match prepared scope')
        receipt=dict(model=model,maximum_calls=1,network_calls=0,retries=0,
            max_output_tokens=2048 if a.rm_plan else 1024,timeout_s=20,
            protocol='rm_plan_v1' if a.rm_plan else 'bounded_recovery_choice',
            new_game_actions=0,key_recorded=False,request_sha256=digest(exchange/'request.json'))
        receipt_path=exchange/'api-receipt.json'
        with receipt_path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2)
        def transport(request,timeout):
            if receipt['network_calls'] or closed() or time.time()-(exchange/'request.json').stat().st_mtime>=20:
                raise RuntimeError('Call already used or episode closed/expired')
            assert request.full_url=='https://api.openai.com/v1/responses' and timeout==20
            receipt['network_calls']=1;atomic_json(receipt_path,receipt);started=time.monotonic()
            try:
                response=_transport(request,timeout)
                atomic_json(exchange/'api-response.json',response)
                receipt.update(status=response.get('status'),response_id=response.get('id'),usage=response.get('usage'),
                    response_sha256=digest(exchange/'api-response.json'))
                return response
            except Exception as exc:
                receipt.update(status='failed',error_type=type(exc).__name__)
                raise RuntimeError('bounded_live_transport_failed') from None
            finally:
                receipt['elapsed_s']=time.monotonic()-started;atomic_json(receipt_path,receipt)
        return make_api_provider(model,key,rm_plan=a.rm_plan,transport=transport)
    result=serve_once(exchange,provider_factory,stop,live=a.allow_paid)
    print(json.dumps(dict(status=result['status'],provider_invocations=result['provider_invocations'],
        mode='paid_live' if a.allow_paid else 'fixed_local',new_game_actions=0)))
    return 0 if result['status'] in ('delivered','discarded_after_close','stopped') else 2


if __name__=='__main__':raise SystemExit(main())
