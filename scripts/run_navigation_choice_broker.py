"""One paid candidate choice under the user's persistent $1 / four-call budget."""
import argparse,ast,contextlib,io,json,os,time
from pathlib import Path
from minecraftrl.memory import atomic_json
from minecraftrl.navigation_choice import CandidateProvider,PaidBudget,validate_packet,validate_choice
from minecraftrl.recovery_exchange import read_bounded
from minecraftrl.semantic import _transport
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--exchange',type=Path,required=True);p.add_argument('--allow-paid',action='store_true',required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];batch=root/'results/adaptive-recovery-20261014';exchange=a.exchange.resolve()
    if not exchange.is_relative_to(batch) or exchange.name not in ('request-001','request-002'):raise ValueError('invalid paid exchange')
    auth=read_bounded(batch/'authorization.json')
    if not auth['approved'] or auth['maximum_usd']!=1 or auth['maximum_calls']!=4 or auth['model']!='gpt-5.3-codex':raise ValueError('budget not authorized')
    stop=root/'results/STOP';request_path=exchange/'request.json'
    closed=lambda:stop.exists() or (exchange/'closed.json').exists() or (exchange.parent/'closed.json').exists()
    packet=validate_packet(read_bounded(request_path)['packet'])
    with (exchange/'broker-claim.json').open('x',encoding='utf-8') as f:json.dump(dict(request_sha256=digest(request_path)),f)
    report=dict(status='claimed',provider_invocations=0,retries=0);receipt=None
    atomic_json(exchange/'broker-result.json',report)
    try:
        if closed() or time.time()-request_path.stat().st_mtime>=20:report['status']='stopped';return 0
        from dotenv import dotenv_values
        mywife=Path('D:/DATA/Desktop/MyWife')
        with contextlib.redirect_stderr(io.StringIO()):settings=dotenv_values(mywife/'.env')
        default=None
        for n in ast.parse((mywife/'firefly/config.py').read_text(encoding='utf-8')).body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_MODEL' for t in n.targets):default=ast.literal_eval(n.value)
        key=os.getenv('OPENAI_API_KEY') or settings.get('OPENAI_API_KEY')
        model=os.getenv('TOOL_PLANNER_MODEL') or settings.get('TOOL_PLANNER_MODEL') or default
        if not key or model!='gpt-5.3-codex':raise ValueError('existing key/model outside approved scope')
        receipt=dict(model=model,network_calls=0,retries=0,max_output_tokens=2048,timeout_s=20,service_tier='default',key_recorded=False)
        receipt_path=exchange/'api-receipt.json'
        with receipt_path.open('x',encoding='utf-8') as f:json.dump(receipt,f)
        def transport(request,timeout):
            if receipt['network_calls'] or closed() or time.time()-request_path.stat().st_mtime>=20:raise RuntimeError('closed/expired/used request')
            if request.full_url!='https://api.openai.com/v1/responses' or timeout!=20:raise ValueError('unexpected transport')
            # 16 KiB body + conservative 8192 framing input tokens, 2048 output tokens,
            # even at twice standard published rates: < $0.15, reserved $0.25.
            PaidBudget(batch).reserve(packet['request_id'],len(request.data))
            receipt.update(network_calls=1,reserved_usd=.25,request_bytes=len(request.data));atomic_json(receipt_path,receipt)
            started=time.monotonic()
            try:
                response=_transport(request,timeout);atomic_json(exchange/'api-response.json',response)
                usage=response.get('usage') or {};cost=(usage.get('input_tokens',0)*1.75+usage.get('output_tokens',0)*14)/1e6
                receipt.update(status=response.get('status'),response_id=response.get('id'),usage=usage,
                    estimated_usd=cost if usage else None,response_sha256=digest(exchange/'api-response.json'))
                return response
            except Exception as exc:
                receipt.update(status='failed',error_type=type(exc).__name__,http_status=getattr(exc,'code',None))
                raise RuntimeError('bounded_candidate_transport_failed') from None
            finally:receipt['elapsed_s']=time.monotonic()-started;atomic_json(receipt_path,receipt)
        provider=CandidateProvider(model,key,allow_paid=True,max_calls=1,transport=transport)
        report['provider_invocations']=1;atomic_json(exchange/'broker-result.json',report)
        proposal=validate_choice(provider.propose(packet),packet);report['proposal']=proposal
        if closed():report['status']='discarded_after_close'
        else:
            atomic_json(exchange/'response.json',dict(request_id=packet['request_id'],proposal=proposal));report['status']='delivered'
    except Exception as exc:report.update(status='failed',error_type=type(exc).__name__)
    finally:atomic_json(exchange/'broker-result.json',report)
    print(json.dumps(dict(status=report['status'],network_calls=(receipt or {}).get('network_calls',0))))
    return 0 if report['status'] in ('delivered','discarded_after_close','stopped') else 2


if __name__=='__main__':raise SystemExit(main())
