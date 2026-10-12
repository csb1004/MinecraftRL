"""One-use proposal bridge. Caller constructs an explicitly authorized provider."""
import time

from .memory import atomic_json
from .recovery_exchange import read_bounded
from .recovery_live import checked_reply,validate_live_packet
from .world_store import digest


def serve_once(exchange,provider_factory,stop_file,*,wait_timeout_s=900.,live=False):
    from pathlib import Path
    exchange,stop_file=Path(exchange),Path(stop_file)
    started=time.monotonic()
    def closed():return stop_file.exists() or (exchange/'closed.json').exists()
    while not (exchange/'request.json').exists():
        if closed():return dict(status='stopped',provider_invocations=0)
        if time.monotonic()-started>=wait_timeout_s:return dict(status='request_wait_timeout',provider_invocations=0)
        time.sleep(.02)
    if closed():return dict(status='stopped',provider_invocations=0)
    request_path=exchange/'request.json'
    if time.time()-request_path.stat().st_mtime>=20:raise ValueError('expired mailbox request')
    request=read_bounded(request_path,131072)
    if not isinstance(request,dict) or set(request)!={'schema','packet'} or request['schema']!=1:
        raise ValueError('invalid mailbox request schema')
    packet=request['packet']
    rm_plan=packet.get('protocol')=='rm_plan_v1'
    if rm_plan:
        from .rm_planner import validate_packet,validate_plan
        validate_packet(packet)
        if packet.get('mode')=='archived_review':raise ValueError('archived analysis cannot enter live broker')
    else:validate_live_packet(packet)
    if (exchange/'response.json').exists():raise FileExistsError('response already exists')
    # Exclusive claim before provider creation; crashes/failures never allow a second call.
    claim=exchange/'broker-claim.json'
    with claim.open('x',encoding='utf-8') as stream:
        import json
        json.dump(dict(request_id=packet['request_id'],request_sha256=digest(request_path)),stream)
    report=dict(status='claimed',request_id=packet['request_id'],request_sha256=digest(request_path),
        provider_invocations=0,retries=0,execution_authorized=False,new_game_actions=0)
    def save():atomic_json(exchange/'broker-result.json',report)
    save()
    try:
        if closed():report['status']='stopped';return report
        provider=provider_factory()
        if closed():report['status']='stopped';return report
        if time.time()-request_path.stat().st_mtime>=20:
            report['status']='request_expired';return report
        report['provider_invocations']=1;save();call_started=time.monotonic()
        result=provider.propose(packet)
        report['provider_elapsed_s']=time.monotonic()-call_started
        validated=validate_plan(result,packet) if rm_plan else checked_reply(result,packet,live=live)
        report['proposal']=validated
        if closed():report['status']='discarded_after_close';return report
        atomic_json(exchange/'response.json',dict(request_id=packet['request_id'],proposal=validated))
        report['status']='delivered'
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__)
    finally:save()
    return report
