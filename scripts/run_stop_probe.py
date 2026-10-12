"""One explicit STOP experiment against a real policy/environment (not a suite sample)."""
import argparse
import datetime
import json
from pathlib import Path
import threading
import time
from minecraftrl.cli import condition_for, episode
from minecraftrl.runtime import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',choices=['vpt','steve1'],default='vpt')
    parser.add_argument('--trigger-step',type=int,default=1)
    parser.add_argument('--phase',choices=['initialization','inference','step'],required=True)
    args=parser.parse_args()
    if args.trigger_step < 1: parser.error('--trigger-step must be positive')
    root=Path(__file__).resolve().parents[1]; stop=root/'results/STOP'
    if stop.exists(): raise SystemExit('STOP is latched; inspect the previous shutdown first')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=root/'results'/('real-stop-'+args.policy+'-'+args.phase+'-'+stamp)
    finished=threading.Event(); request={}
    def watch():
        while not finished.wait(.005):
            if stop.exists(): return
            events=output/'events.jsonl'
            if not events.exists():
                finished.wait(.05); continue
            try:
                rows=[json.loads(x) for x in events.read_text().splitlines()]
            except (OSError,json.JSONDecodeError): continue
            trigger=None
            if args.phase=='initialization' and not rows:
                import psutil
                receipt=output/'runtime-jar-cache.json'
                if not receipt.exists(): continue
                cached=json.loads(receipt.read_text())['cached']
                for process in psutil.process_iter(['cmdline']):
                    command = process.info['cmdline'] or []
                    if cached in command and '-jar' in command:
                        trigger={'java_pid':process.pid,'created':process.create_time()}; break
            else:
                wanted='inference_started' if args.phase=='inference' else 'policy_dispatch_started'
                match=next((r for r in rows if r['event']==wanted and r['step']==args.trigger_step),None)
                if match: trigger=match
            if trigger is None: continue
            before=time.monotonic()
            try:
                with stop.open('x') as stream: stream.write('Explicit real STOP probe\n')
            except FileExistsError: return
            request.update(phase=args.phase,before_s=before,after_s=time.monotonic(),trigger=trigger)
            write_json(output/'stop-test-request.json',request)
            return
    watcher=threading.Thread(target=watch,daemon=True); watcher.start()
    run=argparse.Namespace(root=root,policy=args.policy,output=output,seed=104729,limit_s=180.,stop_file=stop)
    condition='spontaneous' if args.policy=='vpt' else 'wood'
    try: report=episode(run,condition_for(root,args.policy,condition))
    finally: finished.set(); watcher.join(timeout=2)
    events=output/'events.jsonl'
    rows=[json.loads(x) for x in events.read_text().splitlines()] if events.exists() else []
    ack=next((r['monotonic_s'] for r in rows if r['event']=='stop_acknowledged'),None)
    verified=False
    if request and args.phase!='initialization':
        end_name='inference_finished' if args.phase=='inference' else 'policy_dispatch_finished'
        end=next((r for r in rows if r['event']==end_name and r['step']==request['trigger']['step']),None)
        verified=bool(end and 'call_started_s' in end and
                      end['call_started_s']<=request['before_s']<=request['after_s']<=end['call_finished_s'])
    elif request:
        starts=[r for r in rows if r['event']=='episode_started']
        verified=not starts or request['after_s']<starts[0]['monotonic_s']
    result={'policy':args.policy,'phase':args.phase,'trigger_step':args.trigger_step,'request_during_phase_verified':verified,
            'acknowledged':ack is not None,
            'request_to_ack_ms':None if ack is None or not request else (ack-request['after_s'])*1000,
            'policy_dispatches_after_request':None if not request else sum(r['event']=='policy_dispatch_started' and r['monotonic_s']>request['after_s'] for r in rows),
            'policy_dispatches_after_ack':None if ack is None else sum(r['event']=='policy_dispatch_started' and r['monotonic_s']>ack for r in rows),
            'release':report.get('release'),'remaining_live_pids':report['process_cleanup']['remaining_live_pids'],
            'status':report['status'],'output':str(output),'suite_sample':False}
    write_json(output/'stop-test-evaluation.json',result)
    print(json.dumps(result,indent=2))
    # Keep STOP latched. The caller must inspect the shutdown before re-arming.


if __name__=='__main__': main()
