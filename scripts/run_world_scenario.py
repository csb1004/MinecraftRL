"""Actual STEVE-1 save, process restart, restore and continued collection proof."""
import argparse
import datetime
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from minecraftrl.cli import episode, condition_for
from minecraftrl.runtime import write_json
from minecraftrl.task_runtime import write_command


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--semantic-fixture',action='store_true',help='Local synthetic Responses, real game; no API calls')
    args=parser.parse_args(); root=args.root.resolve()
    if (root/'results/STOP').exists(): raise SystemExit('Global STOP is latched')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix='semantic-world-scenario-' if args.semantic_fixture else 'world-scenario-'
    base=root/'results'/(prefix+stamp); base.mkdir()
    slot='validation-'+stamp; errors=[]; reports=[]; expected=0
    print('World scenario output: '+str(base),flush=True)
    try:
        for index in range(2):
            output=base/str(index); stop=output/'STOP'; finished=threading.Event()
            def watch():
                position=0; pending=b''; stage='ready'; resume_at=None; steps_before_resume=0
                fixture=None
                try:
                    while not finished.wait(.01):
                        if (root/'results/STOP').exists():
                            if output.exists(): stop.touch()
                            return
                        if resume_at is not None and time.monotonic()>=resume_at:
                            if fixture: fixture.confirm()
                            else: write_command(output/'commands',dict(command_id='resume',task_id='gather',op='resume',expected_revision=1))
                            resume_at=None; stage='collect'
                        events=output/'events.jsonl'
                        if not events.exists(): continue
                        with events.open('rb') as stream:
                            stream.seek(position); raw=stream.read(); position=stream.tell()
                        lines=(pending+raw).split(b'\n'); pending=lines.pop()
                        for line in lines:
                            event=json.loads(line); name=event['event']
                            if name=='world_restored':
                                if event['task']['status']!='paused': raise AssertionError('restored task was not paused')
                                if event['task']['progress']!=expected: raise AssertionError('progress changed across restart')
                            if name=='session_ready' and stage=='ready':
                                if args.semantic_fixture:
                                    from semantic_fixture import DialogueFixture
                                    fixture=DialogueFixture(root,output,base/'fixture-memory.json',restore=index==1)
                                    fixture.prepare(); resume_at=time.monotonic()+1.; stage='paused'
                                elif index==0:
                                    write_command(output/'commands',dict(command_id='request',task_id='gather',op='request',goal='dirt',count=64))
                                    stage='collect'
                                else: resume_at=time.monotonic()+1.; stage='paused'
                            if name=='policy_dispatch_started' and stage=='paused':
                                raise AssertionError('policy acted before resume')
                            if name=='inventory_observed' and stage=='collect' and event['inventory'].get('dirt',0)>=expected+2:
                                if fixture: fixture.pause()
                                else: write_command(output/'commands',dict(command_id='pause',task_id='gather',op='pause',expected_revision=1))
                                stage='stopping'
                            if name=='command_acknowledged' and event['op']=='pause': stop.touch(); return
                except BaseException as exc:
                    errors.append(repr(exc))
                    if output.exists(): stop.touch()
            watcher=threading.Thread(target=watch); watcher.start()
            options=SimpleNamespace(root=root,policy='steve1',task_session=True,world_slot=slot,
                                    seed=130363,limit_s=180.,output=output,stop_file=stop)
            try: report=episode(options,condition_for(root,'steve1','dirt')); reports.append(report)
            finally: finished.set(); watcher.join(timeout=3)
            print(json.dumps({'phase':index,'summary':report},ensure_ascii=False),flush=True)
            if errors or report.get('world_save',{}).get('status')!='saved': break
            if report['task']['status']!='paused' or report['task']['progress']<expected+2:
                errors.append('collection/pause target not met'); break
            if report.get('process_cleanup',{}).get('remaining_live_pids'):
                errors.append('processes survived shutdown'); break
            expected=report['task']['progress']
        passed=(len(reports)==2 and not errors and reports[-1].get('world_restore',{}).get('inventory_match') is True
                and reports[-1].get('world_save',{}).get('status')=='saved')
        result={'passed':passed,'slot':slot,'reports':reports,'errors':errors,'additional_paid_api_calls':0}
        result['semantic_source']='deterministic_local_fixture_not_llm' if args.semantic_fixture else 'structured_commands'
        write_json(base/'acceptance.json',result)
        return 0 if passed else 2
    finally:
        (root/'results/STOP').touch(exist_ok=True)


if __name__=='__main__': raise SystemExit(main())
