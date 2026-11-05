"""One actual-game STOP diagnostic; excluded from performance samples.

Run inside the isolated WSL STEVE-1 environment after the performance batch.
The probe leaves the project STOP latched and never rearms it automatically.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch', default='rm-conditioning-20261031')
    parser.add_argument('--mode', choices=('staged', 'purpose-switch', 'logo-staged', 'goal-queue'), default='staged')
    parser.add_argument('--program',type=Path)
    parser.add_argument('--s1-record',action='store_true')
    parser.add_argument('--s2-variant',choices=('linear-off','linear-on','nonlinear-off','nonlinear-on'))
    parser.add_argument('--s2-checkpoint',type=Path)
    parser.add_argument('--recovery-variant',choices=('nonlinear-off','nonlinear-on'))
    parser.add_argument('--recovery-checkpoint',type=Path)
    parser.add_argument('--recovery-world',type=Path)
    parser.add_argument('--effect-learning',action='store_true')
    parser.add_argument('--execution-learning',action='store_true')
    args = parser.parse_args()
    if (args.recovery_world or args.recovery_checkpoint) and not args.recovery_variant:
        raise ValueError('recovery STOP requires an explicit variant')
    root = args.root.resolve()
    batch = (root/'results'/args.batch).resolve()
    bank = batch/'goal-bank'
    manifest = (bank/'wood' if args.mode=='goal-queue' else bank)/'manifest.json'
    if not batch.is_relative_to(root/'results') or not manifest.exists():
        raise ValueError('existing project-local goal bank required')
    output = batch/'stop-probe'
    stop = root/'results/STOP'
    receipt = batch/'stop-probe-request.json'
    if stop.exists() or output.exists() or receipt.exists():
        raise RuntimeError('STOP latched or probe already attempted; inspect evidence')
    argv = [sys.executable, '-m', 'minecraftrl.cli', 'episode', '--root', str(root),
            '--policy', 'steve1', '--condition', 'wood', '--seed', '81031999',
            '--limit-s', '180', '--rm-conditioning', args.mode,
            '--rm-goal-bank', str(batch/'goal-bank'), '--output', str(output),
            '--stop-file', str(stop)]
    if args.program is not None:argv.extend(['--rm-program',str(args.program)])
    if args.effect_learning:argv.append('--effect-learning')
    if args.execution_learning:argv.append('--execution-learning')
    if args.s1_record:argv.append('--s1-record')
    if args.s2_variant:argv.extend(['--s2-variant',args.s2_variant])
    if args.s2_checkpoint:argv.extend(['--s2-checkpoint',str(args.s2_checkpoint)])
    if args.recovery_variant:
        if args.recovery_world is None or args.s2_variant or args.s1_record:
            raise ValueError('isolated saved-start recovery STOP required')
        world=json.loads(args.recovery_world.with_suffix('.json').read_text(encoding='utf-8-sig'))
        argv[argv.index('--seed')+1]=str(world['seed'])
        argv.extend(['--recovery-variant',args.recovery_variant,'--recovery-world',str(args.recovery_world),
            '--recovery-task','collection','--recovery-split',world['experimental_split']])
        if args.recovery_checkpoint:argv.extend(['--recovery-checkpoint',str(args.recovery_checkpoint)])
    request = dict(schema=1, argv=argv, trigger='after 10 completed policy transitions',
                   performance_sample=False, requested=False)
    write(receipt, request)
    with (batch/'stop-probe-console.log').open('w') as log:
        proc = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                path = output/'events.jsonl'
                if not request['requested'] and path.exists():
                    events = []
                    for line in path.read_text(encoding='utf-8').splitlines():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # Writer may still be flushing its last line.
                    transitions = [e for e in events if e['event']=='transition']
                    if len(transitions) >= 10:
                        request.update(requested=True, observed_steps=len(transitions),
                                       request_begin_s=time.monotonic())
                        # Exclusive create cannot overwrite a concurrent user STOP.
                        try:
                            with stop.open('x', encoding='utf-8') as handle:
                                handle.write('Actual-game STOP probe; do not resume automatically.\n')
                                handle.flush()
                                os.fsync(handle.fileno())
                            request['created_stop']=True
                        except FileExistsError:
                            request['created_stop']=False
                        request['request_visible_s']=time.monotonic()
                        write(receipt, request)
                time.sleep(.02)
        except KeyboardInterrupt:
            stop.touch(exist_ok=True)
            proc.wait(timeout=35)
            raise
    request.update(supervisor_exit_code=proc.returncode, finished_s=time.monotonic())
    write(receipt, request)
    if not request['requested']:
        raise RuntimeError('Run ended before trigger: STOP contract not exercised')
    events = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    ack = next((i for i,e in enumerate(events) if e['event']=='stop_acknowledged'), None)
    summary = json.loads((output/'summary.json').read_text())
    release = json.loads((output/'input-release-state.json').read_text())
    after_ack = None if ack is None else sum(e['event']=='policy_dispatch_started' for e in events[ack+1:])
    after_request = sum(e['event']=='policy_dispatch_started' and
                        e['monotonic_s']>=request['request_visible_s'] for e in events)
    result = dict(performance_sample=False, status=summary['status'], steps=summary.get('steps'),
                  worker_exit_code=summary.get('worker_exit_code'),
                  dispatches_after_ack=after_ack, dispatches_after_request_visible=after_request,
                  acknowledgment_delay_s=None if ack is None else
                      events[ack]['monotonic_s']-request['request_visible_s'],
                  acknowledgment_latency_upper_bound_s=None if ack is None else
                      events[ack]['monotonic_s']-request['request_begin_s'],
                  release=summary.get('release'), held_key_count=release.get('held_key_count'),
                  release_verified=release.get('verified'),
                  remaining_live_pids=summary.get('process_cleanup',{}).get('remaining_live_pids'),
                  stop_file_remains=stop.exists(), optimizer_updates=0)
    result['passed'] = (request.get('created_stop') is True and result['status']=='cancelled' and
                        after_ack==0 and after_request==0 and result['worker_exit_code']==2 and
                        result['release']=='neutral_command_sent' and
                        result['release_verified'] is True and result['held_key_count']==0 and
                        result['remaining_live_pids']==[] and result['stop_file_remains'])
    write(batch/'stop-probe-audit.json', result)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('STOP diagnostic did not pass; preserve raw evidence')


if __name__ == '__main__':
    main()
