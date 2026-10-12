"""One actual cancellation trial after intervention; excluded from performance."""
import json
import subprocess
import sys
import time
from pathlib import Path

from run_rl_batch import ROOT, rearm_allowed
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
state = read(ROOT/'results/goal-recovery-comparison-20261009/experiment.json')
assert state['completed'] and not state['errors']
assert any(r['audited'] and r['report']['intervention_committed'] and r['condition']=='explore' for r in state['runs'])
rearm_allowed()
base = ROOT/'results/goal-recovery-stop-20261009'; base.mkdir(exist_ok=False)
stop = ROOT/'results/STOP'
marker = 'RL collection ended; owned goal-recovery STOP audit 20261009 request.'
checkpoint = ROOT/'results/onset-coverage-training-20261009/checkpoint.pt'
world = state['worlds']['corner']
argv = [sys.executable,str(ROOT/'scripts/run_goal_recovery_pilot.py'),'--seed',str(world['seed']),
    '--sampling-seed','9721','--checkpoint',str(checkpoint),
    '--embed',str(ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'),
    '--recovery-embed',str(ROOT/'results/goal-conditioning-diagnosis-20261009/explore_9711.npy'),
    '--world',world['path']]
result = dict(completed=False,performance_sample=False,policy_sampling_seed=9721,requested=False,
    error=None,checkpoint_sha256=digest(checkpoint),world_sha256=digest(world['path']),
    sources={str(p):digest(p) for p in (Path(__file__),ROOT/'src/minecraftrl/goal_recovery.py',ROOT/'src/minecraftrl/cli.py')})
logpath = base/'launch.log'; process = None; run = None
try:
    with logpath.open('w',encoding='utf-8') as log:
        process = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic()+900
        offset = 0; pending = b''
        while process.poll() is None:
            if time.monotonic() >= deadline: raise TimeoutError('STOP probe launcher timeout')
            if run is None:
                with logpath.open(encoding='utf-8') as source: first = source.readline().strip()
                if first:
                    run = Path(first)/'run'; result['run'] = str(run); write_json(base/'work.json',result)
                    print(str(run),flush=True)
            if run and (run/'events.jsonl').exists() and not result['requested']:
                with (run/'events.jsonl').open('rb') as source:
                    source.seek(offset); block = source.read(2**21); offset += len(block)
                lines = (pending+block).split(b'\n'); pending = lines.pop()
                for line in lines:
                    event = json.loads(line)
                    if event['event']=='policy_dispatch_finished' and event.get('controller')=='steve1_intervention':
                        # Exclusive creation preserves any concurrent/manual STOP.
                        with stop.open('x',encoding='utf-8') as destination: destination.write(marker)
                        result.update(requested=True,request_after_dispatch_step=event['step'],
                                      witnessed_dispatch_finished_s=event['monotonic_s'])
                        write_json(base/'work.json',result); print('STOP requested after intervention dispatch',flush=True)
                        break
            time.sleep(.05)
    assert result['requested'], 'No committed intervention available for cancellation trial'
    report = read(run/'summary.json'); release = read(run/'input-release-state.json')
    assert report['status']=='cancelled' and not report['execution_valid'] and not report.get('rl_training_eligible',False)
    assert report['release']=='neutral_command_sent' and release['verified'] and release['held_key_count']==0
    assert not report['process_cleanup']['remaining_live_pids']
    stopped = False; last_step = 0; intervention_steps = 0
    for line in (run/'events.jsonl').open(encoding='utf-8'):
        e = json.loads(line)
        if e['event']=='stop_acknowledged': stopped=True
        if e['event'] in ('inference_started','policy_dispatch_started'): assert not stopped
        if e['event']=='transition':
            assert not stopped; last_step=e['step']; intervention_steps += e['controller']=='steve1_intervention'
    assert stopped and intervention_steps>=1 and last_step>=result['request_after_dispatch_step']
    assert stop.read_text(encoding='utf-8')==marker, 'A new manual STOP is retained'
    result.update(completed=True,status=report['status'],committed_steps=last_step,
        intervention_steps=intervention_steps,post_acknowledgment_policy_dispatches=0,
        post_acknowledgment_inferences=0,held_keys_after_release=0,remaining_owned_pids=[],
        worker_exit_code=report['worker_exit_code'],stop_retained=True,
        scope='Actual mid-intervention cancellation; no performance outcome or policy improvement claim',
        evidence_sha256={name:digest(run/name) for name in ('summary.json','events.jsonl','input-release-state.json')})
except BaseException as exc:
    result['error']=repr(exc)
finally:
    if process is not None and process.poll() is None:
        try:
            with stop.open('x',encoding='utf-8') as destination: destination.write(marker)
        except FileExistsError: pass
        try: process.wait(timeout=90)
        except subprocess.TimeoutExpired: result['cleanup_timeout']=True
    write_json(base/'work.json',result)
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['completed'] else 2)
