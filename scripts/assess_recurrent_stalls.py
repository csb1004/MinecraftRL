"""All real alarm times, including re-stalls after return; no new game actions."""
import json
from pathlib import Path

from minecraftrl.recovery_evidence import assess_stall
from minecraftrl.runtime import write_json
from minecraftrl.stall_replay import log_count
from minecraftrl.world_store import digest


root=Path(__file__).resolve().parents[1];base=root/'results/goal-recovery-comparison-20261009'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
state=read(base/'experiment.json');assert state['completed'] and not state['errors']
out=base/'all-stall-assessments.json'
if out.exists():raise FileExistsError(out)
rows=[]
for trial in state['runs']:
    run=Path(trial['path']);audit=read(run/'goal-recovery-audit.json');assert audit['verified']
    decisions=read(run/'stall-decisions.json')
    alarms={d['step']:d for d in decisions if d['event']=='stalled'}
    baseline=log_count(read(run/'initial-state.json')['server'])
    returned=next((s['before_action'] for s in trial['report']['switches'] if s['mode']=='wood' and s['committed']),None)
    cases=[];started=None
    if alarms:
        for line in (run/'events.jsonl').open(encoding='utf-8'):
            event=json.loads(line)
            if event['event']=='episode_started':started=event['monotonic_s']
            if event['event']!='transition' or event['step'] not in alarms:continue
            elapsed=event['monotonic_s']-started
            cases.append(dict(step=event['step'],elapsed_control_s=elapsed,remaining_control_s=max(0.,180.-elapsed),
                after_return_to_root=returned is not None and event['step']>=returned,
                assessment=assess_stall(event['info']['workshop'],alarms[event['step']],baseline_count=baseline)))
    assert len(cases)==len(alarms) and all(c['assessment']['replan_candidate'] for c in cases)
    rows.append(dict(label=trial['label'],alarms=cases,event_source_sha256=audit['source_sha256']['events.jsonl']))
    print(json.dumps(dict(label=trial['label'],alarms=[dict(step=c['step'],remaining_s=round(c['remaining_control_s'],2),
        after_return=c['after_return_to_root'],reason=c['assessment']['evidence']['reason']) for c in cases])),flush=True)
write_json(out,dict(verified=True,rows=rows,alarm_count=sum(len(r['alarms']) for r in rows),new_game_actions=0,
    actual_llm_calls=0,optimizer_updates=0,source_sha256=digest(__file__),
    scope='Advisory actual alarm evidence, not automatic retry authorization; terminal success is not a false-alarm label after intervention'))
