"""Actual first-discovery versus stale-lock boundary; no new gameplay."""
import json
from pathlib import Path

from minecraftrl.planning_recovery import PlanningRecoveryController
from minecraftrl.runtime import write_json
from minecraftrl.stall_replay import log_count
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1]
out=root/'results/recovery-outcome-fix-20261009/fresh-target-replay';out.mkdir(exist_ok=False)
run=root/'results/goal-recovery-eval-104761-20261009T144548946737Z/run'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rows=read(run/'goal-recovery-rows.json')
request=read(run/'planner-history.json')['requests'][0]
first=next(i for i,r in enumerate(rows,1) if r['mode']=='intervention' and
    r['server_before']['wood_v3_kind']==0 and r['server_after']['wood_v3_kind']==1)
assert first==922
receipts=[]
for end,reason,phase in ((first,'wood_target_acquired','resumed'),(1002,'tick_budget','paused')):
    p=PlanningRecoveryController.__new__(PlanningRecoveryController)
    p.output=out/str(end);p.output.mkdir()
    p.rows=rows[:end];p.attempt_start=802;p.attempt_before=rows[802]['server_before']
    p.baseline=log_count(rows[0]['server_before']);p.attempts=[];p.planner_events=[];p.generation=0
    p.request=request;p.elapsed=lambda:0.;p.emit=lambda *a,**k:None
    p.finish_recovery(reason)
    assert p.phase==phase and (p.scheduled is not None)==(phase=='resumed')
    receipts.append(p.attempts[0])
report=dict(verified=True,first_discovery_step=first,old_handoff_step=1003,
    corrected_handoff_next_step=first+1,old_extra_exploration_actions=1002-first,receipts=receipts,
    game_actions=0,api_calls=0,optimizer_updates=0,
    source_sha256={name:digest(root/name) for name in ('src/minecraftrl/planning_recovery.py',
        'src/minecraftrl/recovery_evidence.py','src/minecraftrl/WorkshopObservation.java')},
    limitation='Actual observation replay; corrected gameplay and inventory outcome require a new run.')
write_json(out/'report.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('receipts','source_sha256')}))
