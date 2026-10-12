"""Exercise the corrected boundary on real archived observations, without gameplay."""
import argparse
import json
from pathlib import Path

from minecraftrl.planning_recovery import PlanningRecoveryController
from minecraftrl.runtime import write_json
from minecraftrl.stall_replay import log_count
from minecraftrl.world_store import digest


root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output.resolve()
if not out.is_relative_to(root/'results'):raise ValueError('project-local output required')
out.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sources={
    'live_failure':('20261009T123206681466Z','paused'),
    'fixed_later_success':('20261009T104906245333Z','paused'),
    'fixed_later_failure':('20261009T110944517695Z','paused'),
    'observed_interaction':('20261009T074716850590Z','resumed'),
}
results=[]
for label,(stamp,expected) in sources.items():
    run=root/f'results/goal-recovery-eval-104761-{stamp}/run'
    rows=read(run/'goal-recovery-rows.json');summary=read(run/'summary.json')
    switches=read(run/'goal-switches.json')
    intervention=next(s for s in switches if s['mode']=='intervention')
    returned=next(s for s in switches if s['mode']=='wood')
    start,end=intervention['after_step'],returned['after_step']
    controller=PlanningRecoveryController.__new__(PlanningRecoveryController)
    controller.output=out/label;controller.output.mkdir()
    controller.rows=rows[:end];controller.attempt_start=start
    controller.attempt_before=rows[start]['server_before'];controller.baseline=log_count(rows[0]['server_before'])
    controller.attempts=[];controller.planner_events=[];controller.generation=0
    controller.elapsed=lambda:0.;controller.emit=lambda *a,**k:None
    controller.request=dict(request_id='archived-boundary-'+label,
        snapshot=dict(goal_ref=dict(kind='episode',episode_id=str(run),goal='wood')))
    controller.finish_recovery(returned['reason'])
    receipt=controller.attempts[0]
    assert receipt['next_phase']==controller.phase==expected
    assert (controller.scheduled is not None)==(expected=='resumed')
    assert not receipt['assessment']['goal_completed']
    results.append(dict(label=label,run=run.relative_to(root).as_posix(),result=receipt,
        original_final_goal_completed=summary['server_target_reached'],
        observed_transitions_after_boundary=len(rows)-end,
        sources_sha256={name:digest(run/name) for name in ('goal-recovery-rows.json','summary.json','goal-switches.json')}))
report=dict(verified=True,rows=results,game_actions=0,optimizer_updates=0,api_calls=0,
    implementation_sha256={name:digest(root/name) for name in ('src/minecraftrl/planning_recovery.py',
        'src/minecraftrl/recovery_evidence.py','scripts/replay_recovery_outcomes.py')},
    limitation='Boundary replay only. Pausing would also interrupt one later successful archived run; no improvement claim.')
write_json(out/'report.json',report)
print(json.dumps(dict(verified=True,cases=[dict(label=r['label'],next_phase=r['result']['next_phase'],
    classification=r['result']['assessment']['classification'],original_later_success=r['original_final_goal_completed']) for r in results])))
