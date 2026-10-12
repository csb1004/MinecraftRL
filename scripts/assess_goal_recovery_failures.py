"""Project audited games into advisory failure evidence and existing planner guards."""
import json
from pathlib import Path

from minecraftrl.recovery_evidence import assess_stall, assess_attempt
from minecraftrl.replanning import PlanHistory, ReplanGuard, collection_capability, make_snapshot
from minecraftrl.runtime import write_json
from minecraftrl.stall_replay import log_count
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root/'results/goal-recovery-comparison-20261009'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
state = read(base/'experiment.json')
output = base/'planner-evidence'; output.mkdir(exist_ok=True)
for trial in state['runs']:
    if not trial['audited']: continue
    destination = output/(trial['label']+'.json')
    if destination.exists(): continue  # Never replay a completed sidecar transaction.
    run = Path(trial['path']); report = read(run/'summary.json')
    audit = read(run/'goal-recovery-audit.json'); assert audit['verified']
    initial = read(run/'initial-state.json'); baseline = log_count(initial['server'])
    trigger = report['trigger']; intervention = []; trigger_obs = None; started = elapsed = None
    for line in (run/'events.jsonl').open(encoding='utf-8'):
        e = json.loads(line)
        if e['event'] == 'episode_started': started = e['monotonic_s']
        if e['event'] != 'transition': continue
        if trigger and e['step'] == trigger['step']:
            trigger_obs = e['info']['workshop']; elapsed = e['monotonic_s']-started
        if e['controller'] == 'steve1_intervention': intervention.append(e['info']['workshop'])
    result = dict(label=trial['label'],scope='Read-only actual-game evidence projection; no live planning or dispatch',
        actual_llm_calls=0,new_game_actions=0,root_goal='collect wood',target_count=1,rm_effect='preserve',
        episode_status=report['status'],actual_log_gain=audit['log_gain'],goal_completed=audit['server_success'],
        source_sha256={name:digest(run/name) for name in ('summary.json','events.jsonl','goal-recovery-audit.json')},
        implementation_sha256={str(p.relative_to(root)):digest(p) for p in
            (Path(__file__),root/'src/minecraftrl/recovery_evidence.py',root/'src/minecraftrl/replanning.py')})
    if trigger is not None:
        assert trigger_obs is not None
        result['stall_assessment'] = assess_stall(trigger_obs,trigger,baseline_count=baseline)
        assert result['stall_assessment']['replan_candidate']
        snapshot = make_snapshot(trigger_obs,world_id=initial['initial_world']['sha256'],
            episode_id=run.parent.name,goal='wood',task=None,generation=None,sequence=trigger['step'],
            elapsed_s=elapsed,status='running',baseline_count=baseline)
        evidence = dict(trigger,event_id=run.parent.name+':stall-'+str(trigger['step']))
        history = PlanHistory(output/(trial['label']+'-history.json'))
        guard = ReplanGuard(history,{'collect_wood':collection_capability('wood',audit['checkpoint_sha256'])})
        guard.record_failure(snapshot,evidence,'collect_wood')
        request = guard.request(snapshot,evidence)
        result['planner_input'] = guard.planner_input(request['request_id'])
        assert not result['planner_input']['call_model']  # No invented detour capability.
        result['live_execution_gaps'] = ['no_runtime_generation_binding','no_neutral_release_at_stall',
                                        'no_validated_alternative_skill']
    if intervention:
        entry = next(s for s in report['switches'] if s['mode']=='intervention' and s['committed'])
        back = next((s for s in report['switches'] if s['mode']=='wood'),None)
        if back:
            reason = back['reason']
        elif report['server_target_reached']: reason = 'goal_completed'
        elif report['status'] == 'environment_done': reason = 'environment_done'
        elif report['status'] == 'time_limit': reason = 'control_budget'
        else: reason = 'stop'
        result['attempt_assessment'] = assess_attempt(entry['server_before'],intervention,
            reason=reason,baseline_count=baseline,budget_ticks=200)
    else:
        result['attempt_assessment'] = dict(classification='not_executed',counts_as_recovery_failure=False)
    write_json(destination,result)
    print(json.dumps(dict(label=trial['label'],stall=result.get('stall_assessment',{}).get('classification'),
        recovery=result['attempt_assessment']['classification'],wood=audit['log_gain'])))
