"""Read-only real recovery cases for reviewing a future planner contract; no LLM."""
import hashlib,json
from pathlib import Path
from minecraftrl.runtime import write_json

root=Path(__file__).resolve().parents[1]
old=json.loads((root/'results/recovery-curriculum-20261007/experiment.json').read_text())
new=json.loads((root/'results/recovery-gap-20261007/experiment.json').read_text())
records=old['runs'][:2]+new['runs'][:1];cases=[]
for record in records:
    run=Path(record['path']);summary=record['report'];chosen=None
    for line in (run/'events.jsonl').open():
        event=json.loads(line)
        if event['event']!='transition':continue
        if event['info']['rl_reward']['recovery']['event']=='entered' or event['info']['rl_reward']['stage']=='complete':
            chosen=(event,hashlib.sha256(line.encode()).hexdigest());break
    if chosen is None:continue
    event,sha=chosen;server=event['info']['workshop']
    cases.append(dict(source_kind='offline_RL_episode_not_task_session',source=str(run.relative_to(root)),
        event_sha256=sha,step=event['step'],world_sha256=record['world']['sha256'],
        current_task=None,task_revision=None,goal_from_experiment=summary['goal'],
        observations={**{k:server[k] for k in ('server_tick','server_health','server_alive',
            'server_x','server_y','server_z','wood_v3_kind','wood_v3_distance','wood_v3_reachable')},
            'server_inventory':{k:v for k,v in server['server_inventory'].items() if v}},
        inventory_encoding='nonzero counts; omitted registered items have zero count in this snapshot',
        observation_access='privileged_server',reward_stage=event['info']['rl_reward']['stage'],
        recovery=event['info']['rl_reward']['recovery'],
        review_question='What supported subgoal could make progress without replacing the accepted goal?' if event['info']['rl_reward']['stage']!='complete'
            else 'Completion is already observed; a later planner response must not dispatch another action.',
        execution_authorized=False))
write_json(root/'results/high-level-planner-cases-20261007.json',dict(
    schema=1,scope='offline contract examples only; no task creation, generated plan, or LLM invocation',
    llm_calls=0,environment_actions=0,cases=cases))
print(json.dumps(dict(cases=len(cases),llm_calls=0,environment_actions=0)))
