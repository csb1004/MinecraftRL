"""Freeze two archived stall requests for review; no keys or network access."""
import json
from pathlib import Path

from minecraftrl.recovery_draft import payload
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
batch=root/'results/goal-recovery-comparison-20261009'
state=read(batch/'experiment.json'); assert state['completed'] and not state['errors']
stop=read(root/'results/goal-recovery-stop-20261009/work.json')
assert stop['completed'] and not stop['performance_sample']
base=root/'results/recovery-draft-preparation-20261009';base.mkdir(exist_ok=False)
cases=[]
# Fixed first paired fixture; selection is not based on which choice the LLM gives.
for label in ('corner-wood','corner-explore'):
    path=batch/'planner-evidence'/(label+'.json'); evidence=read(path)
    original=evidence['planner_input']; snapshot=original['snapshot']
    packet=dict(request_id='archived-'+label,evidence_id=original['evidence']['event_id'],
        root_goal=dict(goal='wood',target_count=1),root_reference=snapshot['goal_ref'],
        assessment=evidence['stall_assessment'],status='stalled',remaining_control_s=180.-snapshot['elapsed_s'],
        scope='Archived pre-intervention state; no live execution authority or task generation',
        option_binding=dict(explore_then_resume='Existing opt-in STEVE-1 goal conditioning; maximum 200 ticks, one window; not a guaranteed detour',
                            remain_paused='No motor action; root goal remains preserved'))
    request=payload(packet)  # Does not contain future outcomes, labels or another run's results.
    write_json(base/(label+'-input.json'),packet);write_json(base/(label+'-payload.json'),request)
    cases.append(dict(label=label,input=str(base/(label+'-input.json')),sha256=digest(base/(label+'-input.json')),
        payload_sha256=digest(base/(label+'-payload.json')),source_evidence_sha256=digest(path)))
manifest=dict(prepared=True,actual_api_calls=0,paid_approval_received=False,maximum_calls=2,retries=0,
    model='gpt-5.3-codex',model_basis='Same identifier as the previously approved project API probes',
    max_output_tokens_per_call=1024,timeout_s=20,cases=cases,
    execution_authorized=False,scope='Text-only archived proposal quality/contract smoke test; not live Minecraft recovery',
    sent_data='Only two recorded pre-intervention game stagnation summaries and fixed option contracts; no private dialogue or secrets',
    source_sha256={str(p):digest(p) for p in (Path(__file__),root/'scripts/run_recovery_draft_probe.py',
        root/'src/minecraftrl/recovery_draft.py',root/'src/minecraftrl/semantic.py',
        root/'src/minecraftrl/goal_recovery.py',root/'docs/research-spec.md')})
write_json(base/'manifest.json',manifest);print(base)
