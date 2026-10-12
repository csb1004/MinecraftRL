"""Read-only accounting and server-route analysis of the authorized live pilot."""
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np

from minecraftrl.recovery_evidence import assess_attempt
from minecraftrl.recovery_draft import validate_draft
from minecraftrl.recovery_live import checked_reply,LIVE_SCOPE
from minecraftrl.runtime import write_json
from minecraftrl.semantic import response_text
from minecraftrl.stall_replay import log_count,wood_sample
from minecraftrl.world_store import digest


def main():
    root=Path(__file__).resolve().parents[1]
    base=root/'results/planning-recovery-live-20261009';exchange=base/'exchange'
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    state=read(base/'work.json');assert state['completed'] and state['mode']=='live_api'
    run=Path(state['run']);audit=read(run/'planning-recovery-audit.json');assert audit['verified']
    authorization=read(root/'results/planning-recovery-live-authorization-20261009.json')
    assert authorization['maximum_api_calls']==state['maximum_api_calls']==1
    assert state['preparation']['sha256']==authorization['prepared_manifest_sha256']
    for relative,expected in state['sources'].items():assert digest(root/relative)==expected,relative
    summary=read(run/'summary.json');initial=read(run/'initial-state.json');world=initial['initial_world']
    rows=read(run/'goal-recovery-rows.json');switches=read(run/'goal-switches.json')
    receipt=read(exchange/'api-receipt.json') if (exchange/'api-receipt.json').exists() else None
    calls=receipt['network_calls'] if receipt else 0
    assert calls==state['actual_api_calls']==audit['actual_llm_calls'] and calls<=1
    if receipt:
        assert receipt['model']=='gpt-5.3-codex' and receipt['retries']==0
        assert receipt['max_output_tokens']==1024 and receipt['timeout_s']==20 and not receipt['key_recorded']
        assert receipt['request_sha256']==digest(exchange/'request.json')
    broker=read(exchange/'broker-result.json') if (exchange/'broker-result.json').exists() else None
    proposal=broker.get('proposal') if broker else None
    if proposal:
        checked_reply(proposal,read(exchange/'request.json')['packet'],live=True)
        assert proposal['execution_authorized'] is False
    response=read(exchange/'api-response.json') if (exchange/'api-response.json').exists() else None
    if response:
        assert digest(exchange/'api-response.json')==receipt['response_sha256']
        assert response['usage']==receipt['usage']
        if proposal:
            expected=validate_draft(json.loads(response_text(response)),read(exchange/'request.json')['packet'])
            expected['scope']=LIVE_SCOPE
            assert expected==proposal  # Actual model output must equal the locally reviewed proposal.
    output=base/'analysis';output.mkdir(exist_ok=False)
    def position(obs):
        x=obs['server_x']-world['chunk'][0]*16;z=obs['server_z']-world['chunk'][1]*16
        return [x,16-z if world['curriculum']['mirror_z'] else z]
    points=np.array([position(initial['server'])]+[position(r['server_after']) for r in rows])
    back=next((s for s in switches if s['mode']=='wood'),None)
    first_interaction=None;phases={};colors=[]
    for step,row in enumerate(rows,1):
        if row.get('context_only'):phase='neutral'
        elif row['mode']=='intervention':phase='explore'
        elif back and step>=back['before_action']:phase='wood_after'
        else:phase='wood_before'
        sample=wood_sample(row['server_after'],row['action'],str(run))
        stats=phases.setdefault(phase,dict(steps=0,attack=0,movement=0,yaw=0,raw_stages=Counter()))
        stats['steps']+=1;stats['attack']+=int('attack' in sample.action)
        stats['movement']+=int(any(k in sample.action for k in ('forward','back','left','right')))
        stats['yaw']+=int(any(k.startswith('yaw') for k in sample.action));stats['raw_stages'][sample.stage]+=1
        if first_interaction is None and sample.stage in ('mine','collect'):first_interaction=step
        colors.append('#777777' if phase=='neutral' else '#d45c25' if phase=='explore' else '#306ea6')
    intervention=[r['server_after'] for r in rows if r['mode']=='intervention']
    attempt=dict(classification='not_executed')
    if intervention:
        entry=next(s for s in switches if s['mode']=='intervention')
        reason=back['reason'] if back else ('goal_completed' if audit['goal_completed'] else 'control_budget')
        if reason in ('tick_budget','wood_interaction_ready','control_budget','goal_completed'):
            attempt=assess_attempt(entry['server_before'],intervention,reason=reason,
                                  baseline_count=log_count(initial['server']),budget_ticks=200)
        else:attempt=dict(classification='other_boundary',reason=reason,committed_ticks=len(intervention))
    stalls=[dict(step=r['step'],reason=r['reason']) for r in read(run/'stall-decisions.json') if r['event']=='stalled']
    events=read(run/'planner-events.json')
    timeline=[{k:v for k,v in e.items() if k in ('event','step','elapsed_s','reason','wait_s')} for e in events]
    gain=log_count(rows[-1]['server_after'])-log_count(initial['server'])
    assert gain==audit['log_gain'] and len(rows)==audit['steps']
    report=dict(completed=True,scope='Live LLM-to-neural-policy integration; privileged server evidence; not a causal comparison',
        actual_api_calls=calls,remaining_authorized_calls=1-calls,api_retries=0,optimizer_updates=0,
        status=summary['status'],execution_valid=summary['execution_valid'],steps=len(rows),log_gain=gain,
        proposal=proposal,receipt=receipt,broker_status=broker['status'] if broker else None,
        timeline=timeline,switches=[dict(step=s['before_action'],mode=s['mode'],reason=s['reason']) for s in switches],
        first_wall_crossing=next((i for i,p in enumerate(points) if p[0]>=8.3),None),
        first_wood_interaction=first_interaction,stalls=stalls,recovery=attempt,phases=phases,
        final_position=points[-1].tolist(),control_s=summary['elapsed_control_s'],audit=audit,
        preparation=state['preparation'],game_launcher_exit=state['game_launcher_exit'],broker_exit=state['broker_exit'],
        new_analysis_game_actions=0,new_analysis_api_calls=0,
        source_sha256={str(p.relative_to(root)):digest(p) for p in (
            base/'work.json',run/'summary.json',run/'events.jsonl',run/'planning-recovery-audit.json',
            run/'goal-recovery-rows.json',Path(__file__))})
    fig,ax=plt.subplots(figsize=(7,6))
    for x,z,w,h in ((0,0,16,1),(0,15,16,1),(0,0,1,16),(15,0,1,16),(7,1,1,world['curriculum']['wall_end'])):
        ax.add_patch(Rectangle((x,z),w,h,color='#b5bbc4'))
    ax.add_patch(Rectangle((11,3),1,1,color='#9b663f'))
    ax.add_collection(LineCollection(np.stack((points[:-1],points[1:]),axis=1),colors=colors,linewidths=1.8))
    markers=[(0,'o','start','#111111'),(len(rows),'x','end','#b51b1b')]
    if summary['trigger']:markers.append((summary['trigger']['step'],'*','stall','#9f32ad'))
    if back:markers.append((back['before_action']-1,'s','return wood','#17805b'))
    for step,marker,label,color in markers:ax.plot(*points[step],marker,color=color,label=label,markersize=9)
    ax.set(xlim=(0,16),ylim=(0,16),xlabel='x (blocks)',ylabel='canonical z',
           title=f"Live LLM pilot: wood +{gain}, {len(rows)} transitions\nBlue = wood; orange = exploration; no causal claim")
    ax.set_aspect('equal');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(output/'route.png',dpi=170);plt.close(fig)
    report['route_sha256']=digest(output/'route.png');write_json(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('audit','source_sha256','receipt','proposal')}))


if __name__=='__main__':main()
