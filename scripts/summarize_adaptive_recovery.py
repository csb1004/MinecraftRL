"""Post-game-only metrics. Fixture geometry is never input to a live candidate planner."""
import json,math
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from summarize_blind_recovery import local_escape,point


def main():
    root=Path(__file__).resolve().parents[1];base=root/'results/adaptive-recovery-20261014'
    out=root/'docs/reports/2026-10-14';out.mkdir(parents=True,exist_ok=True)
    if (out/'recovery-summary.json').exists():raise FileExistsError('summary exists')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'));trials={};raw={};sources={}
    index=read(base/'completed-trials.json')
    planners=('visit_memory','adaptive_rule','adaptive_llm')
    for case in ('far','corner'):
        for planner in planners:
            key=case+'/'+planner;run=base/index[key];s=read(run/'summary.json');audit=read(run/'waypoint-audit.json');assert audit['verified']
            rows=read(run/'waypoint-rows.json');raw[key]=rows;switches=read(run/'waypoint-switches.json');route=read(run/'exploration-route.json')
            start=switches[0]['sequence'] if switches else None;nav=[r for r in rows if r['mode']=='waypoint'];escape=crossing=None
            if start is not None and nav:
                positions=[point(r['server_after']) for r in rows];origin=point(rows[start]['server_before']);escape=local_escape(positions[start:],origin)
                chunk_x=16*s['initial_world']['chunk'][0];left=point(rows[0]['server_before'])[0]-chunk_x<7;edge=chunk_x+(8.3 if left else 6.7)
                other=lambda p:p[0]>=edge if left else p[0]<=edge
                crossed=[i+1 for i,p in enumerate(positions) if other(p)];after=[i for i in crossed if i>start];already=other(origin)
                crossing=dict(success=None if already else bool(after),already_at_stall=already,
                    first_action=after[0] if after else None,first_episode_crossing=crossed[0] if crossed else None)
            receipts=[read(p) for p in (run/'candidate-exchange').glob('request-*/api-receipt.json')]
            decisions=route['decisions'];blocked=route.get('blocked_directions',[])
            trials[key]=dict(steps=s['steps'],mode_steps=s['mode_steps'],status=s['status'],recovery_evaluated=bool(nav),trigger_action=start,
                local_escape=escape,wall_crossing=crossing,root_success=s['root_inventory_success'],
                inventory_delta={k:v for k,v in s['inventory_final_delta'].items() if v},actual_goal_arrivals=len(s['route_receipts']),
                route_failure=s['route_failure'],candidate_failures=route['failures'],blocked_directions=blocked,probes=route.get('probes',[]),
                candidate_decisions=decisions,goals=audit['goals'],path_blocks=audit['path_blocks'],
                elapsed_control_s=s['elapsed_control_s'],inference_ms=s['inference_ms'],control_cycle_ms=s['control_cycle_ms'],
                gpu=s['gpu'],device_vram=s['device_vram'],actual_llm_calls=s['actual_llm_calls'],api_receipts=receipts,
                stop_dispatches=audit['stop_dispatches'],keys_remaining=audit['keys_remaining'],owned_pids_remaining=audit['owned_pids_remaining'],
                action_logp_max_error=audit['action_logp_max_error'])
            for name in ('summary.json','waypoint-audit.json','waypoint-rows.json','events.jsonl','exploration-route.json'):
                sources[(run/name).relative_to(root).as_posix()]=digest(run/name)
    prefixes={}
    from PIL import Image
    for case in ('far','corner'):
        for planner in planners[1:]:
            keys=[case+'/visit_memory',case+'/'+planner];limit=min(trials[k]['mode_steps']['root'] for k in keys)
            first=dict(frame=None,action=None,pose=None)
            for i in range(limit):
                a,b=[raw[k][i] for k in keys];frame_diff=False
                if first['frame'] is None:
                    with Image.open(base/index[keys[0]]/a['frame']) as x,Image.open(base/index[keys[1]]/b['frame']) as y:frame_diff=x.tobytes()!=y.tobytes()
                different=dict(frame=frame_diff,action=a['action']!=b['action'],pose=any(a['server_before'][k]!=b['server_before'][k] for k in ('server_x','server_y','server_z','yaw','pitch')))
                for name,diff in different.items():
                    if diff and first[name] is None:first[name]=i+1
            prefixes[case+'/'+planner]=dict(compared_actions=limit,first_difference=first,identical=not any(first.values()))
    api=[v for trial in trials.values() for v in trial['api_receipts']];calls=sum(v['network_calls'] for v in api)
    ledger=read(base/'paid-budget.json') if (base/'paid-budget.json').exists() else dict(calls=[],reserved_micro_usd=0)
    assert calls==len(ledger['calls'])<=4 and ledger['reserved_micro_usd']<=1000000
    result=dict(trials=trials,root_prefix_comparison=prefixes,source_sha256=sources,api_calls=calls,
        reserved_usd=ledger['reserved_micro_usd']/1e6,estimated_usd=sum(v.get('estimated_usd') or 0 for v in api),
        unknown_usage_calls=sum(v['network_calls'] and not v.get('usage') for v in api),optimizer_updates=0,final_evaluation_seeds_used=0,
        scope='Six development trials; privileged pose/root RM, map-free candidates, learned motors. Limited LLM/rule hybrid; not F3-only.')
    result['runs']=index
    write_json(out/'recovery-summary.json',result);plot(base,out,trials,raw,read,index)
    print(json.dumps(dict(trials={k:{n:v[n] for n in ('steps','mode_steps','status','root_success','local_escape','wall_crossing','actual_llm_calls')} for k,v in trials.items()},
        api_calls=calls,reserved_usd=result['reserved_usd'],estimated_usd=result['estimated_usd'],prefixes=prefixes)))


def plot(base,out,trials,raw,read,index):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,3,figsize=(15,12));colors=dict(root='#4386b7',waypoint='#db792d',root_probe='#845caf',resumed='#208357',planner_neutral='#222222')
    for row,case in enumerate(('far','corner')):
        for col,planner in enumerate(('visit_memory','adaptive_rule','adaptive_llm')):
            key=case+'/'+planner;s=read(base/index[key]/'summary.json');world=s['initial_world'];rows=raw[key];r=trials[key];ax=axes[row,col]
            def canonical(p):
                x=p[0]-world['chunk'][0]*16;z=p[2]-world['chunk'][1]*16
                return [x,16-z if world['curriculum']['mirror_z'] else z]
            points=np.asarray([canonical(point(rows[0]['server_before']))]+[canonical(point(v['server_after'])) for v in rows])
            for x,z,w,h in ((0,0,16,1),(0,15,16,1),(0,0,1,16),(15,0,1,16),(7,1,1,world['curriculum']['wall_end'])):ax.add_patch(Rectangle((x,z),w,h,color='#bec5cd'))
            ax.add_patch(Rectangle((11,3),1,1,color='#895d35'))
            ax.add_collection(LineCollection(np.stack([points[:-1],points[1:]],axis=1),colors=[colors[v['mode']] for v in rows],linewidths=1.5))
            for i,g in enumerate(r['candidate_decisions'],1):
                x,z=canonical(g['candidates'][g['selected']]['target']);ax.plot(x,z,'+',color='#784eaa');ax.annotate(str(i),(x,z),fontsize=8)
            ax.plot(*points[0],'o',color='#222');ax.plot(*points[-1],'x',color='#bf2832')
            if r['trigger_action'] is not None:ax.plot(*points[r['trigger_action']],'*',color='#f1ba24',markersize=12)
            escape=r['local_escape']['success'] if r['local_escape'] else 'uncalled';cross=r['wall_crossing']
            cross='uncalled' if cross is None else ('already before stall' if cross['already_at_stall'] else cross['success'])
            ax.set_title(f'{case}: {planner}\nescape {escape} | wall {cross}\nwood {r["root_success"]} | API {r["actual_llm_calls"]}',fontsize=10)
            ax.set(xlim=(0,19),ylim=(0,16),xlabel='canonical x (blocks)',ylabel='canonical z (blocks)');ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('Actual Minecraft development trials\nBlue: root | Orange: learned navigation | Purple: bounded root probe | Green: resumed root\nWalls shown only by offline evaluator; no terrain supplied to planner',fontsize=12)
    fig.subplots_adjust(left=.06,right=.98,bottom=.06,top=.86,hspace=.42,wspace=.24)
    fig.savefig(out/'actual-routes.png',dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
