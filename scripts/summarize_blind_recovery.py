"""Offline-only escape metrics and publication evidence; never feeds a live policy."""
import json, math
from pathlib import Path
from minecraftrl.world_store import digest
from minecraftrl.runtime import write_json


def local_escape(points,origin):
    """Reach 3 blocks, then stay >=2 blocks for 20 subsequent transitions."""
    distances=[math.hypot(p[0]-origin[0],p[2]-origin[2]) for p in points]
    count=0;start=None;maximum=max(distances,default=0.)
    for i,distance in enumerate(distances,1):
        if distance<2:count=0;start=None
        elif start is not None:count+=1
        elif distance>=3:start=i
        if count>=20:return dict(success=True,first_offset=start,verified_offset=i,maximum_distance=maximum)
    return dict(success=False,first_offset=None,verified_offset=None,maximum_distance=maximum)


def point(g):return [g['server_'+axis] for axis in 'xyz']


def main():
    root=Path(__file__).resolve().parents[1];base=root/'results/blind-recovery-20261012'
    out=root/'docs/reports/2026-10-12';out.mkdir(parents=True,exist_ok=True)
    if (out/'recovery-summary.json').exists():raise ValueError('one-use summary already exists')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    trials={};sources={};raw={}
    for case in ('far','corner'):
        for planner in ('fixture','visit_memory'):
            key=case+'/'+planner;run=base/key/'run';s=read(run/'summary.json');audit=read(run/'waypoint-audit.json')
            assert audit['verified'];rows=read(run/'waypoint-rows.json');raw[key]=rows
            switches=read(run/'waypoint-switches.json');start=switches[0]['sequence'] if switches else None
            positions=[point(r['server_after']) for r in rows]
            escape=None;crossing=None
            if start is not None and s['mode_steps']['waypoint']:
                origin=point(rows[start]['server_before']);escape=local_escape(positions[start:],origin)
                # Fixture's solid wall is x=[7,8). Require the full player body (radius .3)
                # on the opposite side. This geometry is read only AFTER game termination.
                chunk_x=16*s['initial_world']['chunk'][0]
                left=point(rows[0]['server_before'])[0]-chunk_x<7
                edge=chunk_x+(8.3 if left else 6.7)
                other_side=lambda p:p[0]>=edge if left else p[0]<=edge
                crossed=[i+1 for i,p in enumerate(positions) if other_side(p)]
                after=[i for i in crossed if i>start]
                already=other_side(origin)
                crossing=dict(success=None if already else bool(after),
                    status='already_on_far_side_at_stall' if already else ('crossed_after_stall' if after else 'not_crossed'),
                    first_action=after[0] if after else None,
                    first_episode_crossing=crossed[0] if crossed else None,
                    already_on_far_side_at_stall=already,episode_origin_side='left' if left else 'right')
            nav=[r for r in rows if r['mode']=='waypoint']
            failures=read(run/'exploration-route.json')['failures'] if planner=='visit_memory' else []
            scalar=lambda value: value[0] if isinstance(value,list) else value
            forward=[r for r in nav if scalar(r['action'].get('forward',0))]
            trials[key]=dict(steps=s['steps'],mode_steps=s['mode_steps'],status=s['status'],recovery_evaluated=bool(nav),
                trigger_action=start,trigger=switches[0]['trigger'] if switches else None,
                local_escape=escape,wall_crossing=crossing,root_success=s['root_inventory_success'],
                inventory_delta=s['inventory_final_delta'],route_completed=s['route_completed'],
                actual_goal_arrivals=sum(r['actions']>0 for r in s['route_receipts']),route_failure=s['route_failure'],
                candidate_failures=failures,goals=audit['goals'],path_blocks=audit['path_blocks'],
                navigation_path_blocks=sum(math.dist(point(r['server_after'])[::2],point(r['server_before'])[::2]) for r in nav),
                navigation_forward=len(forward),navigation_forward_heading_gt45=sum(abs(math.degrees(math.atan2(r['goal'][0],r['goal'][1])))>45 for r in forward),
                navigation_attack=sum(bool(scalar(r['action'].get('attack',0))) for r in nav),
                elapsed_control_s=s['elapsed_control_s'],inference_ms=s['inference_ms'],control_cycle_ms=s['control_cycle_ms'],
                gpu=s['gpu'],device_vram=s['device_vram'],stop_dispatches=audit['stop_dispatches'],
                keys_remaining=audit['keys_remaining'],owned_pids_remaining=audit['owned_pids_remaining'],
                action_logp_max_error=audit['action_logp_max_error'],world_sha256=s['initial_world']['sha256'],
                checkpoint_sha256=s['condition']['waypoint_actor_sha256'])
            for name in ('summary.json','waypoint-rows.json','waypoint-audit.json','waypoint-switches.json','events.jsonl'):
                sources[(run/name).relative_to(root).as_posix()]=digest(run/name)
    prefixes={}
    for case in ('far','corner'):
        from PIL import Image
        keys=[case+'/'+p for p in ('fixture','visit_memory')]
        limit=min(trials[k]['mode_steps']['root'] for k in keys);first={k:None for k in ('frame','action','pose')}
        for i in range(limit):
            a,b=[raw[k][i] for k in keys]
            frame_different=False
            if first['frame'] is None:
                with Image.open(base/keys[0]/'run'/a['frame']) as left,Image.open(base/keys[1]/'run'/b['frame']) as right:
                    frame_different=left.size!=right.size or left.mode!=right.mode or left.tobytes()!=right.tobytes()
            different=dict(frame=frame_different,
                action=a['action']!=b['action'],pose=any(a['server_before'][k]!=b['server_before'][k] for k in ('server_x','server_y','server_z','yaw','pitch')))
            for name,value in different.items():
                if value and first[name] is None:first[name]=i+1
        prefixes[case]=dict(compared_actions=limit,first_difference=first,identical=not any(first.values()))
    result=dict(trials=trials,root_prefix_comparison=prefixes,source_sha256=sources,api_calls=0,optimizer_updates=0,
        scope='Four development trials, known map vs pose/visit heuristic; privileged server pose and root RM, learned motor actions; not F3-only or LLM planning.')
    write_json(out/'recovery-summary.json',result)
    plot(root,base,out,trials,raw,read)
    print(json.dumps(result))


def plot(root,base,out,trials,raw,read):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,2,figsize=(12,11))
    for row,case in enumerate(('far','corner')):
        for col,planner in enumerate(('fixture','visit_memory')):
            key=case+'/'+planner;run=base/key/'run';s=read(run/'summary.json');world=s['initial_world'];rows=raw[key];r=trials[key]
            def canonical(p):
                x=p[0]-world['chunk'][0]*16;z=p[2]-world['chunk'][1]*16
                return [x,16-z if world['curriculum']['mirror_z'] else z]
            points=np.asarray([canonical(point(rows[0]['server_before']))]+[canonical(point(v['server_after'])) for v in rows])
            ax=axes[row,col]
            for x,z,w,h in ((0,0,16,1),(0,15,16,1),(0,0,1,16),(15,0,1,16),(7,1,1,world['curriculum']['wall_end'])):
                ax.add_patch(Rectangle((x,z),w,h,color='#bec5cd'))
            ax.add_patch(Rectangle((11,3),1,1,color='#895d35'))
            colors={'root':'#4386b7','waypoint':'#db792d','resumed':'#208357'}
            ax.add_collection(LineCollection(np.stack([points[:-1],points[1:]],axis=1),colors=[colors[v['mode']] for v in rows],linewidths=1.5))
            targets=[g['target'] for g in r['goals'].values()]
            for i,target in enumerate(targets,1):
                x,z=canonical(target);ax.plot(x,z,'+',color='#784eaa',markersize=9);ax.annotate(str(i),(x,z),xytext=(4,4),textcoords='offset points',fontsize=8)
            ax.plot(*points[0],'o',color='#222');ax.plot(*points[-1],'x',color='#bf2832',markersize=9)
            if r['trigger_action'] is not None:ax.plot(*points[r['trigger_action']],'*',color='#f1ba24',markersize=12)
            escape=r['local_escape']['success'] if r['local_escape'] else 'uncalled'
            crossing=r['wall_crossing']['success'] if r['wall_crossing'] else 'uncalled'
            if r['wall_crossing'] and r['wall_crossing']['already_on_far_side_at_stall']:crossing='already before stall'
            ax.set_title(f'{case}: {planner}\nlocal escape {escape} | wall crossed {crossing}\nwood {r["root_success"]} | {r["mode_steps"]["waypoint"]} recovery actions',fontsize=10)
            ax.set(xlim=(0,19),ylim=(0,16),xlabel='canonical x (blocks)',ylabel='canonical z (blocks)');ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('Real Minecraft recovery after detected stalls\nBlue: original policy | Orange: learned navigation | Green: resumed policy\nStar: detected stall | Crosses: speculative / supplied goals\nWall geometry shown only by offline evaluator',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/'actual-routes.png',dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
