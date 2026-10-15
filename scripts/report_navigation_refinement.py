"""Summarize only finished audited trials; retain unrun and teacher outcomes separately."""
import json,math
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/navigation-refinement-20261014'


def main():
    runs=[]
    for path in sorted(BASE.glob('*/work.json')):
        work=json.loads(path.read_text());run=path.parent/'run'
        if not work['completed']:continue
        report=json.loads((run/'summary.json').read_text());config=json.loads((path.parent/'protocol.json').read_text())
        rows=json.loads((run/'navigation-rows.json').read_text());goals=[]
        for receipt in report['navigation_receipts']:
            subset=[r for r in rows if r['index']==receipt['index']]
            if not subset:goals.append(receipt);continue
            def pos(g):return [g['server_x'],g['server_z']]
            target=receipt['target'];points=[pos(subset[0]['server_before'])]+[pos(r['server_after']) for r in subset]
            distances=[math.hypot(x-target[0],z-target[2]) for x,z in points]
            path_length=sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(points,points[1:]))
            goals.append(dict(receipt,start_distance=distances[0],minimum_distance=min(distances),path_blocks=path_length,
                distance_increase_steps=sum(b>a+.02 for a,b in zip(distances,distances[1:])),path_xz=points))
        labels=[r for r in rows if r['owner']=='teacher'];bins=[0]*4;near=0
        for r in labels:
            angle=abs(math.degrees(math.atan2(r['goal'][0],r['goal'][1])))
            bins[0 if angle<=15 else 1 if angle<=45 else 2 if angle<=135 else 3]+=1
            near+=r['goal'][2]*16<=1
        runs.append(dict(label=path.parent.name,mode=config['mode'],split=config['split'],steps=report['steps'],
            actor_sha256=report['actor_sha256'],seed=report['initial_world']['seed'],
            learned_arrivals=report['learned_arrivals'],teacher_arrivals=report['teacher_arrivals'],goals=goals,
            unrun_goal_count=len(config['points'])-len(goals),owner_steps=report['mode_steps'],
            expert_angle_bins=bins,expert_within_one_block=near,
            root_inventory_success=report['root_inventory_success'],inference_ms=report['inference_ms'],
            cycle_ms=report['control_cycle_ms'],gpu=report['gpu'],device_vram=report['device_vram'],
            first_frame_sha256=digest(run/'frames/000000.png'),world_sha256=config['world_sha256'],
            probability_audited=(run/'navigation-probability-audit.json').exists()))
    result=dict(runs=runs,api_calls=0,scope='local privileged-coordinate development; no final evaluation; unrun excluded')
    write_json(BASE/'summary.json',result)
    labels={r['label']:r for r in runs}
    if all('evaluate-'+kind+'-'+world in labels for kind in ('prior','linear','mlp','ppo') for world in ('far','corner')):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        folder=ROOT/'docs/reports/2026-10-14';folder.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
        limits={}
        for world in ('far','corner'):
            points=[]
            for kind in ('prior','linear','mlp','ppo'):
                run=labels['evaluate-'+kind+'-'+world];origin=run['goals'][0]['path_xz'][0]
                for goal in run['goals']:
                    if 'path_xz' not in goal:continue
                    points.extend((p[0]-origin[0],p[1]-origin[1]) for p in goal.get('path_xz',[]))
                    points.append((goal['target'][0]-origin[0],goal['target'][2]-origin[1]))
            limits[world]=((min(p[0] for p in points)-.5,max(p[0] for p in points)+.5),
                (min(p[1] for p in points)-.5,max(p[1] for p in points)+.5))
        for column,kind in enumerate(('prior','linear','mlp','ppo')):
            for row,world in enumerate(('far','corner')):
                run=labels['evaluate-'+kind+'-'+world];ax=axes[row,column]
                origin=run['goals'][0]['path_xz'][0]
                for goal in run['goals']:
                    if 'path_xz' not in goal:continue
                    color=plt.get_cmap('tab10')(goal['index'])
                    points=goal['path_xz'];x=[p[0]-origin[0] for p in points];z=[p[1]-origin[1] for p in points]
                    ax.plot(x,z,color=color,linewidth=1,alpha=.8)
                    tx,tz=goal['target'][0]-origin[0],goal['target'][2]-origin[1]
                    ax.scatter([tx],[tz],marker='x',color=color,s=35)
                    ax.annotate(str(goal['index']+1),(tx,tz),xytext=(4,4),textcoords='offset points',fontsize=8,color=color)
                    ax.add_patch(plt.Circle((tx,tz),.35,color=color,fill=False,alpha=.35))
                    ax.scatter(x[-1],z[-1],marker='o' if goal['student_success'] else 's',color=color,s=18)
                ax.scatter(0,0,color='black',s=25);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
                ax.set_xlim(*limits[world][0]);ax.set_ylim(*limits[world][1])
                attempted=sum(g['outcome']!='already_satisfied' for g in run['goals'])
                ax.set_title(f"{world} / {kind}: {run['learned_arrivals']}/{attempted} arrivals")
                ax.set_xlabel('x from episode start (blocks)');ax.set_ylabel('z from episode start (blocks)')
        fig.suptitle('Learned local navigation: fixed goals, 100 ticks per goal; matched axes within each world\nNumbered cross = goal; circle = arrival radius; square endpoint = deadline failure',fontsize=12)
        fig.savefig(folder/'navigation-refinement-paths.png',dpi=160);plt.close(fig)
        compact=json.loads(json.dumps(result))
        for run in compact['runs']:
            for goal in run['goals']:goal.pop('path_xz',None)
        write_json(folder/'navigation-refinement-summary.json',compact)
    print(json.dumps([{k:r[k] for k in ('label','steps','learned_arrivals','teacher_arrivals','owner_steps')} for r in runs]))


if __name__=='__main__':main()
