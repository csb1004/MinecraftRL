"""Read-only routes, discovery timing and input comparability for the new fix."""
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from minecraftrl.runtime import write_json
from minecraftrl.stall_replay import wood_sample,log_count
from minecraftrl.world_store import digest


root=Path(__file__).resolve().parents[1]
base=root/'results/fresh-target-handoff-game-20261009'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
state=read(base/'work.json');assert state['completed'] and state['error'] is None
new=Path(state['run']);old=root/'results/goal-recovery-eval-104761-20261009T144548946737Z/run'
out=base/'analysis';out.mkdir(exist_ok=False)
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
reports=[];all_rows=[]
for ax,run,label in zip(axes,(old,new),('Prior boundary','Fresh acquisition handoff')):
    rows=read(run/'goal-recovery-rows.json');summary=read(run/'summary.json')
    initial=read(run/'initial-state.json');world=initial['initial_world']
    switches=read(run/'goal-switches.json');all_rows.append(rows)
    def position(s):
        x=s['server_x']-world['chunk'][0]*16;z=s['server_z']-world['chunk'][1]*16
        return [x,16-z if world['curriculum']['mirror_z'] else z]
    points=np.array([position(initial['server'])]+[position(r['server_after']) for r in rows])
    back=next((s for s in switches if s['mode']=='wood'),None)
    first_fresh=next((i for i,r in enumerate(rows,1) if r['mode']=='intervention' and
        r['server_before']['wood_v3_kind']==0 and r['server_after']['wood_v3_kind']==1),None)
    colors=[];phases={}
    for i,row in enumerate(rows,1):
        phase=('neutral' if row.get('context_only') else 'explore' if row['mode']=='intervention' else
               'wood_after' if back and i>=back['before_action'] else 'wood_before')
        sample=wood_sample(row['server_after'],row['action'],str(run))
        metrics=phases.setdefault(phase,dict(steps=0,attack=0,movement=0,raw_stages=Counter()))
        metrics['steps']+=1;metrics['attack']+=int('attack' in sample.action)
        metrics['movement']+=int(any(k in sample.action for k in ('forward','back','left','right')))
        metrics['raw_stages'][sample.stage]+=1
        colors.append('#d45c25' if phase=='explore' else '#888888' if phase=='neutral' else '#246dae')
    gain=log_count(rows[-1]['server_after'])-log_count(initial['server'])
    reports.append(dict(label=label,run=str(run),steps=len(rows),status=summary['status'],log_gain=gain,
        first_fresh_discovery=first_fresh,return_action=back['before_action'] if back else None,
        extra_exploration_after_discovery=(back['after_step']-first_fresh if back and first_fresh else None),
        phases=phases,first_wall_crossing=next((i for i,p in enumerate(points) if p[0]>=8.3),None),
        control_s=summary['elapsed_control_s'],inference_ms=summary['inference_ms'],
        control_cycle_ms=summary['control_cycle_ms'],gpu=summary['gpu'],device_vram=summary['device_vram'],
        final_position=points[-1].tolist(),attempts=summary.get('recovery_attempts',[]),
        sources_sha256={name:digest(run/name) for name in ('summary.json','goal-recovery-rows.json','goal-switches.json','planner-events.json')}))
    for x,z,w,h in ((0,0,16,1),(0,15,16,1),(0,0,1,16),(15,0,1,16),(7,1,1,world['curriculum']['wall_end'])):
        ax.add_patch(Rectangle((x,z),w,h,color='#b5bbc4'))
    ax.add_patch(Rectangle((11,3),1,1,color='#9b663f'))
    ax.add_collection(LineCollection(np.stack((points[:-1],points[1:]),axis=1),colors=colors,linewidths=1.5))
    markers=[(0,'o','start','#111111'),(len(rows),'x','end','#b51b1b')]
    if first_fresh:markers.append((first_fresh,'*','discovery','#9f32ad'))
    if back:markers.append((back['after_step'],'s','wood handoff','#17805b'))
    for step,marker,name,color in markers:ax.plot(*points[step],marker,color=color,label=name,markersize=8)
    ax.set(xlim=(0,16),ylim=(0,16),xlabel='x',ylabel='canonical z',title=f'{label}: wood +{gain}\n{len(rows)} transitions')
    ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=7)
comparison=dict(first_pixel_difference=None,first_action_difference=None,first_pose_difference=None)
bound=min(len(r) for r in all_rows)
if any(r['first_fresh_discovery'] for r in reports):bound=min(r['first_fresh_discovery'] for r in reports if r['first_fresh_discovery'])
for i,(a,b) in enumerate(zip(*all_rows),1):
    if i>bound:break
    if comparison['first_action_difference'] is None and a['action']!=b['action']:comparison['first_action_difference']=i
    if comparison['first_pose_difference'] is None and any(a['server_before'][k]!=b['server_before'][k] for k in ('server_x','server_y','server_z','yaw','pitch')):
        comparison['first_pose_difference']=i
    if comparison['first_pixel_difference'] is None:
        with Image.open(old/a['frame']) as x,Image.open(new/b['frame']) as y:
            if not np.array_equal(np.asarray(x),np.asarray(y)):comparison['first_pixel_difference']=i
    if all(v is not None for v in comparison.values()):break
comparison['checked_pre_handoff_prefix_steps']=bound
comparison['pre_handoff_difference_present']=any(comparison[k] is not None for k in ('first_pixel_difference','first_action_difference','first_pose_difference'))
fig.suptitle('Actual development trajectories | orange exploration, blue wood | no causal performance claim')
fig.savefig(out/'routes.png',dpi=160);plt.close(fig)
audit=read(new/'fresh-target-handoff-audit.json');assert audit['verified']
result=dict(completed=True,actual_new_games=1,api_calls=0,optimizer_updates=0,rows=reports,
    comparison=comparison,audit=audit,scope='Boundary fix validation; prior trajectory shown for diagnosis, not a success-rate comparison.')
write_json(out/'report.json',result)
print(json.dumps(dict(rows=[{k:r[k] for k in ('label','steps','status','log_gain','first_fresh_discovery','return_action','extra_exploration_after_discovery','first_wall_crossing')} for r in reports],comparison=comparison)))
