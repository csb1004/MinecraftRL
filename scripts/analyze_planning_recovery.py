"""Read-only actual server-route diagnostics; no new episodes or model calls."""
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
from minecraftrl.runtime import write_json
from minecraftrl.stall_replay import wood_sample,log_count
from minecraftrl.world_store import digest


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'results/planning-recovery-implementation-20261009/route-analysis'
    output.mkdir(exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    runs=[root/f'results/goal-recovery-eval-104761-{stamp}/run' for stamp in (
        '20261009T104906245333Z','20261009T110944517695Z')]
    fig,axes=plt.subplots(1,2,figsize=(11,5));results=[]
    for ax,run,label in zip(axes,runs,('In-process fixed reply','Windows fixed-reply broker')):
        summary=read(run/'summary.json');initial=read(run/'initial-state.json');world=initial['initial_world']
        rows=read(run/'goal-recovery-rows.json');switches=read(run/'goal-switches.json')
        def position(obs):
            x=obs['server_x']-world['chunk'][0]*16;z=obs['server_z']-world['chunk'][1]*16
            return [x,16-z if world['curriculum']['mirror_z'] else z]
        points=np.array([position(initial['server'])]+[position(r['server_after']) for r in rows])
        entry=next(s for s in switches if s['mode']=='intervention')
        back=next(s for s in switches if s['mode']=='wood')
        phase_stats={}
        for i,row in enumerate(rows,1):
            phase=('neutral' if row['context_only'] else 'explore' if row['mode']=='intervention'
                   else 'wood_after' if i>=back['before_action'] else 'wood_before') if row.get('context_only') else (
                   'explore' if row['mode']=='intervention' else 'wood_after' if i>=back['before_action'] else 'wood_before')
            sample=wood_sample(row['server_after'],row['action'],str(run))
            stats=phase_stats.setdefault(phase,dict(steps=0,attack=0,movement=0,yaw=0,raw_stages=Counter()))
            stats['steps']+=1;stats['attack']+=int('attack' in sample.action)
            stats['movement']+=int(any(k in sample.action for k in ('forward','back','left','right')))
            stats['yaw']+=int(any(k.startswith('yaw') for k in sample.action));stats['raw_stages'][sample.stage]+=1
        intervention=[r['server_after'] for r in rows if r['mode']=='intervention']
        assessment=assess_attempt(entry['server_before'],intervention,reason=back['reason'],
            baseline_count=log_count(initial['server']),budget_ticks=200)
        crossing=next((i for i,p in enumerate(points) if p[0]>=8.3),None)
        stalled=[dict(step=r['step'],reason=r['reason']) for r in read(run/'stall-decisions.json') if r['event']=='stalled']
        item=dict(label=label,run=str(run),status=summary['status'],steps=len(rows),
            log_gain=log_count(rows[-1]['server_after'])-log_count(initial['server']),
            first_wall_crossing=crossing,first_stall=summary['trigger']['step'],
            switches=[dict(step=s['before_action'],mode=s['mode'],reason=s['reason']) for s in switches],
            recovery=assessment,phases=phase_stats,stalls=stalled,
            canonical_min=points.min(axis=0).tolist(),canonical_max=points.max(axis=0).tolist(),
            final_position=points[-1].tolist(),control_s=summary['elapsed_control_s'],
            source_sha256={name:digest(run/name) for name in ('events.jsonl','goal-recovery-rows.json','summary.json')})
        results.append(item)
        end=world['curriculum']['wall_end']
        for x,z,w,h in ((0,0,16,1),(0,15,16,1),(0,0,1,16),(15,0,1,16),(7,1,1,end)):
            ax.add_patch(Rectangle((x,z),w,h,color='#b5bbc4'))
        ax.add_patch(Rectangle((11,3),1,1,color='#9b663f'))
        colors=['#777777' if r.get('context_only') else '#d45c25' if r['mode']=='intervention' else '#306ea6' for r in rows]
        ax.add_collection(LineCollection(np.stack((points[:-1],points[1:]),axis=1),colors=colors,linewidths=1.8))
        for step,marker,name,color in ((0,'o','start','#111111'),(item['first_stall'],'*','stall','#9f32ad'),
                                      (back['before_action']-1,'s','return wood','#17805b'),(len(rows),'x','end','#b51b1b')):
            ax.plot(*points[step],marker,color=color,label=name,markersize=8)
        ax.set(xlim=(0,16),ylim=(0,16),xlabel='x (blocks)',ylabel='canonical z',
               title=f"{label}\nwood +{item['log_gain']}, {len(rows)} transitions")
        ax.set_aspect('equal');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Actual routes: blue = wood, orange = exploration\nDifferent rollouts; not a causal comparison')
    fig.tight_layout();fig.savefig(output/'routes.png',dpi=170);plt.close(fig)
    write_json(output/'analysis.json',dict(actual_api_calls=0,new_game_actions=0,optimizer_updates=0,runs=results,
        scope='Privileged server-route diagnostic; no causal advantage or path impossibility claim',
        plot_sha256=digest(output/'routes.png'),script_sha256=digest(Path(__file__))))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('source_sha256','run')} for r in results]))


if __name__=='__main__':main()
