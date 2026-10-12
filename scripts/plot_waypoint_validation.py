"""Compare preserved actual routes; dashed goals are plans, never inferred actions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];out=root/'results/waypoint-comparison-20261010'
out.mkdir(exist_ok=True)
plot=out/'actual-routes-v2.png';receipt=out/'plot-sources-v2.json'
if plot.exists() or receipt.exists():raise ValueError('revised plot already exists')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
fig,axes=plt.subplots(2,3,figsize=(16,10));sources={};records=[]
for col,(label,folder) in enumerate((('Initial head / stall','waypoint-validation-20261010'),
        ('Focused head / stall','waypoint-focused-validation-20261010'),
        ('Focused head / episode start','waypoint-initial-validation-20261010'))):
    for row,case in enumerate(('far','corner')):
        run=root/'results'/folder/case/'run';report=read(run/'summary.json');audit=read(run/'waypoint-audit.json')
        assert audit['verified'];world=report['initial_world'];rows=read(run/'waypoint-rows.json');switches=read(run/'waypoint-switches.json')
        def point(g):
            x=g['server_x']-world['chunk'][0]*16;z=g['server_z']-world['chunk'][1]*16
            return [x,16-z if world['curriculum']['mirror_z'] else z]
        points=np.asarray([point(rows[0]['server_before'])]+[point(r['server_after']) for r in rows])
        ax=axes[row,col];end=world['curriculum']['wall_end']
        for x,z,w,h in ((0,0,16,1),(0,15,16,1),(0,0,1,16),(15,0,1,16),(7,1,1,end)):
            ax.add_patch(Rectangle((x,z),w,h,color='#bcc3ce'))
        ax.add_patch(Rectangle((11,3),1,1,color='#895d35'))
        segments=np.stack([points[:-1],points[1:]],axis=1)
        colors={'root':'#4984b8','waypoint':'#db782c','resumed':'#238457'}
        ax.add_collection(LineCollection(segments,colors=[colors[r['mode']] for r in rows],linewidths=1.5))
        if switches:
            goals=np.asarray([point(dict(zip(('server_x','server_y','server_z'),t))) for t in switches[0]['targets']])
            ax.plot(goals[:,0],goals[:,1],':o',color='#75489f',markersize=4,linewidth=1,label='planned waypoints')
            for i,(x,z) in enumerate(goals):ax.annotate(str(i+1),(x,z),xytext=(5,4),textcoords='offset points',color='#75489f')
        ax.plot(*points[0],'o',color='#111111',markersize=5,label='episode start')
        ax.plot(*points[-1],'x',color='#bc2727',markersize=9,label='episode end')
        reached=sum(r['actions']>0 for r in report['route_receipts']);skipped=sum(r['actions']==0 for r in report['route_receipts'])
        skill=f'waypoints reached {reached}, already satisfied {skipped}' if report['mode_steps']['waypoint'] else 'waypoint skill NOT CALLED'
        ax.set_title(f'{case} / {label}\n{skill}\nwood success: {report["root_inventory_success"]}',fontsize=10)
        ax.set(xlim=(0,16),ylim=(0,16),xlabel='canonical x (blocks)',ylabel='canonical z (blocks)');ax.set_aspect('equal');ax.grid(alpha=.15)
        records.append(dict(case=case,condition=label,steps=report['steps'],waypoint_actions=report['mode_steps']['waypoint'],
            reached=reached,skipped=skipped,route_completed=report['route_completed'],root_success=report['root_inventory_success']))
        for name in ('waypoint-rows.json','summary.json','waypoint-audit.json'):sources[str((run/name).relative_to(root))]=digest(run/name)
fig.suptitle('Actual learned-policy routes on known development fixtures\nBlue: original wood | Orange: learned waypoint | Green: wood resumed\nDifferent trigger conditions and root prefixes: not a controlled improvement estimate',fontsize=13)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.90),ncol=3,fontsize=10)
fig.tight_layout(rect=(0,0,1,.87));fig.savefig(plot,dpi=165);plt.close(fig)
write_json(receipt,dict(rows=records,source_sha256=sources,plot_sha256=digest(plot)))
print(plot)
