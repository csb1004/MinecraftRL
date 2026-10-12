"""Plot preserved server routes by goal mode after a completed diagnostic batch."""
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


root = Path(__file__).resolve().parents[1]
base = root/'results/goal-recovery-comparison-20261009'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
state = read(base/'experiment.json')
assert state['completed'] and not state['errors']
target = base/'actual-server-routes.png'
if target.exists(): raise FileExistsError(target)
kinds = list(dict.fromkeys(r['kind'] for r in state['runs']))
fig,axes = plt.subplots(len(kinds),2,figsize=(12,5.1*len(kinds)),squeeze=False)
phase_metrics = []
sources = {str(base/'experiment.json'):digest(base/'experiment.json')}
for row,kind in enumerate(kinds):
    natural_points = []
    for col,condition in enumerate(('wood','explore')):
        trial = next(r for r in state['runs'] if r['kind']==kind and r['condition']==condition)
        run = Path(trial['path']); audit = read(run/'goal-recovery-audit.json')
        assert trial['audited'] and audit['verified']
        initial = read(run/'initial-state.json'); world = initial['initial_world']
        fixture = 'curriculum' in world
        def position(s):
            x,z = s['server_x'],s['server_z']
            if fixture:
                x -= world['chunk'][0]*16; z -= world['chunk'][1]*16
                if world['curriculum']['mirror_z']: z = 16-z
            return [x,z]
        points = [position(initial['server'])]; modes = []; phases = {}
        for line in (run/'events.jsonl').open(encoding='utf-8'):
            e = json.loads(line)
            if e['event'] != 'transition': continue
            points.append(position(e['info']['workshop']))
            mode = e['controller']; modes.append(mode)
            stats = phases.setdefault(mode,dict(steps=0,attack=0,movement=0,yaw=0))
            stats['steps'] += 1
            stats['attack'] += bool(e['action']['attack'][0])
            stats['movement'] += any(e['action'][k][0] for k in ('forward','back','left','right'))
            stats['yaw'] += bool(e['action']['camera'][0][1])
        p = np.asarray(points); ax = axes[row,col]
        if fixture:
            end = world['curriculum']['wall_end']
            for x,z,w,h in ((0,0,16,1),(0,15,16,1),(0,0,1,16),(15,0,1,16),(7,1,1,end)):
                ax.add_patch(Rectangle((x,z),w,h,color='#b5bbc4',alpha=.8))
            ax.add_patch(Rectangle((11,3),1,1,color='#9b663f'))
            ax.set_xlim(0,16); ax.set_ylim(0,16)
        segments = np.stack([p[:-1],p[1:]],axis=1)
        ax.add_collection(LineCollection(segments,colors=['#d45c25' if m=='steve1_intervention' else '#306ea6' for m in modes],linewidths=1.8))
        if not fixture:
            ax.autoscale(); natural_points.append(p)
        ax.plot(*p[0],'o',color='#111111',label='start')
        ax.plot(*p[-1],'x',color='#b51b1b',markersize=9,label='end')
        step = audit['first_stall_step']
        if step is not None: ax.plot(*p[step],'*',color='#9f32ad',markersize=12,label='stall trigger')
        ax.set_title(f"{kind} / {condition}: wood +{audit['log_gain']}, {audit['intervention_steps']} intervention ticks")
        ax.set_aspect('equal',adjustable='box'); ax.set_xlabel('x (blocks)'); ax.set_ylabel('canonical z' if fixture else 'z (blocks)')
        ax.grid(alpha=.15); ax.legend(loc='upper left',fontsize=8)
        phase_metrics.append(dict(label=trial['label'],phases=phases,first_stall_step=step,
            max_post_intervention_yaw_delta=audit['max_post_intervention_yaw_delta'],
            max_post_intervention_displacement=audit['max_post_intervention_displacement']))
        sources[str(run/'events.jsonl')] = digest(run/'events.jsonl')
    if natural_points:
        both = np.concatenate(natural_points,axis=0)
        for ax in axes[row]:
            ax.set_xlim(both[:,0].min()-2,both[:,0].max()+2)
            ax.set_ylim(both[:,1].min()-2,both[:,1].max()+2)
fig.suptitle('Actual neural gameplay: blue = original wood, orange = switched condition\nPrivileged stall trigger; intervention is not evidence of successful recovery',fontsize=13)
fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(target,dpi=170); plt.close(fig)
write_json(base/'phase-metrics.json',dict(rows=phase_metrics,source_sha256=sources,plot_sha256=digest(target)))
print(target)
