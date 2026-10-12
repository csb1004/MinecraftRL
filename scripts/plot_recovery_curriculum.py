"""Plot actual server paths in a common orientation, with measured RM entries."""
import argparse,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,default=root/'results/recovery-curriculum-20261007');base=p.parse_args().batch
state=json.loads((base/'experiment.json').read_text());assert state['completed']
fig,axes=plt.subplots(math.ceil(len(state['runs'])/2),2,figsize=(11,5*math.ceil(len(state['runs'])/2)),squeeze=False)
for record,ax in zip(state['runs'],axes.flat):
    run=Path(record['path']);world=json.loads(Path(record['world']['path']).with_suffix('.json').read_text())
    ox,oz=[16*v for v in world['chunk']];mirror=world['curriculum']['mirror_z'];end=world['curriculum']['wall_end']
    def point(s):
        z=s['server_z']-oz
        return s['server_x']-ox,16-z if mirror else z
    initial=json.loads((run/'initial-state.json').read_text())['server'];positions=[point(initial)];entries=[]
    for line in (run/'events.jsonl').open():
        event=json.loads(line)
        if event['event']!='transition':continue
        positions.append(point(event['info']['workshop']))
        if event['info']['rl_reward']['recovery']['event']=='entered':entries.append(positions[-1])
    ax.add_patch(Rectangle((7,1),1,end,color='#666666',label='Bedrock wall'))
    ax.add_patch(Rectangle((11,3),1,1,color='#995522',label='Log column'))
    ax.plot(*zip(*positions),color='#2076b4',linewidth=1.5,label='Server path')
    ax.scatter(*positions[0],c='#138443',marker='o',s=50,label='Start',zorder=4)
    ax.scatter(*positions[-1],c='#a71d31',marker='x',s=70,label='End',zorder=5)
    if entries:ax.scatter(*zip(*entries),c='#e0a000',marker='s',s=30,label='Recovery entry',zorder=4)
    report=record['report'];ax.set_title(record['label']+'\n'+str(report['steps'])+' ticks; acquisition: '+str(report['server_target_reached']))
    ax.set(xlim=(0,16),ylim=(0,16),xlabel='Relative x (blocks)',ylabel='Relative z (mirrored runs normalized)',aspect='equal')
    ax.grid(alpha=.15);ax.legend(fontsize=7,loc='upper right')
for ax in list(axes.flat)[len(state['runs']):]:ax.set_visible(False)
fig.suptitle('Actual policy trajectories — local synthetic arenas',fontsize=14)
fig.tight_layout(rect=(0,0,1,.97));fig.savefig(base/'server-paths.png',dpi=150);fig.savefig(base/'server-paths.svg');plt.close(fig)
print(base/'server-paths.png')
