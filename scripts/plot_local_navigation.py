"""Post-run paths on the evaluator-only saved arena map; never imported by the policy."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('labels',nargs='+');p.add_argument('--output-stem',default='trajectory-comparison')
    p.add_argument('--results-root',type=Path,default=Path('results/local-navigation-detour-20261019'));a=p.parse_args()
    if not a.output_stem.replace('-','').isalnum():raise ValueError('simple output name required')
    root=Path(__file__).resolve().parents[1];out=(root/a.results_root).resolve()
    if not out.is_relative_to(root/'results'):raise ValueError('project results required')
    sys.path.insert(0,str(root/'results/local-wall-diagnosis-20261017/analysis-deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    geometry=json.loads((root/'results/local-wall-diagnosis-20261017/maze-geometry.json').read_text())
    cx,cz=geometry['chunk'];ox,oz=cx*16,cz*16
    fig,axes=plt.subplots(1,len(a.labels),figsize=(6*len(a.labels),6),squeeze=False)
    colors=dict(collect_wood='#91a3b0',waypoint='#0077bb',locate_wood='#cc9d18',approach_wood='#cc6677',routed_approach='#009977',neutral='#bbbbbb')
    for ax,label in zip(axes[0],a.labels):
        run=out/label/'run';rows=json.loads((run/'goal-recovery-rows.json').read_text())
        report=json.loads((run/'summary.json').read_text())
        for z,row in enumerate(geometry['grid']):
            for x,block in enumerate(row):
                if block=='minecraft:bedrock':ax.add_patch(Rectangle((ox+x,oz+z),1,1,color='#303a44'))
                elif block.endswith('_log'):ax.add_patch(Rectangle((ox+x,oz+z),1,1,color='#be8b4b'))
        for r in rows:
            phase=('neutral' if r['owner']=='planner_neutral' else 'routed_approach' if 'local_route' in r else
                   'waypoint' if 'waypoint_target' in r else r.get('subgoal_context',{}).get('skill_id','collect_wood'))
            if r.get('local_route',{}).get('stage')=='locate_wood':phase='locate_wood'
            before,after=r['server_before'],r['server_after']
            ax.plot([before['server_x'],after['server_x']],[before['server_z'],after['server_z']],
                color=colors.get(phase,'black'),lw=1.1,alpha=.7)
        start=rows[0]['server_before'];end=rows[-1]['server_after']
        ax.scatter([start['server_x']],[start['server_z']],marker='o',s=70,c='white',edgecolor='black',zorder=5,label='start')
        ax.scatter([end['server_x']],[end['server_z']],marker='X',s=80,c='#ee5533',zorder=5,label='end')
        ax.set(xlim=(ox-.3,ox+16.3),ylim=(oz+16.3,oz-.3),aspect='equal',xlabel='server X',ylabel='server Z')
        gain=sum(v for k,v in report['inventory_final_delta'].items() if k.endswith('_log'))
        title=f"{label}\n{report['elapsed_control_s']:.1f}s | {report['steps']} actions | policy logs +{gain}"
        preservation=out/label/'world-preservation.json'
        if preservation.exists():
            inventories=json.loads(preservation.read_text())['saved_inventory']
            saved=sum(v['Count'] for v in inventories['level.dat/Player'] if v['id'].endswith('_log'))
            if saved!=gain:title+=f"\nSaved logs +{saved} after control ended"
        ax.set_title(title)
        ax.ticklabel_format(useOffset=False,style='plain');ax.grid(alpha=.15)
    for phase,color in colors.items():axes[0][-1].plot([],[],color=color,label=phase)
    axes[0][-1].legend(loc='upper left',fontsize=8)
    fig.suptitle('Actual neural-policy trajectories (development runs)',fontsize=15)
    fig.text(.5,.015,'Initial arena NBT: evaluation figure only.\nPlanner observations: bounded live local sensor.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.94));fig.savefig(out/(a.output_stem+'.png'),dpi=160)
    fig.savefig(out/(a.output_stem+'.svg'));print(out/(a.output_stem+'.png'))


if __name__=='__main__':main()
