"""Small, shareable summaries/plots from audited local runs; no gameplay."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from minecraftrl.runtime import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refined',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];base=root/'results/rm-navigation-integration-20261014'
    if args.refined:base=root/'results/rm-refined-navigation-20261015'
    out=root/('docs/reports/2026-10-15' if args.refined else 'docs/reports/2026-10-14');out.mkdir(parents=True,exist_ok=True)
    stem='rm-refined-navigation' if args.refined else 'rm-navigation'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'));games=[];prefixes=[]
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    colors={'root':'#999999','navigation':'#2764c4','locate_wood':'#d88400','approach_wood':'#13885f','neutral':'#cccccc'}
    for revision,ax in zip(('far','corner') if args.refined else (1,2),axes):
        run=base/(('game-'+revision) if args.refined else ('game-v'+str(revision)))/'run'
        audit=read(run/'rm-navigation-audit.json');assert audit['verified']
        rows=read(run/'goal-recovery-rows.json');events=read(run/'planner-events.json');report=read(run/'summary.json')
        from PIL import Image
        with Image.open(run/'frames/000000.png') as frame:first_pixels=hashlib.sha256(frame.tobytes()).hexdigest()
        prefix=[]
        for row in rows:
            if row['owner']=='planner_neutral' or 'waypoint_goal' in row:break
            prefix.append(row['action'])
        prefixes.append(dict(first_pixels=first_pixels,actions=prefix))
        shown=set()
        for row in rows:
            category=('neutral' if row['owner']=='planner_neutral' else 'navigation' if 'waypoint_goal' in row else
                row['subgoal_context']['skill_id'] if row.get('subgoal_context',{}).get('phase')=='subgoal' else 'root')
            a,b=row['server_before'],row['server_after'];label=category if category not in shown else None;shown.add(category)
            ax.plot([a['server_x'],b['server_x']],[a['server_z'],b['server_z']],color=colors.get(category,'#aa44aa'),
                linewidth=1.5 if category!='root' else .6,label=label)
        for receipt in audit['navigation_receipts']:
            t=receipt['method']['arguments']['target'];ax.scatter(t[0],t[2],marker='x',c='#111111',s=55)
        first=rows[0]['server_before'];last=rows[-1]['server_after']
        ax.scatter(first['server_x'],first['server_z'],marker='o',c='black',s=35,label='start')
        ax.scatter(last['server_x'],last['server_z'],marker='s',c='#d62828',s=35,label='end')
        ax.set_title(f"{'v' if not args.refined else ''}{revision}: {len(rows)} transitions, logs +{audit['log_gain']}")
        ax.set_xlabel('server X');ax.set_ylabel('server Z');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        games.append(dict(revision=revision,steps=len(rows),status=report['status'],
            pause_reasons=[e['reason'] for e in events if e['event']=='paused'],
            planner_calls=audit['planner_calls'],navigation=[dict(outcome=r['outcome'],
                diagnosis=r['assessment']['diagnosis'],steps=r['committed_steps'],evidence=r['assessment']['evidence'])
                for r in audit['navigation_receipts']],subgoals=audit['subgoal_receipts'],
            log_gain=audit['log_gain'],actual_llm_calls=0,optimizer_updates=0,
            inference_ms=report['inference_ms'],control_cycle_ms=report['control_cycle_ms'],
            gpu=report['gpu'],device_vram=report['device_vram'],logp_error=audit['action_head_logp_max_error']))
    fig.suptitle('Fixed-plan integration: observed paths and speculative targets (x)')
    fig.savefig(out/(stem+'-paths.png'),dpi=150);plt.close(fig)
    same_pixels=prefixes[0]['first_pixels']==prefixes[1]['first_pixels']
    first_different=next((i for i,(a,b) in enumerate(zip(prefixes[0]['actions'],prefixes[1]['actions']),1) if a!=b),None)
    write_json(out/(stem+'-summary.json'),dict(games=games,
        prefix_check=dict(first_pixels_equal=same_pixels,first_different_action=first_different,
            common_prefix_checked=min(len(p['actions']) for p in prefixes)),
        scope='Different worlds/prefixes; no causal comparison. Server observations, no live LLM; no wall map in plot.'))
    print(json.dumps([dict(revision=g['revision'],steps=g['steps'],pause=g['pause_reasons'],log_gain=g['log_gain']) for g in games]))


if __name__=='__main__':main()
