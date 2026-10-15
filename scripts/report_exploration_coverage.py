"""Audited local exploration metrics; coverage never substitutes for inventory success."""
import json,math,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/reports/2026-10-15'


def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));games=[];prefixes={}
    cases=[(batch,world) for batch in ('rm-refined-navigation-20261015','exploration-coverage-20261015') for world in ('far','corner')]
    if (ROOT/'results/exploration-motion-20261015/game-corner/run/rm-navigation-audit.json').exists():
        cases.append(('exploration-motion-20261015','corner'))
    fig,axes=plt.subplots(math.ceil(len(cases)/2),2,figsize=(12,5*math.ceil(len(cases)/2)),constrained_layout=True)
    colors={'root':'#999999','navigation':'#2764c4','locate_wood':'#d88400','approach_wood':'#13885f','neutral':'#bbbbbb'}
    for (batch,world),ax in zip(cases,axes.flat):
        folder=ROOT/'results'/batch/('game-'+world);run=folder/'run'
        report=read(run/'summary.json');audit=read(run/'rm-navigation-audit.json');work=read(folder/'work.json')
        assert work['completed'] and not work['error'] and audit['verified'] and audit['root_preserved']
        assert audit['actual_llm_calls']==audit['optimizer_updates']==audit['post_stop_dispatches']==audit['post_stop_inferences']==audit['held_keys_after_release']==0
        assert audit['remaining_owned_pids']==[]
        for file,sha in work['frozen_assets'].items():assert digest(ROOT/file)==sha
        for file,sha in audit['source_sha256'].items():assert digest(run/file)==sha
        rows=read(run/'goal-recovery-rows.json');schedule=read(run/'rm-schedule.json')
        cell=lambda g:tuple(math.floor(g['server_'+a]/s) for a,s in zip('xyz',(2,4,2)))
        seen={cell(rows[0]['server_before'])};new_nav=entries=repeated=0;shown=set();prefix=[]
        for row in rows:
            nav='waypoint_goal' in row;a,b=row['server_before'],row['server_after'];old,c=cell(a),cell(b)
            if nav:
                new_nav+=c not in seen
                if c!=old:entries+=1;repeated+=c in seen
            seen.add(c)
            category=('neutral' if row['owner']=='planner_neutral' else 'navigation' if nav else
                row['subgoal_context']['skill_id'] if row.get('subgoal_context',{}).get('phase')=='subgoal' else 'root')
            if category=='root' and not shown.intersection({'neutral','navigation'}):prefix.append(row['action'])
            label=category if category not in shown else None;shown.add(category)
            ax.plot([a['server_x'],b['server_x']],[a['server_z'],b['server_z']],color=colors[category],
                linewidth=.7 if category=='root' else 1.5,label=label)
        for r in audit['navigation_receipts']:
            x,y,z=r['method']['arguments']['target'];ax.scatter(x,z,marker='x',c='#222222',s=45)
        for marker,r,color in [('o',rows[0]['server_before'],'black'),('s',rows[-1]['server_after'],'#d62828')]:
            ax.scatter(r['server_x'],r['server_z'],marker=marker,c=color,s=30)
        strategy=report['condition']['planning'].get('navigation_strategy','visited')
        ax.set_title(f'{strategy} / {world}: nav new cells {new_nav}, logs +{audit["log_gain"]}')
        ax.set_xlabel('server X');ax.set_ylabel('server Z');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=8)
        with Image.open(run/'frames/000000.png') as im:pixel_sha=hashlib.sha256(im.tobytes()).hexdigest()
        key=(world,strategy);prefixes[key]=(pixel_sha,prefix)
        comparison=None
        if strategy!='visited' and (world,'visited') in prefixes:
            old_sha,old_prefix=prefixes[(world,'visited')]
            comparison=dict(first_pixels_equal=old_sha==pixel_sha,first_different_action=next((i for i,(a,b) in enumerate(zip(prefix,old_prefix),1) if a!=b),None),checked=min(len(prefix),len(old_prefix)),causal_comparison=False)
        games.append(dict(batch=batch,world=world,strategy=strategy,steps=audit['steps'],log_gain=audit['log_gain'],status=report['status'],
            execution_valid=report['execution_valid'],worker_exit_code=report['worker_exit_code'],runtime_error=audit['runtime_error'],
            navigation_new_cells=new_nav,navigation_cell_entries=entries,repeated_navigation_entries=repeated,
            planner_calls=audit['planner_calls'],maximum_requests=report['condition']['planning']['maximum_calls'],
            navigation=[dict(outcome=r['outcome'],reason=r['reason'],steps=r['committed_steps'],diagnosis=r['assessment']['diagnosis'],
                exploration=r.get('exploration'),evidence=r['assessment']['evidence']) for r in audit['navigation_receipts']],
            subgoals=[dict(skill_id=r['skill_id'],outcome=r['outcome'],steps=r['committed_steps']) for r in audit['subgoal_receipts']],
            pending_nodes=[dict(revision=i+1,id=n['id'],status=n['status']) for i,p in enumerate(schedule['plans']) for n in p['nodes'] if not n.get('receipt')],
            inference_ms=report['inference_ms'],control_cycle_ms=report['control_cycle_ms'],gpu=report['gpu'],device_vram=report['device_vram'],
            prefix_comparison=comparison,action_logp_error=audit['action_head_logp_max_error'],post_stop_actions=0,held_keys=0,owned_pids=[],
            source_audit_sha256=digest(run/'rm-navigation-audit.json')))
    for ax in list(axes.flat)[len(cases):]:ax.axis('off')
    fig.savefig(OUT/'exploration-coverage-paths.png',dpi=130);plt.close(fig)
    result=dict(verified=True,games=games,api_calls=0,optimizer_updates=0,
        scope='Development diagnostics; prefixes and request budgets can differ; no causal/generalization claim; source is privileged server')
    write_json(OUT/'exploration-coverage-summary.json',result)
    print(json.dumps([dict(world=g['world'],strategy=g['strategy'],steps=g['steps'],new_cells=g['navigation_new_cells'],
        arrivals=sum(r['outcome']=='completed' for r in g['navigation']),attempts=len(g['navigation']),log_gain=g['log_gain']) for g in games]))


if __name__=='__main__':main()
