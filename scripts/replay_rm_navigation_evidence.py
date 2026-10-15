"""Classify preserved real navigation attempts, without changing games or labels."""
from collections import Counter
import json
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.waypoint_skill import capability,assess_navigation
from minecraftrl.world_store import digest


def main():
    root=Path(__file__).resolve().parents[1];base=root/'results/adaptive-recovery-20261014'
    out=root/'results/rm-navigation-integration-20261014/real-trace-replay.json'
    if out.exists():raise FileExistsError(out)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    games=[];hashes={};assessments=[]
    for key,relative in read(base/'completed-trials.json').items():
        run=base/relative;rows=read(run/'waypoint-rows.json');report=read(run/'summary.json')
        route=read(run/'exploration-route.json')
        for name in ('waypoint-rows.json','summary.json','exploration-route.json'):
            path=run/name;hashes[str(path.relative_to(root))]=digest(path)
        windows=[];current=[];nav_sequence=0
        for step,row in enumerate(rows,1):
            if current and (row['mode']!='waypoint' or row['target']!=current[-1][1]['target']):
                windows.append(current);current=[]
            if row['mode']=='waypoint':
                nav_sequence+=1;current.append((step,row,nav_sequence))
        if current:windows.append(current)
        results=[]
        for window in windows:
            first,last=window[0],window[-1];target=first[1]['target'];sequence=last[2]
            reached=[r for r in report['route_receipts'] if r['target']==target and r['sequence']==sequence]
            failed=[r for r in route['failures'] if r['target']==target and r['sequence']==sequence]
            if reached:reason='predicate_satisfied'
            elif failed:reason='subgoal_handoff' if failed[0]['reason'].startswith('root_probe') else 'budget_exhausted'
            elif last[0]<len(rows) and rows[last[0]]['mode']=='root_probe':reason='subgoal_handoff'
            else:reason='control_budget'
            condition=report['condition']
            cap=capability(0,condition['checkpoint'],condition['waypoint_actor_sha256'],condition['embedding_sha256'],target)
            result=assess_navigation(cap,[v[1] for v in window],reason=reason)
            assert result['outcome'] not in ('unrun','unverified')
            if reached:assert result['outcome']=='completed'
            if reason in ('subgoal_handoff','control_budget'):assert not result['retry_blocking']
            item=dict(start_step=first[0],end_step=last[0],assessment=result);results.append(item);assessments.append(result)
        games.append(dict(case=key,total_steps=len(rows),navigation_steps=sum(len(w) for w in windows),
            recovery_called=bool(windows),attempts=results))
    for path,sha in hashes.items():assert digest(root/path)==sha,path
    result=dict(verified=True,scope='New classifier replay on preserved real trajectories; no new gameplay, LLM call or learning',
        total_games=len(games),called_games=sum(g['recovery_called'] for g in games),
        total_steps=sum(g['total_steps'] for g in games),navigation_steps=sum(g['navigation_steps'] for g in games),
        outcomes=dict(Counter(a['outcome'] for a in assessments)),diagnoses=dict(Counter(a['diagnosis'] for a in assessments)),
        games=games,source_sha256=hashes,raw_sources_unchanged=True)
    write_json(out,result);print(json.dumps({k:v for k,v in result.items() if k not in ('games','source_sha256')}))


if __name__=='__main__':main()
