"""Independently audit E1 effect credit against raw committed server events (CPU)."""
import argparse
import json
from pathlib import Path
from analyze_rm_general_s2 import analyze, read
from minecraftrl.effect_skill import validate_effect_trace


def main(batch):
    analyze(batch)
    plan=read(batch/'plan.json'); result=[]
    for case in plan['runs']:
        folder=batch/case['name']
        if not (folder/'summary.json').exists():continue
        summary=read(folder/'summary.json')
        if not summary.get('execution_valid') or summary.get('worker_exit_code')!=0:continue
        rows=read(folder/'rm-rows.json')
        receipt=read(folder/'recovery-start.json')
        assert receipt['world_sha256']==case['world_sha256']
        assert receipt['task']==case['task'] and receipt['split']==case['role']
        assert summary['condition']['sampling_seed']==case['sampling_seed']
        previous=read(folder/'initial-state.json')['server']; index=0
        with (folder/'events.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                event=json.loads(line)
                if event['event']!='transition':continue
                row=rows[index];index+=1
                request=row['effect_request'];outcome=row['effect_outcome']
                server=event['info']['workshop']
                assert outcome['before_count']==sum(previous['server_inventory'].get(n,0) for n in request['items'])
                assert outcome['after_count']==sum(server['server_inventory'].get(n,0) for n in request['items'])
                assert outcome['before_tick']==previous['server_tick'] and outcome['after_tick']==server['server_tick']
                row.update(skill_reward=outcome['skill_reward'],skill_terminal=outcome['skill_terminal'])
                previous=server
        assert index==len(rows)==summary['steps']
        validate_effect_trace(dict(rows=rows,goal_spec=summary['goal_queue']['goal_spec'],
            context_fields=summary['condition']['context_fields']))
        events=[json.loads(line) for line in (folder/'skill-events.jsonl').read_text().splitlines()]
        successes=sum(row['skill_reward'] for row in rows)
        assert successes==summary['effect_successes']==sum(e['event']=='completed' for e in events)
        result.append(dict(name=case['name'],steps=index,root_rewards=sum(r['reward'] for r in rows),
            skill_rewards=successes,program_completed=summary['program_completed'],
            skill_segments=len({r['effect_request']['request_id'] for r in rows}),
            evidence_mismatches=0))
    output=dict(scope=plan['protocol'],runs=result,verified_rows=sum(r['steps'] for r in result),issues=[])
    (batch/'effect-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    main(parser.parse_args().batch)
