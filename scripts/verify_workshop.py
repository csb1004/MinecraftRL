"""Audit actual workshop events, saved inventories, and saved workstation blocks."""
import argparse
from collections import Counter
import json
from pathlib import Path
from minecraftrl.runtime import distribution,write_json
from minecraftrl.saved_inventory import verify_saved_inventory
from minecraftrl.saved_block import saved_block
from minecraftrl.world_store import digest


def audit(base):
    state=json.loads((base/'experiment.json').read_text());rows=[];cycles=[]
    assert state['completed'] and state['source_unchanged'] and not state['unrun']
    for row in state['reports']:
        seed=row['seed'];folder=base/str(seed)/'run';r=json.loads((folder/'summary.json').read_text())
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        transitions=[e for e in events if e['event']=='transition']
        ack=next(e['monotonic_s'] for e in events if e['event']=='stop_acknowledged')
        assert all(e['monotonic_s']<=ack for e in events if e['event']=='policy_dispatch_started')
        assert len(transitions)==r['steps'] and all(e['action']['sneak']==0 for e in transitions)
        assert r['release']=='neutral_command_sent' and not r['process_cleanup']['remaining_live_pids']
        assert r['workshop_cleanup']['cursor_empty'] and r['workshop_cleanup']['screen']==0
        request=json.loads((folder/'request.json').read_text());assert request['controller']=='workshop_gui' and not request['learned_policy_loaded']
        worlds=base/str(seed)/'worlds';slot=next(p for p in worlds.iterdir() if p.is_dir())
        archive=slot/r['world_save']['generation']/'world.zip'
        inv=r['workshop_cleanup']['inventory'];check=verify_saved_inventory(archive,inv)
        assert check['verified'] and saved_block(archive,r['table_position'])=='minecraft:crafting_table'
        ingredients=Counter()
        for m in r['milestones']:
            if m['event']=='recipe_committed':ingredients.update(m['ingredients'])
        takes=[e for e in transitions if e['info']['workshop_stage'].startswith('take_') and e['action']['attack']==1]
        uses=[e for e in transitions if e['action']['use']==1]
        assert len(uses)==2
        if state.get('stop_at_stage'):
            assert r['status']=='cancelled' and not r['workshop_success']
            assert inv=={'wooden_pickaxe':1,'oak_planks':3,'stick':2} and len(takes)==6
        else:
            assert r['execution_valid'] and r['workshop_success'] and r['status']=='target_reached'
            assert inv=={'wooden_pickaxe':1,'wooden_axe':1}
            assert ingredients=={'oak_log':3,'oak_planks':12,'stick':4} and len(takes)==7
            reached=next(e['monotonic_s'] for e in events if e['event']=='target_reached')
            assert not any(e['event'] in ('inference_started','policy_dispatch_started') and e['monotonic_s']>reached for e in events)
            assert r['milestones'][-1]['event']=='completed' and r['milestones'][-1]['step']==r['steps']
        known={}
        for e in transitions:
            for key,recipe in e['info']['workshop']['recipes'].items():
                if recipe['known'] and recipe['server_known'] and key not in known:known[key]=e['step']
        cycles.extend(e['cycle_ms'] for e in events if e['event']=='cycle_timing')
        rows.append(dict(seed=seed,steps=r['steps'],status=r['status'],elapsed_control_s=r['elapsed_control_s'],
                         gui_decision_ms=r['gui_decision_ms'],control_cycle_ms=r['control_cycle_ms'],
                         device_vram=r['device_vram'],inventory=inv,table=r['table_position'],
                         output_clicks=len(takes),use_clicks=len(uses),recipe_first_known_steps=known,
                         jar=json.loads((folder/'runtime-jar-cache.json').read_text())['sha256']))
    return dict(verified=True,path=str(base),planned=state['planned'],rows=rows,control_cycle_ms=distribution(cycles),
                controller='workshop_gui',learned_policy_loaded=False,reinforcement_learning_updates=0,
                stop_at_stage=state.get('stop_at_stage'),api_calls=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('base',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.base);write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
